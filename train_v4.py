"""
Training Script V4 - Multilingual mT5 Fine-tuning
===================================================
Fine-tunes mT5-small on multilingual Sourashtra/Tamil/English data.

Key innovations over V3:
  - mT5-small (300M params) instead of T5-small (60M params)
  - Native Tamil understanding via multilingual pre-training
  - Multi-task: Sourashtra→English + Tamil→English + Sourashtra→Tamil
  - ~3x training data through cross-lingual augmentation

Usage:
    python train_v4.py
"""
import os
import sys
import json
import time
import numpy as np
import torch
from functools import partial

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

from config_v4 import ConfigV4
from data_loader_v4 import load_and_prepare_data


# =========================================================
# Tokenization
# =========================================================

def preprocess_function(examples, tokenizer, max_source_len, max_target_len):
    """Tokenize inputs and targets for mT5."""
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_source_len,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        examples["target_text"],
        max_length=max_target_len,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# =========================================================
# Metrics
# =========================================================

def compute_metrics(eval_preds, tokenizer):
    """Compute exact match and partial match during training."""
    predictions, labels = eval_preds

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Replace negative values with pad_token_id
    predictions = np.where(predictions >= 0, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean up
    decoded_preds = [pred.strip().lower() for pred in decoded_preds]
    decoded_labels = [label.strip().lower() for label in decoded_labels]

    # Exact match
    exact_matches = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p == l)
    exact_match_acc = exact_matches / len(decoded_labels) * 100

    # Partial match (word overlap)
    partial_matches = 0
    for p, l in zip(decoded_preds, decoded_labels):
        pred_words = set(p.split())
        label_words = set(l.split())
        if pred_words & label_words:
            partial_matches += 1
    partial_match_acc = partial_matches / len(decoded_labels) * 100

    return {
        "exact_match": round(exact_match_acc, 2),
        "partial_match": round(partial_match_acc, 2),
    }


# =========================================================
# Main Training
# =========================================================

def main():
    config = ConfigV4()
    config.ensure_dirs()
    config.summary()

    start_time = time.time()

    # ── Load Multilingual Data ─────────────────────────────────
    data = load_and_prepare_data(config)
    train_dataset = data["train_dataset"]
    val_dataset = data["val_dataset"]
    test_dataset = data["test_dataset"]

    # ── Load mT5 Model & Tokenizer ─────────────────────────────
    print(f"\n[MODEL] Loading {config.MODEL_NAME}...")

    # Use T5Tokenizer for T5, AutoTokenizer for mT5
    if "mt5" in config.MODEL_NAME.lower():
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)
    else:
        tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Set generation config
    model.generation_config.no_repeat_ngram_size = config.NO_REPEAT_NGRAM_SIZE
    model.generation_config.repetition_penalty = config.REPETITION_PENALTY
    model.generation_config.length_penalty = config.LENGTH_PENALTY

    # ── Tokenize Datasets ──────────────────────────────────────
    print("\n[PREP] Tokenizing datasets...")

    # Remove 'task' column if present (not needed for model input)
    cols_to_remove = train_dataset.column_names
    val_cols_to_remove = val_dataset.column_names

    preprocess_fn = partial(
        preprocess_function,
        tokenizer=tokenizer,
        max_source_len=config.MAX_SOURCE_LEN,
        max_target_len=config.MAX_TARGET_LEN,
    )

    train_tokenized = train_dataset.map(
        preprocess_fn, batched=True,
        remove_columns=cols_to_remove,
        desc="Tokenizing train (multilingual)",
    )
    val_tokenized = val_dataset.map(
        preprocess_fn, batched=True,
        remove_columns=val_cols_to_remove,
        desc="Tokenizing val",
    )
    test_tokenized = test_dataset.map(
        preprocess_fn, batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing test",
    )

    # ── Data Collator ──────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # ── Training Arguments ─────────────────────────────────────
    steps_per_epoch = len(train_tokenized) // (config.BATCH_SIZE * config.GRADIENT_ACCUMULATION)
    eval_steps = max(1, steps_per_epoch // config.EVAL_STEPS_PER_EPOCH)
    total_steps = steps_per_epoch * config.NUM_EPOCHS

    print(f"\n[INFO] Training examples: {len(train_tokenized):,}")
    print(f"[INFO] Steps per epoch:   {steps_per_epoch}")
    print(f"[INFO] Eval every:        {eval_steps} steps")
    print(f"[INFO] Total steps:       {total_steps}")

    warmup_steps = int(total_steps * config.WARMUP_RATIO) if config.WARMUP_RATIO > 0 else 0
    if warmup_steps > 0:
        print(f"[INFO] Warmup steps:      {warmup_steps}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.CHECKPOINT_DIR,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        num_train_epochs=config.NUM_EPOCHS,
        warmup_steps=warmup_steps,
        weight_decay=config.WEIGHT_DECAY,
        fp16=config.FP16 and torch.cuda.is_available(),
        bf16=config.BF16 and torch.cuda.is_available(),
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        predict_with_generate=True,
        generation_max_length=config.MAX_GENERATE_LEN,
        generation_num_beams=config.NUM_BEAMS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,
        seed=config.RANDOM_SEED,
        optim=config.OPTIM,
        lr_scheduler_type=config.LR_SCHEDULER,
        max_grad_norm=config.MAX_GRAD_NORM,
    )

    # ── Trainer ────────────────────────────────────────────────
    compute_metrics_fn = partial(compute_metrics, tokenizer=tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=config.PATIENCE,
        )],
    )

    # ── Train! ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STARTING TRAINING (V4 - Multilingual mT5 + Tamil)")
    print("  Model:   mT5-small (300M params, 101 languages)")
    print("  Tasks:   Sourashtra→English + Tamil→English + Sourashtra→Tamil")
    print(f"  Data:    {len(train_tokenized):,} examples ({data['split_info']['augmentation_ratio']:.1f}x)")
    print("=" * 70)

    train_result = trainer.train()

    # Save best model
    best_dir = os.path.join(config.CHECKPOINT_DIR, "best_model_v4")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    train_time = time.time() - start_time
    print(f"\n[TIME] Training time: {train_time/60:.1f} minutes")

    # ── Evaluate on Test Set (Sourashtra→English ONLY) ─────────
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION ON TEST SET")
    print("  (Sourashtra → English only — same test set as V1/V2/V3)")
    print("=" * 70)

    test_results = trainer.predict(test_tokenized)
    test_metrics = test_results.metrics
    print(f"\n  Test Exact Match:   {test_metrics.get('test_exact_match', 'N/A')}%")
    print(f"  Test Partial Match: {test_metrics.get('test_partial_match', 'N/A')}%")

    # ── Generate detailed predictions ──────────────────────────
    print("\n[PRED] Generating detailed test predictions...")
    predictions = test_results.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.where(predictions >= 0, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]

    test_df = data["test_df"]
    test_predictions = []
    exact_count = 0

    for i in range(len(decoded_preds)):
        src = test_df["source"].iloc[i]
        ref = test_df["target"].iloc[i]
        pred = decoded_preds[i]
        is_exact = ref.strip().lower() == pred.strip().lower()
        if is_exact:
            exact_count += 1
        test_predictions.append({
            "source": src,
            "reference": ref,
            "prediction": pred,
            "exact_match": is_exact,
            "category": test_df["category"].iloc[i] if "category" in test_df.columns else "",
        })

    # ── Full metrics using metrics.py ──────────────────────────
    from metrics import evaluate_all, print_metrics as print_all_metrics

    refs = [p["reference"] for p in test_predictions]
    hyps = [p["prediction"] for p in test_predictions]
    full_metrics = evaluate_all(refs, hyps)
    print_all_metrics(full_metrics, "V4 TEST RESULTS (mT5-small + Tamil multilingual)")

    # ── Save Results ───────────────────────────────────────────
    results = {
        "model": config.MODEL_NAME,
        "version": "V4",
        "approach": "Multilingual mT5 + Tamil cross-lingual transfer",
        "test_metrics": full_metrics,
        "hf_test_metrics": {k: v for k, v in test_metrics.items()},
        "training_time_minutes": round(train_time / 60, 1),
        "total_train_examples": data["split_info"]["total_train"],
        "augmentation_ratio": data["split_info"]["augmentation_ratio"],
        "tasks": {
            "sr_en": data["split_info"]["task_sr_en"],
            "ta_en": data["split_info"]["task_ta_en"],
            "sr_ta": data["split_info"]["task_sr_ta"],
        },
    }
    with open(os.path.join(config.RESULTS_DIR, "test_results_v4.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions
    with open(os.path.join(config.RESULTS_DIR, "test_predictions_v4.json"), "w",
              encoding="utf-8") as f:
        json.dump(test_predictions[:300], f, indent=2, ensure_ascii=False)

    # Save training log from trainer_state
    trainer_state_path = os.path.join(config.CHECKPOINT_DIR, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path) as f:
            state = json.load(f)
        training_log = []
        for entry in state.get("log_history", []):
            training_log.append(entry)
        with open(os.path.join(config.RESULTS_DIR, "training_log_v4.json"), "w") as f:
            json.dump(training_log, f, indent=2)

    # ── Sample Outputs ─────────────────────────────────────────
    print("\n  Sample Test Translations (Sourashtra → English):")
    for p in test_predictions[:20]:
        match = "[OK]" if p["exact_match"] else "[FAIL]"
        print(f"    IN:   {p['source']}")
        print(f"    REF:  {p['reference']}")
        print(f"    PRED: {p['prediction']}  {match}")
        if p.get("category"):
            print(f"    CAT:  {p['category']}")
        print()

    # ── Comparison with V1/V2/V3 ───────────────────────────────
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON (All Versions)")
    print("=" * 70)
    v1_em = 0.42
    v2_em = 2.56
    v3_em = 6.01
    v3_hybrid = 7.47
    v4_em = full_metrics["exact_match_accuracy"]
    print(f"  V1 (Char GRU Seq2Seq):      {v1_em:.2f}% exact match")
    print(f"  V2 (Transformer+BPE):       {v2_em:.2f}% exact match")
    print(f"  V3 (T5-small, EN only):     {v3_em:.2f}% exact match")
    print(f"  V3 Hybrid (T5+Retrieval):   {v3_hybrid:.2f}% exact match")
    print(f"  V4 (mT5 + Tamil):           {v4_em:.2f}% exact match  {'<-- NEW' if v4_em > v3_hybrid else ''}")
    print()
    if v3_em > 0:
        print(f"  Improvement V3→V4:  {v4_em/v3_em:.1f}x")
    if v1_em > 0:
        print(f"  Improvement V1→V4:  {v4_em/v1_em:.1f}x")
    print(f"\n  BLEU:  {full_metrics['corpus_bleu']:.2f}")
    print(f"  chrF:  {full_metrics['avg_chrf']:.2f}")
    print("=" * 70)

    print(f"\n  TRAINING COMPLETE!")
    print(f"  Best model saved to: {best_dir}")
    print(f"  Results saved to: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
