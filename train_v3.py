"""
Training Script V3 - T5 Fine-tuning for Sourashtra Translation
================================================================
Fine-tunes T5-small on augmented Sourashtra → English data.

Key features:
  - HuggingFace Seq2SeqTrainer with proper eval
  - Mixed precision (FP16) for speed
  - Early stopping based on exact match accuracy
  - Beam search evaluation
  - Saves best model automatically

Usage:
    python train_v3.py
"""
import os
import sys
import json
import time
import numpy as np
import torch
from functools import partial

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

from config_v3 import ConfigV3
from data_loader_v3 import load_and_prepare_data


# =========================================================
# Tokenization
# =========================================================

def preprocess_function(examples, tokenizer, max_source_len, max_target_len):
    """Tokenize inputs and targets for T5."""
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_source_len,
        truncation=True,
        padding=False,  # DataCollator will handle padding
    )

    # Tokenize targets
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
    """Compute exact match and other metrics during training."""
    predictions, labels = eval_preds

    # Decode predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Replace any negative values (e.g. -100 padding) with pad_token_id
    predictions = np.where(predictions >= 0, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in labels (padding) with pad_token_id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean up
    decoded_preds = [pred.strip().lower() for pred in decoded_preds]
    decoded_labels = [label.strip().lower() for label in decoded_labels]

    # Exact match
    exact_matches = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p == l)
    exact_match_acc = exact_matches / len(decoded_labels) * 100

    # Partial match (at least one word overlap)
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
    config = ConfigV3()
    config.ensure_dirs()
    config.summary()

    start_time = time.time()

    # ── Load Data ──────────────────────────────────────────────
    data = load_and_prepare_data(config)
    train_dataset = data["train_dataset"]
    val_dataset = data["val_dataset"]
    test_dataset = data["test_dataset"]

    # ── Load T5 Model & Tokenizer ──────────────────────────────
    print(f"\n[MODEL] Loading {config.MODEL_NAME}...")
    tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Set generation config to prevent repetition
    model.generation_config.no_repeat_ngram_size = config.NO_REPEAT_NGRAM_SIZE
    model.generation_config.repetition_penalty = config.REPETITION_PENALTY
    model.generation_config.length_penalty = config.LENGTH_PENALTY

    # ── Tokenize Datasets ──────────────────────────────────────
    print("\n[PREP] Tokenizing datasets...")
    preprocess_fn = partial(
        preprocess_function,
        tokenizer=tokenizer,
        max_source_len=config.MAX_SOURCE_LEN,
        max_target_len=config.MAX_TARGET_LEN,
    )

    train_tokenized = train_dataset.map(
        preprocess_fn, batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    val_tokenized = val_dataset.map(
        preprocess_fn, batched=True,
        remove_columns=val_dataset.column_names,
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

    print(f"\n[INFO] Steps per epoch: {steps_per_epoch}")
    print(f"[INFO] Eval every: {eval_steps} steps")
    print(f"[INFO] Total steps: {total_steps}")

    warmup_steps = int(total_steps * config.WARMUP_RATIO) if config.WARMUP_RATIO > 0 else 0

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
        predict_with_generate=True,
        generation_max_length=config.MAX_GENERATE_LEN,
        generation_num_beams=config.NUM_BEAMS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",  # Disable wandb etc.
        dataloader_num_workers=0,
        seed=config.RANDOM_SEED,
        optim=config.OPTIM,
        lr_scheduler_type=config.LR_SCHEDULER,
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
    print("  STARTING TRAINING (V3 - T5 Fine-tuning)")
    print("=" * 70)

    train_result = trainer.train()

    # Save best model
    trainer.save_model(os.path.join(config.CHECKPOINT_DIR, "best_model_v3"))
    tokenizer.save_pretrained(os.path.join(config.CHECKPOINT_DIR, "best_model_v3"))

    train_time = time.time() - start_time
    print(f"\n[TIME] Training time: {train_time/60:.1f} minutes")

    # ── Evaluate on Test Set ───────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION ON TEST SET")
    print("=" * 70)

    test_results = trainer.predict(test_tokenized)
    test_metrics = test_results.metrics
    print(f"\n  Test Exact Match: {test_metrics.get('test_exact_match', 'N/A')}%")
    print(f"  Test Partial Match: {test_metrics.get('test_partial_match', 'N/A')}%")

    # ── Generate detailed predictions ──────────────────────────
    print("\n[PRED] Generating detailed test predictions...")
    predictions = test_results.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Replace any negative values with pad_token_id before decoding
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

    # ── Full metrics using V1/V2 metrics.py ────────────────────
    from metrics import evaluate_all, print_metrics as print_all_metrics

    refs = [p["reference"] for p in test_predictions]
    hyps = [p["prediction"] for p in test_predictions]
    full_metrics = evaluate_all(refs, hyps)
    print_all_metrics(full_metrics, "V3 TEST SET RESULTS (T5-small fine-tuned)")

    # ── Save Results ───────────────────────────────────────────
    results = {
        "model": config.MODEL_NAME,
        "test_metrics": full_metrics,
        "hf_test_metrics": {k: v for k, v in test_metrics.items()},
        "training_time_minutes": round(train_time / 60, 1),
        "total_train_examples": data["split_info"]["total_train"],
        "augmentation_ratio": data["split_info"]["augmentation_ratio"],
    }
    with open(os.path.join(config.RESULTS_DIR, "test_results_v3.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions (first 300)
    with open(os.path.join(config.RESULTS_DIR, "test_predictions_v3.json"), "w",
              encoding="utf-8") as f:
        json.dump(test_predictions[:300], f, indent=2, ensure_ascii=False)

    # ── Sample Outputs ─────────────────────────────────────────
    print("\n  Sample Test Translations:")
    for p in test_predictions[:20]:
        match = "[OK]" if p["exact_match"] else "[FAIL]"
        print(f"    IN:   {p['source']}")
        print(f"    REF:  {p['reference']}")
        print(f"    PRED: {p['prediction']}  {match}")
        if p.get("category"):
            print(f"    CAT:  {p['category']}")
        print()

    # ── Comparison with V1/V2 ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)
    v1_em = 0.42
    v2_em = 2.56
    v3_em = full_metrics["exact_match_accuracy"]
    print(f"  V1 (Char Seq2Seq):    {v1_em:.2f}% exact match")
    print(f"  V2 (Transformer+BPE): {v2_em:.2f}% exact match")
    print(f"  V3 (T5 fine-tuned):   {v3_em:.2f}% exact match")
    if v2_em > 0:
        print(f"  Improvement V2→V3:    {v3_em/v2_em:.1f}x")
    print(f"  BLEU:    {full_metrics['corpus_bleu']:.2f}")
    print(f"  chrF:    {full_metrics['avg_chrf']:.2f}")
    print("=" * 70)

    print("\n  TRAINING COMPLETE!")
    print(f"  Best model saved to: {config.CHECKPOINT_DIR}/best_model_v3/")
    print(f"  Results saved to: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
