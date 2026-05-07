"""
Training Script V5-Reverse — English/Tamil → Sourashtra (ByT5-small)
=====================================================================
Fine-tunes google/byt5-small for REVERSE translation:
  English → Sourashtra (Roman)
  Tamil → Sourashtra (Roman)

Same architecture and hyperparameters as V5, just reversed direction.

Usage:
    python train_v5_reverse.py
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
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

from config_v5_reverse import ConfigV5Reverse
from data_loader_v5_reverse import load_and_prepare_data


# =========================================================
# Tokenization
# =========================================================

def preprocess_function(examples, tokenizer, max_source_len, max_target_len):
    """Tokenize inputs and targets for ByT5 (byte-level)."""
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

    predictions = np.where(predictions >= 0, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip().lower() for p in decoded_preds]
    decoded_labels = [l.strip().lower() for l in decoded_labels]

    exact = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p == l)
    exact_acc = exact / len(decoded_labels) * 100

    partial = 0
    for p, l in zip(decoded_preds, decoded_labels):
        if set(p.split()) & set(l.split()):
            partial += 1
    partial_acc = partial / len(decoded_labels) * 100

    return {
        "exact_match": round(exact_acc, 2),
        "partial_match": round(partial_acc, 2),
    }


# =========================================================
# Main Training
# =========================================================

def main():
    config = ConfigV5Reverse()
    config.ensure_dirs()
    config.summary()

    start_time = time.time()

    # ── Load Data ──
    data = load_and_prepare_data(config)
    train_dataset = data["train_dataset"]
    val_dataset = data["val_dataset"]
    test_dataset = data["test_dataset"]

    # ── Load ByT5 Model & Tokenizer ──
    print(f"\n[MODEL] Loading {config.MODEL_NAME}...")
    print("  ByT5 uses byte-level tokenization — no SentencePiece needed")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Vocab size:           {tokenizer.vocab_size} (byte-level + special)")

    model.generation_config.length_penalty = config.LENGTH_PENALTY

    if config.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: ENABLED")

    # ── Tokenize Datasets ──
    print("\n[PREP] Tokenizing datasets (byte-level)...")

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

    # Show byte-level tokenization example
    sample_text = "translate English to Sourashtra: milk"
    sample_tokens = tokenizer(sample_text)
    print(f"\n  Tokenization example:")
    print(f"    Input:  '{sample_text}'")
    print(f"    Tokens: {len(sample_tokens['input_ids'])} byte-level IDs")
    print(f"    IDs:    {sample_tokens['input_ids'][:20]}...")

    # ── Data Collator ──
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100,
    )

    # ── Training Arguments ──
    steps_per_epoch = len(train_tokenized) // (config.BATCH_SIZE * config.GRADIENT_ACCUMULATION)
    eval_steps = max(1, steps_per_epoch // config.EVAL_STEPS_PER_EPOCH)
    total_steps = steps_per_epoch * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO) if config.WARMUP_RATIO > 0 else 0

    print(f"\n[INFO] Training examples: {len(train_tokenized):,}")
    print(f"[INFO] Steps per epoch:   {steps_per_epoch}")
    print(f"[INFO] Eval every:        {eval_steps} steps")
    print(f"[INFO] Total steps:       {total_steps}")
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
        logging_steps=25,
        report_to="none",
        dataloader_num_workers=0,
        seed=config.RANDOM_SEED,
        optim=config.OPTIM,
        lr_scheduler_type=config.LR_SCHEDULER,
        max_grad_norm=config.MAX_GRAD_NORM,
    )

    # ── Trainer ──
    compute_metrics_fn = partial(compute_metrics, tokenizer=tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.PATIENCE)],
    )

    # ── Train! ──
    print("\n" + "=" * 70)
    print("  STARTING TRAINING (V5-REVERSE: English/Tamil → Sourashtra)")
    print("  Model:   ByT5-small (300M params, byte-level)")
    print("  Tasks:   English→Sourashtra + Tamil→Sourashtra")
    print(f"  Data:    {len(train_tokenized):,} examples")
    print(f"  Epochs:  {config.NUM_EPOCHS}")
    print("=" * 70)

    # Resume from checkpoint if available
    last_checkpoint = None
    if os.path.isdir(config.CHECKPOINT_DIR):
        import glob
        checkpoints = sorted(glob.glob(os.path.join(config.CHECKPOINT_DIR, "checkpoint-*")),
                              key=lambda x: int(x.split("-")[-1]))
        if checkpoints:
            last_checkpoint = checkpoints[-1]
            print(f"  [RESUME] Found checkpoint: {last_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save best model
    best_dir = os.path.join(config.CHECKPOINT_DIR, "best_model_v5_reverse")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    train_time = time.time() - start_time
    print(f"\n[TIME] Training time: {train_time/60:.1f} minutes")

    # ── Evaluate on Test Set ──
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION ON TEST SET (English → Sourashtra)")
    print("=" * 70)

    test_results = trainer.predict(test_tokenized)
    test_metrics = test_results.metrics
    print(f"\n  Test Exact Match:   {test_metrics.get('test_exact_match', 'N/A')}%")
    print(f"  Test Partial Match: {test_metrics.get('test_partial_match', 'N/A')}%")

    # ── Detailed Predictions ──
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
        english = test_df["target"].iloc[i]       # English (input)
        sourashtra = test_df["source"].iloc[i]     # Sourashtra (reference)
        pred = decoded_preds[i]
        is_exact = sourashtra.strip().lower() == pred.strip().lower()
        if is_exact:
            exact_count += 1
        test_predictions.append({
            "english_input": english,
            "sourashtra_reference": sourashtra,
            "prediction": pred,
            "exact_match": is_exact,
            "category": test_df["category"].iloc[i] if "category" in test_df.columns else "",
        })

    # Full metrics
    from metrics import evaluate_all, print_metrics as print_all_metrics

    refs = [p["sourashtra_reference"] for p in test_predictions]
    hyps = [p["prediction"] for p in test_predictions]
    full_metrics = evaluate_all(refs, hyps)
    print_all_metrics(full_metrics, "V5-REVERSE TEST RESULTS (English/Tamil → Sourashtra)")

    # ── Save Results ──
    results = {
        "model": config.MODEL_NAME,
        "version": "V5-Reverse",
        "approach": "ByT5-small byte-level — English/Tamil → Sourashtra (Roman)",
        "test_metrics": full_metrics,
        "hf_test_metrics": {k: v for k, v in test_metrics.items()},
        "training_time_minutes": round(train_time / 60, 1),
        "total_train_examples": data["split_info"]["total_train"],
        "augmentation_ratio": data["split_info"]["augmentation_ratio"],
    }
    with open(os.path.join(config.RESULTS_DIR, "test_results_v5_reverse.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(config.RESULTS_DIR, "test_predictions_v5_reverse.json"), "w",
              encoding="utf-8") as f:
        json.dump(test_predictions, f, indent=2, ensure_ascii=False)

    # ── Sample Outputs ──
    print("\n  Sample Test Translations (English → Sourashtra):")
    for p in test_predictions[:20]:
        match = "[OK]" if p["exact_match"] else "[FAIL]"
        print(f"    EN:   {p['english_input']}")
        print(f"    REF:  {p['sourashtra_reference']}")
        print(f"    PRED: {p['prediction']}  {match}")
        print()

    print(f"\n  Training complete! Model saved to: {best_dir}")
    print(f"  Results saved to: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
