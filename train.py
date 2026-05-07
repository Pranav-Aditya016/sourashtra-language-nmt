"""
Training Script for Sourashtra Translation Model
==================================================
Full training loop with:
  - Teacher forcing annealing
  - Early stopping
  - Learning rate scheduling
  - Checkpoint saving
  - Periodic evaluation with sample translations
  - Comprehensive metric logging

Usage:
    python train.py
"""
import os
import sys
import time
import json
import math
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config
from data_loader import load_data
from model import build_model
from metrics import (
    evaluate_all, print_metrics,
    sentence_bleu, corpus_bleu, exact_match
)


def train_one_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio, grad_clip):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(loader, desc="  Training", leave=False):
        src = batch["source"].to(device)
        tgt = batch["target"].to(device)

        optimizer.zero_grad()
        output = model(src, tgt, teacher_forcing_ratio)

        # Reshape for loss: (batch * tgt_len, vocab_size)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        target = tgt[:, 1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, loader, criterion, device):
    """Validate model. Returns average loss."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Validating", leave=False):
            src = batch["source"].to(device)
            tgt = batch["target"].to(device)

            output = model(src, tgt, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            target = tgt[:, 1:].reshape(-1)

            loss = criterion(output, target)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def generate_translations(model, data, src_vocab, tgt_vocab, device, num_samples=10):
    """Generate translations on a subset and return source/reference/prediction triples."""
    model.eval()
    results = []

    indices = list(range(min(num_samples, len(data["source_text"]))))

    for idx in indices:
        src_text = data["source_text"][idx] if isinstance(data, dict) else data.dataset.sources[idx]
        tgt_text = data["target_text"][idx] if isinstance(data, dict) else data.dataset.targets[idx]

        # Encode source
        src_indices, _ = src_vocab.encode(src_text, 80)
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)

        # Translate
        decoded_indices, _ = model.translate(
            src_tensor, tgt_vocab.sos_idx, tgt_vocab.eos_idx, max_len=120
        )
        prediction = tgt_vocab.decode(decoded_indices)

        results.append({
            "source": src_text,
            "reference": tgt_text,
            "prediction": prediction,
        })

    return results


def run_full_evaluation(model, loader, src_vocab, tgt_vocab, device):
    """Run evaluation on an entire data loader. Returns metrics dict and predictions."""
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Translating", leave=False):
            for i in range(len(batch["source_text"])):
                src_text = batch["source_text"][i]
                tgt_text = batch["target_text"][i]

                # Encode and translate
                src_indices, _ = src_vocab.encode(src_text, 80)
                src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)

                decoded_indices, _ = model.translate(
                    src_tensor, tgt_vocab.sos_idx, tgt_vocab.eos_idx, max_len=120
                )
                prediction = tgt_vocab.decode(decoded_indices)

                references.append(tgt_text)
                hypotheses.append(prediction)

    metrics = evaluate_all(references, hypotheses)
    return metrics, references, hypotheses


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, src_vocab, tgt_vocab, filename):
    """Save model checkpoint with all necessary info."""
    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
        "config": {
            "embedding_dim": config.EMBEDDING_DIM,
            "encoder_hidden_dim": config.ENCODER_HIDDEN_DIM,
            "decoder_hidden_dim": config.DECODER_HIDDEN_DIM,
            "num_layers": config.NUM_LAYERS,
            "dropout": config.DROPOUT,
            "max_source_len": config.MAX_SOURCE_LEN,
            "max_target_len": config.MAX_TARGET_LEN,
        },
    }, filepath)

    # Save vocabularies alongside
    with open(os.path.join(config.CHECKPOINT_DIR, "src_vocab.pkl"), "wb") as f:
        pickle.dump(src_vocab, f)
    with open(os.path.join(config.CHECKPOINT_DIR, "tgt_vocab.pkl"), "wb") as f:
        pickle.dump(tgt_vocab, f)

    return filepath


def main():
    config = Config()
    config.ensure_dirs()
    config.summary()

    # ── Load Data ──────────────────────────────────────────────
    data = load_data(config)
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    src_vocab = data["src_vocab"]
    tgt_vocab = data["tgt_vocab"]

    # ── Build Model ────────────────────────────────────────────
    model = build_model(config, len(src_vocab), len(tgt_vocab))

    # ── Loss, Optimizer, Scheduler ─────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config.SCHEDULER_PATIENCE,
        factor=config.SCHEDULER_FACTOR
    )

    # ── Training Loop ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STARTING TRAINING")
    print("=" * 70)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    teacher_forcing_ratio = config.TEACHER_FORCING_START
    training_log = []

    start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()

        print(f"\n{'─'*70}")
        print(f"  Epoch {epoch}/{config.NUM_EPOCHS} | "
              f"TF ratio: {teacher_forcing_ratio:.3f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'─'*70}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            config.DEVICE, teacher_forcing_ratio, config.GRAD_CLIP
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, config.DEVICE)

        # Step scheduler
        scheduler.step(val_loss)

        # Anneal teacher forcing
        teacher_forcing_ratio = max(
            config.TEACHER_FORCING_END,
            teacher_forcing_ratio * config.TEACHER_FORCING_DECAY
        )

        epoch_time = time.time() - epoch_start

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # ── Log metrics ────────────────────────────────────────
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "teacher_forcing": teacher_forcing_ratio,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
        }

        # ── Sample translations every 5 epochs ────────────────
        if epoch % 5 == 0 or epoch == 1:
            print("\n  [INFO] Sample Translations:")
            samples = generate_translations(
                model,
                {"source_text": data["val_df"]["source"].tolist(),
                 "target_text": data["val_df"]["target"].tolist()},
                src_vocab, tgt_vocab, config.DEVICE, num_samples=5
            )
            for s in samples:
                print(f"    IN:  {s['source']}")
                print(f"    REF: {s['reference']}")
                print(f"    OUT: {s['prediction']}")
                match = "[OK]" if exact_match(s["reference"], s["prediction"]) else "[FAIL]"
                print(f"    {match}")
                print()

        # ── Early Stopping & Checkpointing ─────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            ckpt_path = save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                config, src_vocab, tgt_vocab, "best_model.pt"
            )
            print(f"  [OK] New best model saved! (val_loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  [WARN] No improvement for {epochs_without_improvement}/{config.PATIENCE} epochs")

        # Save periodic checkpoint
        if epoch % config.SAVE_EVERY == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                config, src_vocab, tgt_vocab, f"checkpoint_epoch_{epoch}.pt"
            )

        training_log.append(log_entry)

        # Early stopping
        if epochs_without_improvement >= config.PATIENCE:
            print(f"\n[STOP] Early stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - start_time
    print(f"\n[TIME] Total training time: {total_time/60:.1f} minutes")

    # ── Save Training Log ──────────────────────────────────────
    with open(os.path.join(config.RESULTS_DIR, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    # ── Final Evaluation on Test Set ───────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION ON TEST SET")
    print("=" * 70)

    # Load best model
    best_ckpt = torch.load(
        os.path.join(config.CHECKPOINT_DIR, "best_model.pt"),
        map_location=config.DEVICE, weights_only=False
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"  Loaded best model from epoch {best_ckpt['epoch']} "
          f"(val_loss: {best_ckpt['val_loss']:.4f})")

    test_metrics, test_refs, test_hyps = run_full_evaluation(
        model, test_loader, src_vocab, tgt_vocab, config.DEVICE
    )
    print_metrics(test_metrics, "TEST SET RESULTS")

    # Save results
    with open(os.path.join(config.RESULTS_DIR, "test_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save predictions
    predictions = [
        {"source": data["test_df"]["source"].iloc[i],
         "reference": ref, "prediction": hyp}
        for i, (ref, hyp) in enumerate(zip(test_refs, test_hyps))
    ]
    with open(os.path.join(config.RESULTS_DIR, "test_predictions.json"), "w") as f:
        json.dump(predictions[:100], f, indent=2, ensure_ascii=False)  # Save first 100

    # ── Show sample test translations ──────────────────────────
    print("\n[INFO] Sample Test Translations:")
    for p in predictions[:15]:
        match = "[OK]" if exact_match(p["reference"], p["prediction"]) else "[FAIL]"
        print(f"  IN:  {p['source']}")
        print(f"  REF: {p['reference']}")
        print(f"  OUT: {p['prediction']}  {match}")
        print()

    print("=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nSaved files:")
    print(f"  - checkpoints/best_model.pt")
    print(f"  - checkpoints/src_vocab.pkl")
    print(f"  - checkpoints/tgt_vocab.pkl")
    print(f"  - results/test_results.json")
    print(f"  - results/test_predictions.json")
    print(f"  - results/training_log.json")


if __name__ == "__main__":
    main()
