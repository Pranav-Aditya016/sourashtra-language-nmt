"""
Training Script V2 - Transformer + BPE for Sourashtra Translation
===================================================================
Improvements over V1:
  - BPE subword tokenization (learns morphology)
  - Transformer architecture (better attention mechanism)
  - Noam (inverse sqrt) learning rate schedule with warmup
  - Label smoothing (better generalization)
  - Beam search decoding during evaluation

Usage:
    python train_v2.py
"""
import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
from tqdm import tqdm

from config_v2 import ConfigV2
from data_loader_v2 import load_data_v2
from model_v2 import build_model_v2
from metrics import evaluate_all, print_metrics, exact_match


# =========================================================
# Learning Rate Scheduler (Noam / Inverse Sqrt Warmup)
# =========================================================

class NoamScheduler:
    """
    Noam learning rate schedule from 'Attention Is All You Need'.
    LR = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    Scaled to peak at `base_lr`.
    """

    def __init__(self, optimizer, d_model, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_num = 0
        # Scale factor to hit base_lr at warmup_steps
        self._scale = base_lr * (warmup_steps ** 0.5)

    def step(self):
        self.step_num += 1
        lr = self._scale * min(
            self.step_num ** (-0.5),
            self.step_num * (self.warmup_steps ** (-1.5))
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_lr(self):
        if self.step_num == 0:
            return 0
        return self._scale * min(
            self.step_num ** (-0.5),
            self.step_num * (self.warmup_steps ** (-1.5))
        )


# =========================================================
# Label Smoothing Loss
# =========================================================

class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing."""

    def __init__(self, vocab_size, pad_idx=0, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        """
        Args:
            logits: (batch * seq_len, vocab_size)
            target: (batch * seq_len,)
        """
        logits = logits.contiguous().view(-1, self.vocab_size)
        target = target.contiguous().view(-1)

        # Create smooth distribution
        smooth_dist = torch.full_like(logits, self.smoothing / (self.vocab_size - 2))
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_dist[:, self.pad_idx] = 0

        # Mask padding
        pad_mask = target == self.pad_idx
        smooth_dist[pad_mask] = 0

        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(smooth_dist * log_probs).sum(dim=-1)

        # Average over non-padding tokens
        non_pad = (~pad_mask).sum()
        if non_pad == 0:
            return loss.sum() * 0  # Avoid division by zero
        return loss.sum() / non_pad


# =========================================================
# Training Functions
# =========================================================

def train_one_epoch(model, loader, optimizer, criterion, scheduler, device, grad_clip):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(loader, desc="  Training", leave=False):
        src = batch["source"].to(device)
        tgt = batch["target"].to(device)

        # Teacher forcing: input is tgt[:-1], target is tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_input)

        # Reshape for loss
        logits_flat = logits.reshape(-1, logits.size(-1))
        target_flat = tgt_output.reshape(-1)

        loss = criterion(logits_flat, target_flat)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Validating", leave=False):
            src = batch["source"].to(device)
            tgt = batch["target"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = tgt_output.reshape(-1)

            loss = criterion(logits_flat, target_flat)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def run_translation_evaluation(model, loader, src_tok, tgt_tok, device, use_beam=False):
    """Run full evaluation with greedy or beam search decoding."""
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Translating", leave=False):
            for i in range(len(batch["source_text"])):
                src_text = batch["source_text"][i]
                tgt_text = batch["target_text"][i]

                src_ids, _ = src_tok.encode(src_text, 64, add_sos=False, add_eos=True)
                src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

                if use_beam:
                    decoded = model.translate_beam(
                        src_tensor, tgt_tok.SOS_ID, tgt_tok.EOS_ID, max_len=64, beam_width=5
                    )
                else:
                    decoded = model.translate(
                        src_tensor, tgt_tok.SOS_ID, tgt_tok.EOS_ID, max_len=64
                    )

                prediction = tgt_tok.decode(decoded)
                references.append(tgt_text)
                hypotheses.append(prediction)

    metrics = evaluate_all(references, hypotheses)
    return metrics, references, hypotheses


def save_checkpoint_v2(model, optimizer, scheduler, epoch, val_loss, config, filename):
    """Save model checkpoint."""
    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_step": scheduler.step_num,
        "val_loss": val_loss,
    }, filepath)
    return filepath


# =========================================================
# Main Training Loop
# =========================================================

def main():
    config = ConfigV2()
    config.ensure_dirs()
    config.summary()

    # -- Load Data & Build Tokenizers --------------------------------
    data = load_data_v2(config)
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    src_tok = data["src_tokenizer"]
    tgt_tok = data["tgt_tokenizer"]

    # -- Build Model -------------------------------------------------
    model = build_model_v2(config, src_tok.vocab_size(), tgt_tok.vocab_size())

    # -- Loss, Optimizer, Scheduler ----------------------------------
    criterion = LabelSmoothingLoss(
        vocab_size=tgt_tok.vocab_size(),
        pad_idx=tgt_tok.PAD_ID,
        smoothing=config.LABEL_SMOOTHING,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = NoamScheduler(
        optimizer, config.D_MODEL, config.WARMUP_STEPS, config.LEARNING_RATE
    )

    # -- Training Loop -----------------------------------------------
    print("\n" + "=" * 70)
    print("  STARTING TRAINING (V2 - Transformer + BPE)")
    print("=" * 70)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    training_log = []
    start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()

        print(f"\n{'---'*23}")
        print(f"  Epoch {epoch}/{config.NUM_EPOCHS} | "
              f"LR: {scheduler.get_lr():.6f}")
        print(f"{'---'*23}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scheduler, config.DEVICE, config.GRAD_CLIP
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, config.DEVICE)
        epoch_time = time.time() - epoch_start

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # Log
        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "lr": scheduler.get_lr(),
            "time": round(epoch_time, 1),
        }

        # -- Sample Translations every EVAL_EVERY epochs ----------------
        if epoch % config.EVAL_EVERY == 0 or epoch == 1:
            print("\n  Sample Translations (greedy):")
            model.eval()
            for i in range(min(8, len(data["val_df"]))):
                src_text = data["val_df"]["source"].iloc[i]
                tgt_text = data["val_df"]["target"].iloc[i]

                src_ids, _ = src_tok.encode(src_text, 64, add_sos=False, add_eos=True)
                src_tensor = torch.tensor([src_ids], dtype=torch.long, device=config.DEVICE)
                decoded = model.translate(src_tensor, tgt_tok.SOS_ID, tgt_tok.EOS_ID, max_len=64)
                pred = tgt_tok.decode(decoded)

                match = "[OK]" if exact_match(tgt_text, pred) else "[FAIL]"
                print(f"    IN:  {src_text}")
                print(f"    REF: {tgt_text}")
                print(f"    OUT: {pred}  {match}")
                print()

            # Quick BLEU on val subset
            val_subset_refs = data["val_df"]["target"].head(200).tolist()
            val_subset_hyps = []
            model.eval()
            with torch.no_grad():
                for src_text in data["val_df"]["source"].head(200):
                    src_ids, _ = src_tok.encode(src_text, 64, add_sos=False, add_eos=True)
                    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=config.DEVICE)
                    decoded = model.translate(src_tensor, tgt_tok.SOS_ID, tgt_tok.EOS_ID, max_len=64)
                    val_subset_hyps.append(tgt_tok.decode(decoded))

            quick_metrics = evaluate_all(val_subset_refs, val_subset_hyps)
            print(f"  Quick Val Metrics (200 samples):")
            print(f"    BLEU: {quick_metrics['corpus_bleu']:.2f} | "
                  f"chrF: {quick_metrics['avg_chrf']:.2f} | "
                  f"EM: {quick_metrics['exact_match_accuracy']:.1f}%")
            log_entry["val_bleu"] = quick_metrics["corpus_bleu"]
            log_entry["val_exact_match"] = quick_metrics["exact_match_accuracy"]

        # -- Checkpointing & Early Stopping ----------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint_v2(model, optimizer, scheduler, epoch, val_loss,
                              config, "best_model_v2.pt")
            print(f"  [OK] New best model! (val_loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  [WARN] No improvement for {epochs_no_improve}/{config.PATIENCE} epochs")

        if epoch % config.SAVE_EVERY == 0:
            save_checkpoint_v2(model, optimizer, scheduler, epoch, val_loss,
                              config, f"checkpoint_v2_epoch_{epoch}.pt")

        training_log.append(log_entry)

        if epochs_no_improve >= config.PATIENCE:
            print(f"\n[STOP] Early stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time
    print(f"\n[TIME] Total training time: {total_time/60:.1f} minutes")

    # -- Save Training Log -------------------------------------------
    with open(os.path.join(config.RESULTS_DIR, "training_log_v2.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    # -- Final Test Evaluation ---------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION ON TEST SET")
    print("=" * 70)

    # Load best model
    ckpt = torch.load(
        os.path.join(config.CHECKPOINT_DIR, "best_model_v2.pt"),
        map_location=config.DEVICE, weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded best model from epoch {ckpt['epoch']} (val_loss: {ckpt['val_loss']:.4f})")

    # Greedy decoding evaluation
    print("\n  [Greedy Decoding]")
    greedy_metrics, greedy_refs, greedy_hyps = run_translation_evaluation(
        model, test_loader, src_tok, tgt_tok, config.DEVICE, use_beam=False
    )
    print_metrics(greedy_metrics, "TEST SET - Greedy Decoding")

    # Beam search evaluation
    print("\n  [Beam Search (beam=5)]")
    beam_metrics, beam_refs, beam_hyps = run_translation_evaluation(
        model, test_loader, src_tok, tgt_tok, config.DEVICE, use_beam=True
    )
    print_metrics(beam_metrics, "TEST SET - Beam Search (k=5)")

    # -- Save Results ------------------------------------------------
    results = {
        "greedy": greedy_metrics,
        "beam_search": beam_metrics,
        "best_epoch": ckpt["epoch"],
        "best_val_loss": ckpt["val_loss"],
        "total_training_time_minutes": round(total_time / 60, 1),
    }
    with open(os.path.join(config.RESULTS_DIR, "test_results_v2.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions
    predictions = []
    for i in range(min(200, len(greedy_refs))):
        predictions.append({
            "source": data["test_df"]["source"].iloc[i],
            "reference": greedy_refs[i],
            "greedy": greedy_hyps[i],
            "beam": beam_hyps[i],
        })
    with open(os.path.join(config.RESULTS_DIR, "test_predictions_v2.json"), "w",
              encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # -- Sample outputs -----------------------------------------------
    print("\n  Sample Test Translations:")
    for p in predictions[:15]:
        gm = "[OK]" if exact_match(p["reference"], p["greedy"]) else "[FAIL]"
        bm = "[OK]" if exact_match(p["reference"], p["beam"]) else "[FAIL]"
        print(f"  IN:     {p['source']}")
        print(f"  REF:    {p['reference']}")
        print(f"  GREEDY: {p['greedy']}  {gm}")
        print(f"  BEAM:   {p['beam']}  {bm}")
        print()

    print("=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nSaved files in {config.RESULTS_DIR}/:")
    print(f"  - test_results_v2.json")
    print(f"  - test_predictions_v2.json")
    print(f"  - training_log_v2.json")
    print(f"  - checkpoints_v2/best_model_v2.pt")


if __name__ == "__main__":
    main()
