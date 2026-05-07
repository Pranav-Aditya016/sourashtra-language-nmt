"""
Standalone Evaluation + Visualization Script
==============================================
Run after training to:
  - Load best model and evaluate on test set
  - Generate detailed per-category metrics
  - Create publication-quality plots
  - Visualize attention maps
  - Export results for the research paper

Usage:
    python evaluate.py
"""
import os
import sys
import json
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

from config import Config
from data_loader import load_data, CharVocab
from model import build_model
from metrics import (
    evaluate_all, print_metrics, corpus_bleu, sentence_bleu,
    chrf_score, character_error_rate, word_error_rate, exact_match
)


def load_trained_model(config):
    """Load the best trained model and vocabularies."""
    # Load vocabularies
    with open(os.path.join(config.CHECKPOINT_DIR, "src_vocab.pkl"), "rb") as f:
        src_vocab = pickle.load(f)
    with open(os.path.join(config.CHECKPOINT_DIR, "tgt_vocab.pkl"), "rb") as f:
        tgt_vocab = pickle.load(f)

    # Build model
    model = build_model(config, len(src_vocab), len(tgt_vocab))

    # Load weights
    ckpt = torch.load(
        os.path.join(config.CHECKPOINT_DIR, "best_model.pt"),
        map_location=config.DEVICE, weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  Loaded model from epoch {ckpt['epoch']} (val_loss: {ckpt['val_loss']:.4f})")
    return model, src_vocab, tgt_vocab


def translate_batch(model, sources, src_vocab, tgt_vocab, device, max_len=120):
    """Translate a batch of source texts."""
    predictions = []
    for src_text in tqdm(sources, desc="  Translating", leave=False):
        src_indices, _ = src_vocab.encode(src_text, 80)
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
        decoded, _ = model.translate(src_tensor, tgt_vocab.sos_idx, tgt_vocab.eos_idx, max_len)
        predictions.append(tgt_vocab.decode(decoded))
    return predictions


def evaluate_by_category(model, config, src_vocab, tgt_vocab):
    """Evaluate model performance broken down by semantic category."""
    # Load unified dataset to get categories
    udf = pd.read_csv(config.UNIFIED_FILE)
    trans_df = pd.read_csv(config.TRANSLATION_FILE)

    # Merge to get categories
    udf_slim = udf[["roman_readable", "meaning_english", "category"]].dropna(subset=["roman_readable", "meaning_english", "category"])
    udf_slim = udf_slim.rename(columns={"roman_readable": "source", "meaning_english": "target"})
    udf_slim["source"] = udf_slim["source"].str.strip().str.lower()
    udf_slim["target"] = udf_slim["target"].str.strip().str.lower()
    trans_df["source"] = trans_df["source"].str.strip().str.lower()
    trans_df["target"] = trans_df["target"].str.strip().str.lower()

    merged = trans_df.merge(udf_slim[["source", "category"]], on="source", how="left")
    merged["category"] = merged["category"].fillna("unknown")

    # Sample per category (up to 200 per category for speed)
    category_results = {}
    categories = merged["category"].unique()

    print(f"\n[STATS] Evaluating {len(categories)} categories...")

    for cat in tqdm(sorted(categories), desc="  Categories"):
        cat_df = merged[merged["category"] == cat]
        if len(cat_df) < 3:
            continue

        sample = cat_df.head(200)
        preds = translate_batch(
            model, sample["source"].tolist(), src_vocab, tgt_vocab, config.DEVICE
        )
        metrics = evaluate_all(sample["target"].tolist(), preds)
        metrics["count"] = len(sample)
        category_results[cat] = metrics

    return category_results


def plot_training_curves(config):
    """Plot training and validation loss curves."""
    log_path = os.path.join(config.RESULTS_DIR, "training_log.json")
    if not os.path.exists(log_path):
        print("  [WARN] No training log found, skipping loss curves")
        return

    with open(log_path) as f:
        log = json.load(f)

    epochs = [e["epoch"] for e in log]
    train_loss = [e["train_loss"] for e in log]
    val_loss = [e["val_loss"] for e in log]
    lr = [e["learning_rate"] for e in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_loss, "r-", label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Learning rate
    ax2.plot(epochs, lr, "g-", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Learning Rate", fontsize=12)
    ax2.set_title("Learning Rate Schedule", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] Saved: training_curves.png")


def plot_category_performance(category_results, config):
    """Bar chart of BLEU scores by category."""
    if not category_results:
        return

    # Sort by BLEU score
    sorted_cats = sorted(category_results.items(), key=lambda x: x[1]["corpus_bleu"], reverse=True)

    # Take top 25 and bottom 10
    top = sorted_cats[:25]

    cats = [c[0] for c in top]
    bleu = [c[1]["corpus_bleu"] for c in top]
    exact = [c[1]["exact_match_accuracy"] for c in top]
    counts = [c[1]["count"] for c in top]

    fig, ax = plt.subplots(figsize=(14, 8))
    x = range(len(cats))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], bleu, width, label="BLEU", color="#2196F3", alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], exact, width, label="Exact Match %", color="#4CAF50", alpha=0.8)

    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Translation Quality by Category (Top 25)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add count labels
    for i, count in enumerate(counts):
        ax.annotate(f"n={count}", (i, max(bleu[i], exact[i]) + 2),
                    ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "category_performance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] Saved: category_performance.png")


def plot_attention_heatmap(model, src_text, tgt_text, src_vocab, tgt_vocab, config):
    """Plot attention heatmap for a single translation."""
    src_indices, _ = src_vocab.encode(src_text, 80)
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(config.DEVICE)

    decoded_indices, attn_matrix = model.translate(
        src_tensor, tgt_vocab.sos_idx, tgt_vocab.eos_idx, max_len=120
    )
    prediction = tgt_vocab.decode(decoded_indices)

    if attn_matrix.shape[0] <= 1:
        return

    # Get actual characters (remove padding)
    src_chars = list(src_text.lower()) + ["<EOS>"]
    pred_chars = [tgt_vocab.idx2char.get(idx, "?") for idx in decoded_indices]

    # Trim attention to actual lengths
    attn = attn_matrix[:len(pred_chars), :len(src_chars)].numpy()

    fig, ax = plt.subplots(figsize=(max(8, len(src_chars) * 0.4), max(4, len(pred_chars) * 0.4)))
    sns.heatmap(attn, xticklabels=src_chars, yticklabels=pred_chars,
                cmap="YlOrRd", ax=ax, cbar_kws={"label": "Attention Weight"})
    ax.set_xlabel("Source (Sourashtra)", fontsize=11)
    ax.set_ylabel("Target (English)", fontsize=11)
    ax.set_title(f"Attention: '{src_text}' → '{prediction}'\n(Reference: '{tgt_text}')", fontsize=12)
    plt.tight_layout()

    safe_name = src_text.replace(" ", "_").replace("/", "_")[:30]
    filename = f"attention_{safe_name}.png"
    plt.savefig(os.path.join(config.RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    return filename


def plot_error_distribution(references, hypotheses, config):
    """Plot distribution of error rates."""
    cer_scores = [character_error_rate(r, h) for r, h in zip(references, hypotheses)]
    bleu_scores = [sentence_bleu(r, h) for r, h in zip(references, hypotheses)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # CER distribution
    ax1.hist(cer_scores, bins=50, color="#FF5722", alpha=0.7, edgecolor="white")
    ax1.axvline(np.mean(cer_scores), color="black", linestyle="--", label=f"Mean: {np.mean(cer_scores):.1f}%")
    ax1.set_xlabel("Character Error Rate (%)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("CER Distribution", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # BLEU distribution
    ax2.hist(bleu_scores, bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
    ax2.axvline(np.mean(bleu_scores), color="black", linestyle="--", label=f"Mean: {np.mean(bleu_scores):.1f}")
    ax2.set_xlabel("Sentence BLEU Score", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("BLEU Score Distribution", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "error_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] Saved: error_distribution.png")


def generate_paper_table(test_metrics, category_results, config):
    """Generate LaTeX-formatted tables for the research paper."""
    lines = []

    # Main results table
    lines.append("% ──── Main Results Table ────")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Evaluation Results on Test Set}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\begin{tabular}{lc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Metric} & \\textbf{Score} \\\\")
    lines.append("\\midrule")
    lines.append(f"Corpus BLEU & {test_metrics['corpus_bleu']:.2f} \\\\")
    lines.append(f"Avg. Sentence BLEU & {test_metrics['avg_sentence_bleu']:.2f} \\\\")
    lines.append(f"chrF & {test_metrics['avg_chrf']:.2f} \\\\")
    lines.append(f"CER (\\%) & {test_metrics['avg_cer']:.2f} \\\\")
    lines.append(f"WER (\\%) & {test_metrics['avg_wer']:.2f} \\\\")
    lines.append(f"Exact Match (\\%) & {test_metrics['exact_match_accuracy']:.2f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Category results table (top 15)
    if category_results:
        sorted_cats = sorted(category_results.items(), key=lambda x: x[1]["corpus_bleu"], reverse=True)
        top15 = sorted_cats[:15]

        lines.append("% ──── Category Results Table ────")
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\caption{Translation Quality by Semantic Category (Top 15)}")
        lines.append("\\label{tab:category_results}")
        lines.append("\\begin{tabular}{lccccc}")
        lines.append("\\toprule")
        lines.append("\\textbf{Category} & \\textbf{N} & \\textbf{BLEU} & "
                     "\\textbf{chrF} & \\textbf{CER\\%} & \\textbf{EM\\%} \\\\")
        lines.append("\\midrule")
        for cat, m in top15:
            lines.append(f"{cat} & {m['count']} & {m['corpus_bleu']:.1f} & "
                        f"{m['avg_chrf']:.1f} & {m['avg_cer']:.1f} & "
                        f"{m['exact_match_accuracy']:.1f} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

    output_path = os.path.join(config.RESULTS_DIR, "paper_tables.tex")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [OK] Saved: paper_tables.tex")


def main():
    config = Config()
    config.ensure_dirs()

    print("=" * 70)
    print("  SOURASHTRA TRANSLATION MODEL - EVALUATION")
    print("=" * 70)

    # ── Load Model ─────────────────────────────────────────────
    print("\n[LOAD] Loading trained model...")
    model, src_vocab, tgt_vocab = load_trained_model(config)

    # ── Load Data ──────────────────────────────────────────────
    data = load_data(config)
    test_loader = data["test_loader"]

    # ── Full Test Evaluation ───────────────────────────────────
    print("\n[STATS] Running full test set evaluation...")
    test_metrics, test_refs, test_hyps = run_full_evaluation(
        model, test_loader, src_vocab, tgt_vocab, config.DEVICE
    )
    print_metrics(test_metrics, "TEST SET RESULTS")

    # ── Category-level Evaluation ──────────────────────────────
    print("\n[STATS] Running category-level evaluation...")
    category_results = evaluate_by_category(model, config, src_vocab, tgt_vocab)

    # Print top and bottom categories
    sorted_cats = sorted(category_results.items(), key=lambda x: x[1]["corpus_bleu"], reverse=True)
    print("\n  [TOP] Top 10 Categories by BLEU:")
    for cat, m in sorted_cats[:10]:
        print(f"    {cat:30s} BLEU={m['corpus_bleu']:6.2f}  EM={m['exact_match_accuracy']:5.1f}%  (n={m['count']})")
    print("\n  [LOW] Bottom 10 Categories by BLEU:")
    for cat, m in sorted_cats[-10:]:
        print(f"    {cat:30s} BLEU={m['corpus_bleu']:6.2f}  EM={m['exact_match_accuracy']:5.1f}%  (n={m['count']})")

    # ── Generate Plots ─────────────────────────────────────────
    print("\n[VIZ] Generating visualizations...")
    plot_training_curves(config)
    plot_category_performance(category_results, config)
    plot_error_distribution(test_refs, test_hyps, config)

    # ── Attention Heatmaps ─────────────────────────────────────
    print("\n[ATTN] Generating attention heatmaps...")
    sample_pairs = [
        (data["test_df"]["source"].iloc[i], data["test_df"]["target"].iloc[i])
        for i in range(min(5, len(data["test_df"])))
    ]
    for src, tgt in sample_pairs:
        fname = plot_attention_heatmap(model, src, tgt, src_vocab, tgt_vocab, config)
        if fname:
            print(f"  [OK] Saved: {fname}")

    # ── Generate Paper Tables ──────────────────────────────────
    print("\n[INFO] Generating LaTeX tables for paper...")
    generate_paper_table(test_metrics, category_results, config)

    # ── Save all results ───────────────────────────────────────
    all_results = {
        "test_metrics": test_metrics,
        "category_results": category_results,
    }
    with open(os.path.join(config.RESULTS_DIR, "full_evaluation.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved: full_evaluation.json")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\n  Results saved in: {config.RESULTS_DIR}/")
    print(f"  Files generated:")
    print(f"    - test_results.json")
    print(f"    - full_evaluation.json")
    print(f"    - training_curves.png")
    print(f"    - category_performance.png")
    print(f"    - error_distribution.png")
    print(f"    - attention_*.png")
    print(f"    - paper_tables.tex")


if __name__ == "__main__":
    main()
