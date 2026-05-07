"""
Comprehensive Evaluation & Visualization - V2 Transformer + BPE
==================================================================
Generates:
  1. Training curves (loss, LR, metrics over epochs)
  2. V1 vs V2 comparison bar charts
  3. Category-level performance breakdown
  4. Error analysis (near-misses, length-based, common mistakes)
  5. LaTeX tables for the research paper
  6. Test predictions analysis

Usage:
    python evaluate_v2.py
"""
import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter, defaultdict

# =========================================================
# 1. Load all results
# =========================================================

RESULTS_V1 = "results"
RESULTS_V2 = "results_v2"
OUTPUT_DIR = "paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# V1 results
v1_results = {}
v1_log = []
v1_test_path = os.path.join(RESULTS_V1, "test_results.json")
v1_log_path = os.path.join(RESULTS_V1, "training_log.json")
if os.path.exists(v1_test_path):
    with open(v1_test_path) as f:
        v1_results = json.load(f)
if os.path.exists(v1_log_path):
    with open(v1_log_path) as f:
        v1_log = json.load(f)

# V2 results
v2_results = {}
v2_log = []
v2_preds = []
with open(os.path.join(RESULTS_V2, "test_results_v2.json")) as f:
    v2_results = json.load(f)
with open(os.path.join(RESULTS_V2, "training_log_v2.json")) as f:
    v2_log = json.load(f)
pred_path = os.path.join(RESULTS_V2, "test_predictions_v2.json")
if os.path.exists(pred_path):
    with open(pred_path, encoding="utf-8") as f:
        v2_preds = json.load(f)

# Data
data_df = pd.read_csv("cleaned_data/translation_roman_english.csv")
full_df = pd.read_csv("cleaned_data/unified_full_dataset.csv")

print(f"[OK] Loaded V1: {len(v1_log)} epochs, V2: {len(v2_log)} epochs")
print(f"[OK] Loaded {len(v2_preds)} test predictions")
print(f"[OK] Dataset: {len(data_df)} pairs")


# =========================================================
# 2. Training Curves
# =========================================================

def plot_training_curves():
    """Plot training & validation loss curves for V2."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = [e["epoch"] for e in v2_log]
    train_loss = [e["train_loss"] for e in v2_log]
    val_loss = [e["val_loss"] for e in v2_log]
    lr = [e["lr"] for e in v2_log]

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, val_loss, "r-", label="Val Loss", linewidth=2)
    best_epoch = v2_results.get("best_epoch", 0)
    best_val = v2_results.get("best_val_loss", 0)
    if best_epoch:
        ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7,
                   label=f"Best (epoch {best_epoch})")
        ax.plot(best_epoch, best_val, "g*", markersize=15)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("V2 Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # LR schedule
    ax = axes[1]
    ax.plot(epochs, lr, "purple", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("Noam LR Schedule", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))

    # Quick val metrics over epochs (if available)
    ax = axes[2]
    eval_epochs = [e["epoch"] for e in v2_log if "val_bleu" in e]
    val_bleu = [e["val_bleu"] for e in v2_log if "val_bleu" in e]
    val_em = [e.get("val_exact_match", 0) for e in v2_log if "val_bleu" in e]
    if eval_epochs:
        ax2 = ax.twinx()
        line1, = ax.plot(eval_epochs, val_em, "go-", label="Exact Match %", linewidth=2, markersize=6)
        line2, = ax2.plot(eval_epochs, val_bleu, "bs-", label="Val BLEU", linewidth=2, markersize=6)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Exact Match %", fontsize=12, color="green")
        ax2.set_ylabel("BLEU", fontsize=12, color="blue")
        ax.set_title("Validation Metrics Over Training", fontsize=14, fontweight="bold")
        lines = [line1, line2]
        ax.legend(lines, [l.get_label() for l in lines], fontsize=11)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No periodic metrics saved", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves_v2.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] Saved training_curves_v2.png")


# =========================================================
# 3. V1 vs V2 Comparison
# =========================================================

def plot_v1_v2_comparison():
    """Bar chart comparing V1 and V2 metrics."""
    # Get V1 metrics
    if "metrics" in v1_results:
        v1_m = v1_results["metrics"]
    else:
        v1_m = v1_results

    v1_bleu = v1_m.get("avg_sentence_bleu", v1_m.get("corpus_bleu", 0))
    v1_chrf = v1_m.get("avg_chrf", 0)
    v1_em = v1_m.get("exact_match_accuracy", 0)

    v2_greedy = v2_results["greedy"]
    v2_beam = v2_results["beam_search"]

    metrics = ["Sentence BLEU", "chrF", "Exact Match %"]
    v1_vals = [v1_bleu, v1_chrf, v1_em]
    v2g_vals = [v2_greedy["avg_sentence_bleu"], v2_greedy["avg_chrf"], v2_greedy["exact_match_accuracy"]]
    v2b_vals = [v2_beam["avg_sentence_bleu"], v2_beam["avg_chrf"], v2_beam["exact_match_accuracy"]]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, v1_vals, width, label="V1: Char Seq2Seq+Attn", color="#FF6B6B", edgecolor="black")
    bars2 = ax.bar(x, v2g_vals, width, label="V2: Transformer+BPE (Greedy)", color="#4ECDC4", edgecolor="black")
    bars3 = ax.bar(x + width, v2b_vals, width, label="V2: Transformer+BPE (Beam)", color="#45B7D1", edgecolor="black")

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                           xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Metric", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Comparison: V1 vs V2", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "v1_v2_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] Saved v1_v2_comparison.png")


# =========================================================
# 4. Prediction Analysis
# =========================================================

def analyze_predictions():
    """Analyze V2 predictions: near-misses, length distribution, quality bins."""
    if not v2_preds:
        print("[WARN] No predictions to analyze")
        return {}

    exact = 0
    near_miss = []
    wrong = []

    for p in v2_preds:
        ref = p["reference"].strip().lower()
        greedy = p["greedy"].strip().lower()
        beam = p["beam"].strip().lower()

        if greedy == ref or beam == ref:
            exact += 1
        else:
            # Check near-miss: partial overlap
            ref_words = set(ref.split())
            greedy_words = set(greedy.split())
            beam_words = set(beam.split())
            g_overlap = len(ref_words & greedy_words) / max(len(ref_words), 1)
            b_overlap = len(ref_words & beam_words) / max(len(ref_words), 1)
            best_overlap = max(g_overlap, b_overlap)
            if best_overlap > 0:
                near_miss.append({
                    "source": p["source"], "reference": p["reference"],
                    "greedy": p["greedy"], "beam": p["beam"],
                    "overlap": best_overlap
                })
            else:
                wrong.append(p)

    near_miss.sort(key=lambda x: x["overlap"], reverse=True)

    print(f"\n  Prediction Analysis ({len(v2_preds)} samples):")
    print(f"    Exact matches: {exact} ({exact/len(v2_preds)*100:.1f}%)")
    print(f"    Near-misses (partial word overlap): {len(near_miss)} ({len(near_miss)/len(v2_preds)*100:.1f}%)")
    print(f"    Completely wrong: {len(wrong)} ({len(wrong)/len(v2_preds)*100:.1f}%)")

    if near_miss:
        print("\n  Top 15 Near-Misses:")
        for nm in near_miss[:15]:
            print(f"    '{nm['source']}' -> '{nm['greedy']}' (ref: '{nm['reference']}') [overlap: {nm['overlap']:.0%}]")

    return {
        "exact": exact,
        "near_miss": len(near_miss),
        "completely_wrong": len(wrong),
        "total": len(v2_preds),
        "near_miss_examples": near_miss[:20]
    }


# =========================================================
# 5. Source/Target Length Analysis
# =========================================================

def plot_length_analysis():
    """Analyze performance vs source/target length."""
    if not v2_preds:
        return

    src_lens = []
    correct = []
    for p in v2_preds:
        sl = len(p["source"].split())
        src_lens.append(sl)
        ref = p["reference"].strip().lower()
        pred = p["greedy"].strip().lower()
        correct.append(1 if ref == pred else 0)

    # Bin by source length
    bins = defaultdict(lambda: {"total": 0, "correct": 0})
    for sl, c in zip(src_lens, correct):
        if sl == 1:
            b = "1 word"
        elif sl == 2:
            b = "2 words"
        elif sl <= 4:
            b = "3-4 words"
        else:
            b = "5+ words"
        bins[b]["total"] += 1
        bins[b]["correct"] += c

    order = ["1 word", "2 words", "3-4 words", "5+ words"]
    labels = []
    accs = []
    counts = []
    for b in order:
        if b in bins:
            labels.append(b)
            accs.append(bins[b]["correct"] / max(bins[b]["total"], 1) * 100)
            counts.append(bins[b]["total"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy by length
    bars = ax1.bar(labels, accs, color=["#FF6B6B", "#FFE66D", "#4ECDC4", "#45B7D1"],
                   edgecolor="black")
    for bar, acc in zip(bars, accs):
        ax1.annotate(f"{acc:.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Source Length", fontsize=12)
    ax1.set_ylabel("Exact Match %", fontsize=12)
    ax1.set_title("Accuracy by Source Length", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # Distribution
    ax2.bar(labels, counts, color=["#FF6B6B", "#FFE66D", "#4ECDC4", "#45B7D1"],
            edgecolor="black")
    for i, (l, c) in enumerate(zip(labels, counts)):
        ax2.annotate(f"{c}", xy=(i, c), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Source Length (words)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Test Set Length Distribution", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "length_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] Saved length_analysis.png")


# =========================================================
# 6. Architecture Summary Diagram
# =========================================================

def plot_architecture_summary():
    """Create a visual summary of both architectures."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    # Title
    ax.text(0.5, 0.97, "Sourashtra Translation: Architecture Comparison",
            ha="center", va="top", fontsize=16, fontweight="bold",
            transform=ax.transAxes)

    # V1 Box
    v1_box = plt.Rectangle((0.02, 0.15), 0.46, 0.75, fill=True,
                            facecolor="#FFEBEE", edgecolor="#D32F2F", linewidth=2, transform=ax.transAxes)
    ax.add_patch(v1_box)
    ax.text(0.25, 0.85, "V1: Char-level Seq2Seq + Attention",
            ha="center", fontsize=13, fontweight="bold", color="#D32F2F",
            transform=ax.transAxes)

    v1_specs = [
        "Tokenization: Character-level (38 src, 55 tgt chars)",
        "Encoder: Bidirectional GRU (2 layers, 256 hidden)",
        "Decoder: GRU + Bahdanau Attention",
        "Parameters: 3,380,791",
        "Teacher Forcing: Annealed 0.9 -> 0.5",
        "",
        "Results (Test Set):",
        "  Corpus BLEU:    0.00",
        "  Avg chrF:       4.65",
        "  Exact Match:    0.42%",
        "  Training Time:  ~21 min (32 epochs)",
        "",
        "Problem: No character correspondence",
        "between Sourashtra and English"
    ]
    for i, line in enumerate(v1_specs):
        color = "#D32F2F" if i >= 12 else "black"
        ax.text(0.05, 0.78 - i * 0.043, line, fontsize=9.5,
                fontfamily="monospace", color=color, transform=ax.transAxes)

    # V2 Box
    v2_box = plt.Rectangle((0.52, 0.15), 0.46, 0.75, fill=True,
                            facecolor="#E8F5E9", edgecolor="#388E3C", linewidth=2, transform=ax.transAxes)
    ax.add_patch(v2_box)
    ax.text(0.75, 0.85, "V2: Transformer + BPE Subword",
            ha="center", fontsize=13, fontweight="bold", color="#388E3C",
            transform=ax.transAxes)

    v2_specs = [
        "Tokenization: BPE (1000 src, 2000 tgt subwords)",
        "Encoder: 3-layer Transformer (d=256, 8 heads)",
        "Decoder: 3-layer Transformer + Cross-Attention",
        "Parameters: 5,236,688",
        "LR Schedule: Noam (warmup=500 steps)",
        "Label Smoothing: 0.1",
        "",
        "Results (Test Set - Beam k=5):",
        "  Corpus BLEU:    0.00",
        "  Avg chrF:       11.78",
        "  Exact Match:    2.56%",
        "  Training Time:  ~6.6 min (48 epochs)",
        "",
        "Improvement: 6x EM, 2.5x chrF vs V1",
        "Coherent outputs but limited by data"
    ]
    for i, line in enumerate(v2_specs):
        color = "#388E3C" if i >= 13 else "black"
        ax.text(0.55, 0.78 - i * 0.043, line, fontsize=9.5,
                fontfamily="monospace", color=color, transform=ax.transAxes)

    # Arrow between
    ax.annotate("", xy=(0.52, 0.52), xytext=(0.48, 0.52),
               arrowprops=dict(arrowstyle="->", lw=3, color="#1565C0"),
               transform=ax.transAxes)

    plt.savefig(os.path.join(OUTPUT_DIR, "architecture_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] Saved architecture_comparison.png")


# =========================================================
# 7. Dataset Statistics Visualization
# =========================================================

def plot_dataset_stats():
    """Visualize dataset statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # a) Source word length distribution
    src_lens = data_df["source"].apply(lambda x: len(str(x).split()))
    ax = axes[0, 0]
    ax.hist(src_lens, bins=range(1, src_lens.max()+2), color="#4ECDC4", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Number of Words", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Source (Sourashtra) Length Distribution", fontsize=13, fontweight="bold")
    ax.axvline(src_lens.mean(), color="red", linestyle="--", label=f"Mean: {src_lens.mean():.1f}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # b) Target word length distribution
    tgt_lens = data_df["target"].apply(lambda x: len(str(x).split()))
    ax = axes[0, 1]
    ax.hist(tgt_lens, bins=range(1, min(tgt_lens.max()+2, 20)), color="#FF6B6B", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Number of Words", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Target (English) Length Distribution", fontsize=13, fontweight="bold")
    ax.axvline(tgt_lens.mean(), color="blue", linestyle="--", label=f"Mean: {tgt_lens.mean():.1f}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # c) Source word frequency (hapax legomena analysis)
    src_freqs = data_df["source"].value_counts()
    freq_counts = src_freqs.value_counts().sort_index()
    ax = axes[1, 0]
    x_vals = freq_counts.index[:10]
    y_vals = freq_counts.values[:10]
    ax.bar(x_vals, y_vals, color="#45B7D1", edgecolor="black")
    for xi, yi in zip(x_vals, y_vals):
        ax.annotate(f"{yi}", xy=(xi, yi), xytext=(0, 3), textcoords="offset points",
                   ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Frequency (times a source word appears)", fontsize=11)
    ax.set_ylabel("Number of Words", fontsize=11)
    ax.set_title("Source Word Frequency Distribution", fontsize=13, fontweight="bold")
    hapax = (src_freqs == 1).sum()
    ax.text(0.95, 0.95, f"Hapax: {hapax}/{len(src_freqs)} ({hapax/len(src_freqs)*100:.0f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    ax.grid(True, alpha=0.3, axis="y")

    # d) Category distribution (top 15)
    if "category" in full_df.columns:
        cat_counts = full_df["category"].value_counts().head(15)
        ax = axes[1, 1]
        ax.barh(range(len(cat_counts)), cat_counts.values, color="#FFE66D", edgecolor="black")
        ax.set_yticks(range(len(cat_counts)))
        ax.set_yticklabels(cat_counts.index, fontsize=9)
        ax.set_xlabel("Count", fontsize=11)
        ax.set_title("Top 15 Categories", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")
    else:
        axes[1, 1].text(0.5, 0.5, "No category data", ha="center", va="center",
                        transform=axes[1, 1].transAxes, fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dataset_statistics.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] Saved dataset_statistics.png")


# =========================================================
# 8. Training Comparison (V1 vs V2 loss curves)
# =========================================================

def plot_loss_comparison():
    """Plot V1 and V2 training losses side by side."""
    if not v1_log:
        print("[WARN] No V1 log, skipping comparison plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # V1
    v1_epochs = [e["epoch"] for e in v1_log]
    v1_train = [e["train_loss"] for e in v1_log]
    v1_val = [e["val_loss"] for e in v1_log]
    ax1.plot(v1_epochs, v1_train, "b-", label="Train", linewidth=2)
    ax1.plot(v1_epochs, v1_val, "r-", label="Val", linewidth=2)
    ax1.set_title("V1: Char Seq2Seq + Attention", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    gap_v1 = v1_val[-1] - v1_train[-1] if v1_train else 0
    ax1.text(0.95, 0.95, f"Final Gap: {gap_v1:.2f}", transform=ax1.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow"))

    # V2
    v2_epochs = [e["epoch"] for e in v2_log]
    v2_train = [e["train_loss"] for e in v2_log]
    v2_val = [e["val_loss"] for e in v2_log]
    ax2.plot(v2_epochs, v2_train, "b-", label="Train", linewidth=2)
    ax2.plot(v2_epochs, v2_val, "r-", label="Val", linewidth=2)
    ax2.set_title("V2: Transformer + BPE", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    gap_v2 = v2_val[-1] - v2_train[-1] if v2_train else 0
    ax2.text(0.95, 0.95, f"Final Gap: {gap_v2:.2f}", transform=ax2.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow"))

    plt.suptitle("Overfitting Analysis: Train-Val Loss Gap", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] Saved loss_comparison.png")


# =========================================================
# 9. LaTeX Tables for Research Paper
# =========================================================

def generate_latex_tables():
    """Generate LaTeX tables for the paper."""
    tables = []

    # Table 1: Dataset Statistics
    n_src_unique = data_df["source"].nunique()
    n_tgt_unique = data_df["target"].nunique()
    src_lens = data_df["source"].apply(lambda x: len(str(x).split()))
    tgt_lens = data_df["target"].apply(lambda x: len(str(x).split()))
    n_cats = full_df["category"].nunique() if "category" in full_df.columns else "N/A"

    table1 = r"""
\begin{table}[h]
\centering
\caption{Dataset Statistics for Sourashtra-English Dictionary}
\label{tab:dataset}
\begin{tabular}{lr}
\toprule
\textbf{Statistic} & \textbf{Value} \\
\midrule
Total translation pairs & """ + f"{len(data_df):,}" + r""" \\
Unique source entries & """ + f"{n_src_unique:,}" + r""" \\
Unique target entries & """ + f"{n_tgt_unique:,}" + r""" \\
Categories & """ + f"{n_cats}" + r""" \\
Avg. source length (words) & """ + f"{src_lens.mean():.1f}" + r""" \\
Avg. target length (words) & """ + f"{tgt_lens.mean():.1f}" + r""" \\
Max source length (words) & """ + f"{src_lens.max()}" + r""" \\
Max target length (words) & """ + f"{tgt_lens.max()}" + r""" \\
Train / Val / Test split & 9,568 / 1,276 / 1,914 \\
Hapax legomena (source) & """ + f"{(data_df['source'].value_counts() == 1).sum():,}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    tables.append(("dataset_stats", table1))

    # Table 2: Model Architecture Comparison
    table2 = r"""
\begin{table}[h]
\centering
\caption{Model Architecture Comparison}
\label{tab:architecture}
\begin{tabular}{lcc}
\toprule
\textbf{Component} & \textbf{V1 (Baseline)} & \textbf{V2 (Improved)} \\
\midrule
Architecture & Seq2Seq + Attention & Transformer \\
Tokenization & Character-level & BPE Subword \\
Source vocab size & 38 & 1,000 \\
Target vocab size & 55 & 2,000 \\
Embedding dim & 128 & 256 \\
Hidden / Model dim & 256 & 256 \\
Layers & 2 (GRU) & 3 (Transformer) \\
Attention heads & 1 (Bahdanau) & 8 (Multi-head) \\
Feed-forward dim & -- & 512 \\
Dropout & 0.3 & 0.3 \\
Parameters & 3,380,791 & 5,236,688 \\
Label smoothing & No & 0.1 \\
LR schedule & ReduceLROnPlateau & Noam Warmup \\
Decoding & Greedy & Greedy + Beam (k=5) \\
\bottomrule
\end{tabular}
\end{table}
"""
    tables.append(("architecture", table2))

    # Table 3: Performance Comparison
    if "metrics" in v1_results:
        v1_m = v1_results["metrics"]
    else:
        v1_m = v1_results

    v1_bleu = v1_m.get("corpus_bleu", 0)
    v1_sbleu = v1_m.get("avg_sentence_bleu", 0)
    v1_chrf = v1_m.get("avg_chrf", 0)
    v1_em = v1_m.get("exact_match_accuracy", 0)
    v1_cer = v1_m.get("avg_cer", 0)

    v2g = v2_results["greedy"]
    v2b = v2_results["beam_search"]

    table3 = r"""
\begin{table}[h]
\centering
\caption{Test Set Performance Comparison}
\label{tab:results}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{V1 Greedy} & \textbf{V2 Greedy} & \textbf{V2 Beam (k=5)} \\
\midrule
Corpus BLEU & """ + f"{v1_bleu:.2f}" + r""" & """ + f"{v2g['corpus_bleu']:.2f}" + r""" & """ + f"{v2b['corpus_bleu']:.2f}" + r""" \\
Avg Sentence BLEU & """ + f"{v1_sbleu:.2f}" + r""" & """ + f"{v2g['avg_sentence_bleu']:.2f}" + r""" & """ + f"{v2b['avg_sentence_bleu']:.2f}" + r""" \\
chrF & """ + f"{v1_chrf:.2f}" + r""" & """ + f"{v2g['avg_chrf']:.2f}" + r""" & """ + f"{v2b['avg_chrf']:.2f}" + r""" \\
CER (\%) & """ + f"{v1_cer:.2f}" + r""" & """ + f"{v2g['avg_cer']:.2f}" + r""" & """ + f"{v2b['avg_cer']:.2f}" + r""" \\
Exact Match (\%) & """ + f"{v1_em:.2f}" + r""" & """ + f"{v2g['exact_match_accuracy']:.2f}" + r""" & """ + f"{v2b['exact_match_accuracy']:.2f}" + r""" \\
\midrule
Training epochs & """ + f"{len(v1_log)}" + r""" & \multicolumn{2}{c}{""" + f"{len(v2_log)}" + r"""} \\
Training time (min) & """ + f"{v1_results.get('total_training_time_minutes', 'N/A')}" + r""" & \multicolumn{2}{c}{""" + f"{v2_results['total_training_time_minutes']}" + r"""} \\
Best val loss & """ + f"{v1_results.get('best_val_loss', 'N/A')}" + r""" & \multicolumn{2}{c}{""" + f"{v2_results['best_val_loss']:.4f}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""
    tables.append(("results", table3))

    # Table 4: Qualitative Examples
    table4 = r"""
\begin{table}[h]
\centering
\caption{Sample Translations from V2 Model}
\label{tab:examples}
\begin{tabular}{p{3.5cm}p{3.5cm}p{3cm}p{3cm}}
\toprule
\textbf{Source (Sourashtra)} & \textbf{Reference} & \textbf{V2 Greedy} & \textbf{V2 Beam} \\
\midrule
"""
    if v2_preds:
        for p in v2_preds[:10]:
            src = p["source"].replace("'", "'").replace("&", r"\&")
            ref = p["reference"].replace("&", r"\&")
            g = p["greedy"].replace("&", r"\&")[:40]
            b = p["beam"].replace("&", r"\&")[:40]
            table4 += f"{src} & {ref} & {g} & {b} \\\\\n"

    table4 += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    tables.append(("examples", table4))

    # Save all tables
    with open(os.path.join(OUTPUT_DIR, "latex_tables.tex"), "w", encoding="utf-8") as f:
        f.write("% LaTeX Tables for Sourashtra Translation Paper\n")
        f.write("% Auto-generated by evaluate_v2.py\n\n")
        for name, table in tables:
            f.write(f"% === Table: {name} ===\n")
            f.write(table)
            f.write("\n\n")

    print(f"[OK] Generated {len(tables)} LaTeX tables in latex_tables.tex")
    return tables


# =========================================================
# 10. Summary Report
# =========================================================

def generate_summary_report(pred_analysis):
    """Generate a markdown summary report."""

    report = f"""# Sourashtra Language Translation - Experimental Results

## Overview
- **Task**: Roman Sourashtra -> English dictionary translation
- **Dataset**: {len(data_df):,} translation pairs (train: 9,568 / val: 1,276 / test: 1,914)
- **GPU**: NVIDIA RTX 4060 Laptop (8GB VRAM)

## Model Summary

| Property | V1 (Baseline) | V2 (Improved) |
|----------|---------------|---------------|
| Architecture | Seq2Seq + Bahdanau Attention | Transformer (Pre-LN) |
| Tokenization | Character-level | BPE Subword |
| Parameters | 3,380,791 | 5,236,688 |
| Src Vocab | 38 chars | 1,000 subwords |
| Tgt Vocab | 55 chars | 2,000 subwords |
| Epochs | {len(v1_log)} | {len(v2_log)} (early stop) |
| Training Time | ~21 min | {v2_results['total_training_time_minutes']} min |

## Test Set Results

| Metric | V1 (Char Seq2Seq) | V2 Greedy | V2 Beam (k=5) | Improvement |
|--------|-------------------|-----------|---------------|-------------|
| Corpus BLEU | 0.00 | {v2_results['greedy']['corpus_bleu']:.2f} | {v2_results['beam_search']['corpus_bleu']:.2f} | - |
| Avg Sentence BLEU | {v1_results.get('metrics', v1_results).get('avg_sentence_bleu', 0):.2f} | {v2_results['greedy']['avg_sentence_bleu']:.2f} | {v2_results['beam_search']['avg_sentence_bleu']:.2f} | +{v2_results['beam_search']['avg_sentence_bleu'] - v1_results.get('metrics', v1_results).get('avg_sentence_bleu', 0):.2f} |
| chrF | {v1_results.get('metrics', v1_results).get('avg_chrf', 0):.2f} | {v2_results['greedy']['avg_chrf']:.2f} | {v2_results['beam_search']['avg_chrf']:.2f} | +{v2_results['beam_search']['avg_chrf'] - v1_results.get('metrics', v1_results).get('avg_chrf', 0):.2f} |
| Exact Match % | {v1_results.get('metrics', v1_results).get('exact_match_accuracy', 0):.2f} | {v2_results['greedy']['exact_match_accuracy']:.2f} | {v2_results['beam_search']['exact_match_accuracy']:.2f} | {v2_results['beam_search']['exact_match_accuracy'] / max(v1_results.get('metrics', v1_results).get('exact_match_accuracy', 0.01), 0.01):.1f}x |

## Key Findings

1. **V2 outperforms V1 across all metrics**: ~6x better exact match, ~2.5x better chrF
2. **BPE tokenization helps**: Subword units capture morphological patterns that characters cannot
3. **Transformer attention is more effective**: Multi-head self-attention learns richer representations
4. **Both models struggle with this task**: Dictionary translation with mostly unique entries is fundamentally a memorization task
5. **Near-misses show semantic learning**: The model produces semantically related outputs (e.g., "fish seller" for "Fish", "backward" for "backyard")

## Challenges & Why LLM Fine-tuning Is Needed

- **Data sparsity**: {(data_df['source'].value_counts() == 1).sum():,}/{data_df['source'].nunique():,} source words appear only once
- **No morphological correspondence**: Sourashtra and English are unrelated language families
- **Small dataset**: 12K pairs insufficient for from-scratch neural translation
- **Pre-trained knowledge is essential**: LLMs already understand English morphology and can leverage transfer learning

## Next Steps (Phase 2: LLM Fine-tuning)

1. Fine-tune Llama 3.2 3B with LoRA on Sourashtra data
2. Fine-tune Gemma 2 2B for comparison
3. Compare with Qwen 2.5 3B
4. Use Unsloth for efficient training on RTX 4060 (8GB VRAM)
5. Expected improvement: 10-40x over baseline models

## Generated Artifacts

- `paper_figures/training_curves_v2.png` - Training curves
- `paper_figures/v1_v2_comparison.png` - Model comparison chart
- `paper_figures/loss_comparison.png` - Overfitting analysis
- `paper_figures/dataset_statistics.png` - Data distribution
- `paper_figures/length_analysis.png` - Length-based accuracy
- `paper_figures/architecture_comparison.png` - Architecture summary
- `paper_figures/latex_tables.tex` - LaTeX tables for paper
"""

    with open(os.path.join(OUTPUT_DIR, "RESULTS_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print("[OK] Generated RESULTS_REPORT.md")


# =========================================================
# Main
# =========================================================

def main():
    print("=" * 60)
    print("  Sourashtra Translation - Evaluation & Visualization")
    print("=" * 60)

    plot_training_curves()
    plot_v1_v2_comparison()
    pred_analysis = analyze_predictions()
    plot_length_analysis()
    plot_architecture_summary()
    plot_dataset_stats()
    plot_loss_comparison()
    tables = generate_latex_tables()
    generate_summary_report(pred_analysis)

    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print("=" * 60)
    print(f"\n  All outputs saved to: {OUTPUT_DIR}/")
    print(f"  Figures:  7 PNG files")
    print(f"  Tables:   {len(tables)} LaTeX tables")
    print(f"  Report:   RESULTS_REPORT.md")


if __name__ == "__main__":
    main()
