"""
Evaluation & Visualization Script V3
======================================
Comprehensive comparison: V1 vs V2 vs V3 (Neural + Hybrid)
"""
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config_v3 import ConfigV3

config = ConfigV3()
RESULTS_DIR = config.RESULTS_DIR
FIG_DIR = os.path.join(config.PROJECT_ROOT, "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load results ──
v1_results = json.load(open(os.path.join(config.PROJECT_ROOT, "results", "test_results.json")))
v2_bs = json.load(open(os.path.join(config.PROJECT_ROOT, "results_v2", "test_results_v2.json")))["beam_search"]
v3_results = json.load(open(os.path.join(RESULTS_DIR, "test_results_v3.json")))
hybrid_results = json.load(open(os.path.join(RESULTS_DIR, "hybrid_results_v3.json")))

# ── Extract metrics ──
models = ["V1\n(GRU Seq2Seq)", "V2\n(Transformer+BPE)", "V3 Neural\n(T5-small)", "V3 Hybrid\n(T5+Retrieval)"]
exact_match = [
    v1_results["exact_match_accuracy"],
    v2_bs["exact_match_accuracy"],
    v3_results["test_metrics"]["exact_match_accuracy"],
    hybrid_results["test_metrics"]["exact_match_accuracy"],
]
bleu = [
    v1_results["corpus_bleu"],
    v2_bs["corpus_bleu"],
    v3_results["test_metrics"]["corpus_bleu"],
    hybrid_results["test_metrics"]["corpus_bleu"],
]
chrf = [
    v1_results["avg_chrf"],
    v2_bs["avg_chrf"],
    v3_results["test_metrics"]["avg_chrf"],
    hybrid_results["test_metrics"]["avg_chrf"],
]

print("=" * 60)
print("  V1 vs V2 vs V3 COMPARISON")
print("=" * 60)
for i, m in enumerate(models):
    name = m.replace('\n', ' ')
    print(f"  {name:30s}  EM={exact_match[i]:6.2f}%  BLEU={bleu[i]:5.2f}  chrF={chrf[i]:5.2f}")
print("=" * 60)

# ── Plot 1: Exact Match Comparison ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

ax = axes[0]
bars = ax.bar(range(len(models)), exact_match, color=colors, edgecolor='white', linewidth=1.5)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel("Exact Match (%)", fontsize=12)
ax.set_title("Exact Match Accuracy", fontsize=13, fontweight='bold')
for bar, val in zip(bars, exact_match):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_ylim(0, max(exact_match) * 1.3)
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
bars = ax.bar(range(len(models)), bleu, color=colors, edgecolor='white', linewidth=1.5)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel("Corpus BLEU", fontsize=12)
ax.set_title("BLEU Score", fontsize=13, fontweight='bold')
for bar, val in zip(bars, bleu):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylim(0, max(bleu) * 1.3)
ax.grid(axis='y', alpha=0.3)

ax = axes[2]
bars = ax.bar(range(len(models)), chrf, color=colors, edgecolor='white', linewidth=1.5)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel("chrF Score", fontsize=12)
ax.set_title("chrF Score", fontsize=13, fontweight='bold')
for bar, val in zip(bars, chrf):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylim(0, max(chrf) * 1.3)
ax.grid(axis='y', alpha=0.3)

plt.suptitle("Sourashtra → English Translation: Model Comparison", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "v3_comparison.png"), dpi=150, bbox_inches='tight')
print(f"\n  Saved: {FIG_DIR}/v3_comparison.png")

# ── Plot 2: Improvement progression ──
fig, ax = plt.subplots(figsize=(10, 6))
versions = ['V1', 'V2', 'V3 Neural', 'V3 Hybrid']
em_values = exact_match
ax.plot(versions, em_values, 'o-', color='#e74c3c', linewidth=2.5, markersize=12, zorder=5)
ax.fill_between(range(len(versions)), em_values, alpha=0.15, color='#e74c3c')
for i, (v, em) in enumerate(zip(versions, em_values)):
    ax.annotate(f'{em:.2f}%', (i, em), textcoords="offset points",
                xytext=(0, 15), ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel("Exact Match (%)", fontsize=13)
ax.set_title("Translation Accuracy Progression", fontsize=15, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(em_values) * 1.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "v3_progression.png"), dpi=150, bbox_inches='tight')
print(f"  Saved: {FIG_DIR}/v3_progression.png")

# ── Plot 3: Hybrid method breakdown (pie) ──
if "method_distribution" in hybrid_results:
    pass  # Skip if not available

print("\n  All figures saved!")
plt.close('all')
