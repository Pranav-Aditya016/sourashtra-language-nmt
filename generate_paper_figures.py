"""
Research Paper Figures — Sourashtra-English Machine Translation
================================================================
Publication-quality figures for IEEE paper covering V1 → V5 results.
Includes multilingual Tamil cross-lingual transfer analysis (V4/V5)
and ByT5 byte-level tokenization comparison (V5).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

OUT = "paper_figures"
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ══════════════════════════════════════════════════════════════════════
v1 = json.load(open("results/test_results.json"))
v2_all = json.load(open("results_v2/test_results_v2.json"))
v2 = v2_all['beam_search']
v3 = json.load(open("results_v3/test_results_v3.json"))
hybrid_v3 = json.load(open("results_v3/hybrid_results_v3.json"))
v4 = json.load(open("results_v4/test_results_v4.json"))
hybrid_v4 = json.load(open("results_v4/hybrid_results_v4.json"))
v5 = json.load(open("results_v5/test_results_v5.json"))
hybrid_v5 = json.load(open("results_v5/hybrid_results_v5.json"))
hybrid_v4_enhanced = json.load(open("results_v5/hybrid_results_v4_enhanced.json"))

v1_log = json.load(open("results/training_log.json"))
v2_log = json.load(open("results_v2/training_log_v2.json"))

# V3 trainer state (latest checkpoint)
v3_state = None
for ckpt in ['checkpoint-7080', 'checkpoint-7040', 'checkpoint-5104']:
    path = f"checkpoints_v3/{ckpt}/trainer_state.json"
    if os.path.exists(path):
        v3_state = json.load(open(path))
        break

# V4 trainer state (latest checkpoint)
v4_state = None
for ckpt in ['checkpoint-13040', 'checkpoint-13000', 'checkpoint-10725']:
    path = f"checkpoints_v4/{ckpt}/trainer_state.json"
    if os.path.exists(path):
        v4_state = json.load(open(path))
        break

# V5 trainer state (latest checkpoint)
v5_state = None
for ckpt in ['checkpoint-7060', 'checkpoint-7040', 'checkpoint-5280']:
    path = f"checkpoints_v5/{ckpt}/trainer_state.json"
    if os.path.exists(path):
        v5_state = json.load(open(path))
        break

# Predictions
hybrid_v3_preds = json.load(open("results_v3/hybrid_predictions_v3.json"))
hybrid_v4_preds = json.load(open("results_v4/hybrid_predictions_v4.json"))
hybrid_v5_preds = json.load(open("results_v5/hybrid_predictions_v5.json"))

# Split info
split_v1 = json.load(open("results/split_info.json"))
split_v3 = json.load(open("results_v3/split_info_v3.json"))
split_v4 = json.load(open("results_v4/split_info_v4.json"))
split_v5 = json.load(open("results_v5/split_info_v5.json"))

print("All data loaded successfully (V1–V5).\n")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Model Comparison — Exact Match Accuracy (Main Result)
# ══════════════════════════════════════════════════════════════════════
print("Figure 1: Model Comparison Bar Chart...")

model_labels = [
    'V1\nChar-GRU',
    'V2\nTransformer\n+BPE',
    'V3\nT5-small\n(EN)',
    'V3 Hybrid\nT5+Retrieval',
    'V4\nT5+Tamil\n(Multilingual)',
    'V4 Hybrid\nT5+Tamil\n+Retrieval',
    'V5\nByT5\n(Byte-level)',
    'V5 Hybrid\nByT5+Enhanced\nRetrieval',
]
em_scores = [
    v1['exact_match_accuracy'],
    v2['exact_match_accuracy'],
    v3['test_metrics']['exact_match_accuracy'],
    hybrid_v3['test_metrics']['exact_match_accuracy'],
    v4['test_metrics']['exact_match_accuracy'],
    hybrid_v4['test_metrics']['exact_match_accuracy'],
    v5['test_metrics']['exact_match_accuracy'],
    hybrid_v5['test_metrics']['exact_match_accuracy'],
]
colors_main = ['#4472C4', '#5B9BD5', '#ED7D31', '#FFC000',
               '#9B59B6', '#70AD47', '#E74C3C', '#2ECC71']

fig, ax = plt.subplots(figsize=(16, 6))
bars = ax.bar(model_labels, em_scores, color=colors_main, edgecolor='black',
              linewidth=0.8, width=0.65)

for bar, score in zip(bars, em_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
            f'{score:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_ylabel('Exact Match Accuracy (%)')
ax.set_title('Model Performance Comparison \u2014 Sourashtra\u2192English Translation')
ax.set_ylim(0, max(em_scores) * 1.30)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Highlight best
best_idx = int(np.argmax(em_scores))
bars[best_idx].set_edgecolor('#C00000')
bars[best_idx].set_linewidth(2.5)

plt.tight_layout()
plt.savefig(f"{OUT}/fig1_model_comparison_em.png")
plt.close()
print("  Saved fig1_model_comparison_em.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Multi-Metric Comparison (EM, BLEU, chrF)
# ══════════════════════════════════════════════════════════════════════
print("Figure 2: Multi-Metric Comparison...")

short_labels = ['V1\nChar-GRU', 'V2\nTransformer', 'V3\nT5 (EN)',
                'V3 Hybrid', 'V4\nT5+Tamil', 'V4 Hybrid',
                'V5\nByT5', 'V5 Hybrid']

em_vals = [v1['exact_match_accuracy'], v2['exact_match_accuracy'],
           v3['test_metrics']['exact_match_accuracy'],
           hybrid_v3['test_metrics']['exact_match_accuracy'],
           v4['test_metrics']['exact_match_accuracy'],
           hybrid_v4['test_metrics']['exact_match_accuracy'],
           v5['test_metrics']['exact_match_accuracy'],
           hybrid_v5['test_metrics']['exact_match_accuracy']]

bleu_vals = [v1['corpus_bleu'], v2['corpus_bleu'],
             v3['test_metrics']['corpus_bleu'],
             hybrid_v3['test_metrics']['corpus_bleu'],
             v4['test_metrics']['corpus_bleu'],
             hybrid_v4['test_metrics']['corpus_bleu'],
             v5['test_metrics']['corpus_bleu'],
             hybrid_v5['test_metrics']['corpus_bleu']]

chrf_vals = [v1['avg_chrf'], v2['avg_chrf'],
             v3['test_metrics']['avg_chrf'],
             hybrid_v3['test_metrics']['avg_chrf'],
             v4['test_metrics']['avg_chrf'],
             hybrid_v4['test_metrics']['avg_chrf'],
             v5['test_metrics']['avg_chrf'],
             hybrid_v5['test_metrics']['avg_chrf']]

x = np.arange(len(short_labels))
width = 0.22

fig, ax = plt.subplots(figsize=(16, 6))
b1 = ax.bar(x - width, em_vals, width, label='Exact Match (%)',
            color='#4472C4', edgecolor='black', linewidth=0.5)
b2 = ax.bar(x, bleu_vals, width, label='Corpus BLEU',
            color='#ED7D31', edgecolor='black', linewidth=0.5)
b3 = ax.bar(x + width, chrf_vals, width, label='chrF',
            color='#70AD47', edgecolor='black', linewidth=0.5)

for bb in [b1, b2, b3]:
    for bar in bb:
        h = bar.get_height()
        if h >= 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=7)

ax.set_ylabel('Score')
ax.set_title('Multi-Metric Performance Comparison (V1\u2013V5)')
ax.set_xticks(x)
ax.set_xticklabels(short_labels)
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig2_multi_metric_comparison.png")
plt.close()
print("  Saved fig2_multi_metric_comparison.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Training Loss Curves (V1 / V2 / V3 / V4 / V5)
# ══════════════════════════════════════════════════════════════════════
print("Figure 3: Training Loss Curves...")

fig, axes = plt.subplots(3, 2, figsize=(14, 14))

# --- V1 ---
v1_epochs = [e['epoch'] for e in v1_log]
v1_train  = [e['train_loss'] for e in v1_log]
v1_val    = [e['val_loss'] for e in v1_log]
axes[0, 0].plot(v1_epochs, v1_train, 'b-', linewidth=1.5, label='Train Loss')
axes[0, 0].plot(v1_epochs, v1_val, 'r-', linewidth=1.5, label='Val Loss')
axes[0, 0].set_title('V1: Char-level GRU Seq2Seq')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# --- V2 ---
v2_epochs = [e['epoch'] for e in v2_log]
v2_train  = [e['train_loss'] for e in v2_log]
v2_val    = [e['val_loss'] for e in v2_log]
axes[0, 1].plot(v2_epochs, v2_train, 'b-', linewidth=1.5, label='Train Loss')
axes[0, 1].plot(v2_epochs, v2_val, 'r-', linewidth=1.5, label='Val Loss')
axes[0, 1].set_title('V2: Transformer + BPE')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# --- V3 ---
if v3_state:
    v3_train_logs = [x for x in v3_state['log_history']
                     if 'loss' in x and 'eval_loss' not in x]
    v3_eval_logs  = [x for x in v3_state['log_history'] if 'eval_loss' in x]
    v3t_ep  = [x['epoch'] for x in v3_train_logs]
    v3t_l   = [x['loss'] for x in v3_train_logs]
    v3e_ep  = [x['epoch'] for x in v3_eval_logs]
    v3e_l   = [x['eval_loss'] for x in v3_eval_logs]

    axes[1, 0].plot(v3t_ep, v3t_l, 'b-', linewidth=1, alpha=0.7, label='Train Loss')
    axes[1, 0].plot(v3e_ep, v3e_l, 'r-', linewidth=1.5, label='Val Loss')
axes[1, 0].set_title('V3: T5-small Fine-tuning (EN only)')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# --- V4 ---
if v4_state:
    v4_train_logs = [x for x in v4_state['log_history']
                     if 'loss' in x and 'eval_loss' not in x]
    v4_eval_logs  = [x for x in v4_state['log_history'] if 'eval_loss' in x]
    v4t_ep  = [x['epoch'] for x in v4_train_logs]
    v4t_l   = [x['loss'] for x in v4_train_logs]
    v4e_ep  = [x['epoch'] for x in v4_eval_logs]
    v4e_l   = [x['eval_loss'] for x in v4_eval_logs]

    axes[1, 1].plot(v4t_ep, v4t_l, 'b-', linewidth=1, alpha=0.7, label='Train Loss')
    axes[1, 1].plot(v4e_ep, v4e_l, 'r-', linewidth=1.5, label='Val Loss')
axes[1, 1].set_title('V4: T5-small + Tamil (Multilingual)')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# --- V5 ---
if v5_state:
    v5_train_logs = [x for x in v5_state['log_history']
                     if 'loss' in x and 'eval_loss' not in x]
    v5_eval_logs  = [x for x in v5_state['log_history'] if 'eval_loss' in x]
    v5t_ep  = [x['epoch'] for x in v5_train_logs]
    v5t_l   = [x['loss'] for x in v5_train_logs]
    v5e_ep  = [x['epoch'] for x in v5_eval_logs]
    v5e_l   = [x['eval_loss'] for x in v5_eval_logs]

    axes[2, 0].plot(v5t_ep, v5t_l, 'b-', linewidth=1, alpha=0.7, label='Train Loss')
    axes[2, 0].plot(v5e_ep, v5e_l, 'r-', linewidth=1.5, label='Val Loss')
axes[2, 0].set_title('V5: ByT5-small (Byte-level)')
axes[2, 0].set_xlabel('Epoch')
axes[2, 0].set_ylabel('Loss')
axes[2, 0].legend()
axes[2, 0].grid(alpha=0.3)

# Hide unused subplot
axes[2, 1].axis('off')

for ax in axes.flat:
    if ax.get_visible() and ax.has_data():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

plt.suptitle('Training Loss Curves Across All Model Versions', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_training_loss_curves.png")
plt.close()
print("  Saved fig3_training_loss_curves.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: V3 vs V4 vs V5 — Validation EM Progression During Training
# ══════════════════════════════════════════════════════════════════════
print("Figure 4: V3 vs V4 vs V5 EM Progression...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# V3 EM
if v3_state:
    v3_em = [x['eval_exact_match'] for x in v3_eval_logs]
    v3_pm = [x['eval_partial_match'] for x in v3_eval_logs]
    axes[0].plot(v3e_ep, v3_em, 'g-o', markersize=3, linewidth=1.5, label='Exact Match (%)')
    axes[0].plot(v3e_ep, v3_pm, 'b-s', markersize=3, linewidth=1.5, label='Partial Match (%)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('V3 (T5-small, English only)')
    axes[0].legend(loc='center right')
    axes[0].grid(alpha=0.3, linestyle='--')
    best3 = int(np.argmax(v3_em))
    axes[0].axvline(x=v3e_ep[best3], color='red', linestyle='--', alpha=0.5)
    axes[0].annotate(f'Best: {v3_em[best3]:.1f}%\n(ep {v3e_ep[best3]:.0f})',
                     xy=(v3e_ep[best3], v3_em[best3]),
                     xytext=(v3e_ep[best3]+3, v3_em[best3]+1),
                     fontsize=10, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='red'))

# V4 EM
if v4_state:
    v4_em = [x['eval_exact_match'] for x in v4_eval_logs]
    v4_pm = [x['eval_partial_match'] for x in v4_eval_logs]
    axes[1].plot(v4e_ep, v4_em, 'g-o', markersize=3, linewidth=1.5, label='Exact Match (%)')
    axes[1].plot(v4e_ep, v4_pm, 'b-s', markersize=3, linewidth=1.5, label='Partial Match (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('V4 (T5 + Tamil, Multilingual)')
    axes[1].legend(loc='center right')
    axes[1].grid(alpha=0.3, linestyle='--')
    best4 = int(np.argmax(v4_em))
    axes[1].axvline(x=v4e_ep[best4], color='red', linestyle='--', alpha=0.5)
    axes[1].annotate(f'Best: {v4_em[best4]:.1f}%\n(ep {v4e_ep[best4]:.0f})',
                     xy=(v4e_ep[best4], v4_em[best4]),
                     xytext=(v4e_ep[best4]+3, v4_em[best4]+1),
                     fontsize=10, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='red'))

# V5 EM
if v5_state:
    v5_em = [x['eval_exact_match'] for x in v5_eval_logs]
    v5_pm = [x['eval_partial_match'] for x in v5_eval_logs]
    axes[2].plot(v5e_ep, v5_em, 'g-o', markersize=3, linewidth=1.5, label='Exact Match (%)')
    axes[2].plot(v5e_ep, v5_pm, 'b-s', markersize=3, linewidth=1.5, label='Partial Match (%)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('V5 (ByT5-small, Byte-level)')
    axes[2].legend(loc='center right')
    axes[2].grid(alpha=0.3, linestyle='--')
    best5 = int(np.argmax(v5_em))
    axes[2].axvline(x=v5e_ep[best5], color='red', linestyle='--', alpha=0.5)
    axes[2].annotate(f'Best: {v5_em[best5]:.1f}%\n(ep {v5e_ep[best5]:.0f})',
                     xy=(v5e_ep[best5], v5_em[best5]),
                     xytext=(v5e_ep[best5]+2, v5_em[best5]+1),
                     fontsize=10, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='red'))

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Validation Accuracy Progression During Training', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_v3_v4_v5_em_progression.png")
plt.close()
print("  Saved fig4_v3_v4_v5_em_progression.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5: Hybrid System Analysis — V3 vs V4 vs V5
# ══════════════════════════════════════════════════════════════════════
print("Figure 5: Hybrid System Breakdown...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 5a: V5 Hybrid method distribution (pie) — best model
v5_ret = hybrid_v5.get('method_distribution', {}).get('retrieval', 0)
v5_neu = hybrid_v5.get('method_distribution', {}).get('neural', 0)
labels_pie = [f'Retrieval\n(n={v5_ret})', f'Neural\n(n={v5_neu})']
sizes_pie  = [v5_ret, v5_neu]
colors_pie = ['#4472C4', '#ED7D31']

wedges, texts, autotexts = axes[0].pie(
    sizes_pie, labels=labels_pie, colors=colors_pie,
    autopct='%1.1f%%', startangle=90, explode=(0.05, 0.05),
    textprops={'fontsize': 11})
for t in autotexts:
    t.set_fontweight('bold')
axes[0].set_title('V5 Hybrid: Method Distribution\n(Best Model)')

# 5b: V3 vs V4 vs V5 EM comparison (+ V4 Enhanced)
comp_labels = ['V3\nNeural', 'V3\nHybrid', 'V4\nNeural', 'V4\nHybrid',
               'V4+Enh.\nRetrieval', 'V5\nNeural', 'V5\nHybrid']
comp_em = [v3['test_metrics']['exact_match_accuracy'],
           hybrid_v3['test_metrics']['exact_match_accuracy'],
           v4['test_metrics']['exact_match_accuracy'],
           hybrid_v4['test_metrics']['exact_match_accuracy'],
           hybrid_v4_enhanced['test_metrics']['exact_match_accuracy'],
           v5['test_metrics']['exact_match_accuracy'],
           hybrid_v5['test_metrics']['exact_match_accuracy']]
comp_colors = ['#ED7D31', '#FFC000', '#9B59B6', '#70AD47',
               '#3498DB', '#E74C3C', '#2ECC71']

bb = axes[1].bar(comp_labels, comp_em, color=comp_colors,
                 edgecolor='black', linewidth=0.5, width=0.55)
for bar, val in zip(bb, comp_em):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[1].set_ylabel('Exact Match (%)')
axes[1].set_title('Neural & Hybrid Accuracy Across Versions')
axes[1].set_ylim(0, max(comp_em) * 1.35)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# 5c: BLEU comparison
comp_bleu_labels = ['V3\nNeural', 'V3\nHybrid', 'V4\nNeural', 'V4\nHybrid',
                    'V5\nNeural', 'V5\nHybrid']
comp_bleu = [v3['test_metrics']['corpus_bleu'],
             hybrid_v3['test_metrics']['corpus_bleu'],
             v4['test_metrics']['corpus_bleu'],
             hybrid_v4['test_metrics']['corpus_bleu'],
             v5['test_metrics']['corpus_bleu'],
             hybrid_v5['test_metrics']['corpus_bleu']]
comp_bleu_colors = ['#ED7D31', '#FFC000', '#9B59B6', '#70AD47',
                    '#E74C3C', '#2ECC71']

bb = axes[2].bar(comp_bleu_labels, comp_bleu, color=comp_bleu_colors,
                 edgecolor='black', linewidth=0.5, width=0.55)
for bar, val in zip(bb, comp_bleu):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[2].set_ylabel('Corpus BLEU')
axes[2].set_title('BLEU Score Across Versions')
axes[2].set_ylim(0, max(comp_bleu) * 1.4)
axes[2].grid(axis='y', alpha=0.3, linestyle='--')
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig5_hybrid_analysis.png")
plt.close()
print("  Saved fig5_hybrid_analysis.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 6: Per-Category Performance — V3 vs V4 vs V5 Hybrid
# ══════════════════════════════════════════════════════════════════════
print("Figure 6: Per-Category Performance...")

def category_accuracy(preds):
    total = defaultdict(int)
    correct = defaultdict(int)
    for p in preds:
        cat = p.get('category', 'unknown')
        if cat and str(cat) != 'nan':
            total[cat] += 1
            if p['exact_match']:
                correct[cat] += 1
    return total, correct

cat_tot_v3, cat_cor_v3 = category_accuracy(hybrid_v3_preds)
cat_tot_v4, cat_cor_v4 = category_accuracy(hybrid_v4_preds)
cat_tot_v5, cat_cor_v5 = category_accuracy(hybrid_v5_preds)

# Top 20 categories by V5 count (best model)
all_cats = set(cat_tot_v4.keys()) | set(cat_tot_v5.keys())
top_cats = sorted(all_cats, key=lambda c: -(cat_tot_v5.get(c, 0) + cat_tot_v4.get(c, 0)))[:20]
cat_names = top_cats
cat_em_v3 = [cat_cor_v3[c] / cat_tot_v3[c] * 100 if cat_tot_v3.get(c, 0) > 0 else 0 for c in cat_names]
cat_em_v4 = [cat_cor_v4[c] / cat_tot_v4[c] * 100 if cat_tot_v4.get(c, 0) > 0 else 0 for c in cat_names]
cat_em_v5 = [cat_cor_v5[c] / cat_tot_v5[c] * 100 if cat_tot_v5.get(c, 0) > 0 else 0 for c in cat_names]
cat_sizes = [max(cat_tot_v4.get(c, 0), cat_tot_v5.get(c, 0)) for c in cat_names]

fig, ax = plt.subplots(figsize=(14, 9))
y = np.arange(len(cat_names))
bh = 0.25

ax.barh(y - bh, cat_em_v3, bh, label='V3 Hybrid', color='#FFC000',
        edgecolor='black', linewidth=0.5)
ax.barh(y, cat_em_v4, bh, label='V4 Hybrid', color='#70AD47',
        edgecolor='black', linewidth=0.5)
ax.barh(y + bh, cat_em_v5, bh, label='V5 Hybrid', color='#2ECC71',
        edgecolor='black', linewidth=0.5)

for i, (e3, e4, e5, s) in enumerate(zip(cat_em_v3, cat_em_v4, cat_em_v5, cat_sizes)):
    mx = max(e3, e4, e5)
    ax.text(mx + 0.5, i, f'n={s}', va='center', fontsize=8, color='gray')

ax.set_yticks(y)
ax.set_yticklabels(cat_names, fontsize=10)
ax.set_xlabel('Exact Match Accuracy (%)')
ax.set_title('Per-Category Performance: V3 vs V4 vs V5 Hybrid (Top 20)')
ax.invert_yaxis()
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig6_category_performance.png")
plt.close()
print("  Saved fig6_category_performance.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 7: Dataset Statistics (V3 vs V4 vs V5 data)
# ══════════════════════════════════════════════════════════════════════
print("Figure 7: Dataset Statistics...")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# 7a: Data split (base)
split_labels = ['Train', 'Validation', 'Test']
split_sizes  = [split_v1['train_size'], split_v1['val_size'], split_v1['test_size']]
colors_split = ['#4472C4', '#ED7D31', '#70AD47']
axes[0].pie(split_sizes, labels=split_labels, colors=colors_split,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
axes[0].set_title(f'Base Data Split (n={sum(split_sizes):,})')

# 7b: V3 vs V4 vs V5 training data composition
v4_sr_en = split_v4['task_sr_en']
v4_ta_en = split_v4['task_ta_en']
v4_sent  = split_v4['sent_augment_en']

v5_sr_en = split_v5['task_sr_en']
v5_ta_en = split_v5['task_ta_en']
v5_sent_en = split_v5.get('sent_augment_en', 0)
v5_sent_ta = split_v5.get('sent_augment_ta', 0)

# V3 stacked
axes[1].bar('V3\n(EN only)', split_v3['primary_train'],
            color='#4472C4', edgecolor='black', linewidth=0.5, width=0.3, label='SR\u2192EN')
axes[1].bar('V3\n(EN only)', split_v3['sentence_augmented'],
            bottom=split_v3['primary_train'],
            color='#FFC000', edgecolor='black', linewidth=0.5, width=0.3, label='Sentences (EN)')

# V4 stacked
axes[1].bar('V4\n(EN+Tamil)', v4_sr_en,
            color='#4472C4', edgecolor='black', linewidth=0.5, width=0.3)
axes[1].bar('V4\n(EN+Tamil)', v4_ta_en, bottom=v4_sr_en,
            color='#9B59B6', edgecolor='black', linewidth=0.5, width=0.3, label='TA\u2192EN')
axes[1].bar('V4\n(EN+Tamil)', v4_sent, bottom=v4_sr_en + v4_ta_en,
            color='#FFC000', edgecolor='black', linewidth=0.5, width=0.3)

# V5 stacked
axes[1].bar('V5\n(ByT5)', v5_sr_en,
            color='#4472C4', edgecolor='black', linewidth=0.5, width=0.3)
axes[1].bar('V5\n(ByT5)', v5_ta_en, bottom=v5_sr_en,
            color='#9B59B6', edgecolor='black', linewidth=0.5, width=0.3)
v5_sent_total = v5_sent_en + v5_sent_ta
axes[1].bar('V5\n(ByT5)', v5_sent_total, bottom=v5_sr_en + v5_ta_en,
            color='#FFC000', edgecolor='black', linewidth=0.5, width=0.3)

# Total annotations
axes[1].text(0, split_v3['total_train'] + 300, f"{split_v3['total_train']:,}",
             ha='center', fontsize=10, fontweight='bold')
axes[1].text(1, split_v4['total_train'] + 300, f"{split_v4['total_train']:,}",
             ha='center', fontsize=10, fontweight='bold')
axes[1].text(2, split_v5['total_train'] + 300, f"{split_v5['total_train']:,}",
             ha='center', fontsize=10, fontweight='bold')

axes[1].set_ylabel('Training Examples')
axes[1].set_title('Training Data: V3 vs V4 vs V5')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# 7c: Source/target word-length distribution (using V5 — best model)
src_lens = [len(p['source'].split()) for p in hybrid_v5_preds]
tgt_lens = [len(p['reference'].split()) for p in hybrid_v5_preds]
axes[2].hist(src_lens, bins=range(1, 15), alpha=0.7, label='Source',
             color='#4472C4', edgecolor='black', linewidth=0.5)
axes[2].hist(tgt_lens, bins=range(1, 15), alpha=0.7, label='Target',
             color='#ED7D31', edgecolor='black', linewidth=0.5)
axes[2].set_xlabel('Number of Words')
axes[2].set_ylabel('Count')
axes[2].set_title('Word Count Distribution (Test Set)')
axes[2].legend()
axes[2].grid(alpha=0.3, linestyle='--')
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig7_dataset_statistics.png")
plt.close()
print("  Saved fig7_dataset_statistics.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 8: Error Analysis — V5 Hybrid (Best Model)
# ══════════════════════════════════════════════════════════════════════
print("Figure 8: Error Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 8a: Predicted vs Reference length scatter
ref_lens  = [len(p['reference']) for p in hybrid_v5_preds]
pred_lens = [len(p['prediction']) for p in hybrid_v5_preds]

axes[0].scatter(ref_lens, pred_lens, alpha=0.15, s=8, c='#4472C4')
ml = max(max(ref_lens), max(pred_lens))
axes[0].plot([0, ml], [0, ml], 'r--', linewidth=1, alpha=0.7, label='Perfect')
axes[0].set_xlabel('Reference Length (chars)')
axes[0].set_ylabel('Prediction Length (chars)')
axes[0].set_title('Prediction vs Reference Length (V5 Hybrid)')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# 8b: Error type distribution
exact = partial = near_miss = wrong = 0
for p in hybrid_v5_preds:
    if p['exact_match']:
        exact += 1
        continue
    ref_w  = set(p['reference'].lower().split())
    pred_w = set(p['prediction'].lower().split())
    if ref_w & pred_w:
        partial += 1
    elif p['prediction'] and abs(len(p['prediction']) - len(p['reference'])) <= 3:
        near_miss += 1
    else:
        wrong += 1

err_labels = ['Exact\nMatch', 'Partial\nMatch', 'Near\nMiss', 'Wrong']
err_counts = [exact, partial, near_miss, wrong]
err_colors = ['#70AD47', '#FFC000', '#ED7D31', '#C00000']

bb = axes[1].bar(err_labels, err_counts, color=err_colors,
                 edgecolor='black', linewidth=0.5, width=0.6)
n_total = len(hybrid_v5_preds)
for bar, cnt in zip(bb, err_counts):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f'{cnt}\n({cnt/n_total*100:.1f}%)',
                 ha='center', va='bottom', fontsize=10)
axes[1].set_ylabel('Count')
axes[1].set_title('Error Type Distribution (V5 Hybrid \u2014 Best Model)')
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig8_error_analysis.png")
plt.close()
print("  Saved fig8_error_analysis.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 9: Improvement Progression Timeline (V1 → V5 Hybrid)
# ══════════════════════════════════════════════════════════════════════
print("Figure 9: Improvement Progression...")

fig, ax = plt.subplots(figsize=(16, 6))

versions = ['V1\nChar-GRU\nSeq2Seq', 'V2\nTransformer\n+BPE',
            'V3\nT5-small\n(EN only)', 'V3 Hybrid\nT5+Retrieval',
            'V4\nT5+Tamil\n(Multilingual)', 'V4 Hybrid\nT5+Tamil\n+Retrieval',
            'V5\nByT5\n(Byte-level)', 'V5 Hybrid\nByT5+Enhanced\nRetrieval']
em_prog = [v1['exact_match_accuracy'],
           v2['exact_match_accuracy'],
           v3['test_metrics']['exact_match_accuracy'],
           hybrid_v3['test_metrics']['exact_match_accuracy'],
           v4['test_metrics']['exact_match_accuracy'],
           hybrid_v4['test_metrics']['exact_match_accuracy'],
           v5['test_metrics']['exact_match_accuracy'],
           hybrid_v5['test_metrics']['exact_match_accuracy']]
xp = list(range(1, len(versions) + 1))

ax.plot(xp, em_prog, 'o-', color='#4472C4', linewidth=2.5, markersize=12,
        markerfacecolor='white', markeredgewidth=2.5, markeredgecolor='#4472C4', zorder=5)

for i, (px, py) in enumerate(zip(xp, em_prog)):
    ax.annotate(f'{py:.2f}%', (px, py), textcoords="offset points",
                xytext=(0, 18), ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='lightyellow', edgecolor='gray'))

ax.fill_between(xp, em_prog, alpha=0.12, color='#4472C4')
ax.set_xticks(xp)
ax.set_xticklabels(versions, fontsize=8)
ax.set_ylabel('Exact Match Accuracy (%)')
ax.set_title('Model Improvement Progression (V1 \u2192 V5)')
ax.set_ylim(0, max(em_prog) * 1.45)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Overall improvement arrow
best_em_val = hybrid_v5['test_metrics']['exact_match_accuracy']
first_em_val = v1['exact_match_accuracy']
ax.annotate('', xy=(len(xp), best_em_val), xytext=(1, first_em_val),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(4.0, best_em_val * 0.75, f'{best_em_val/first_em_val:.1f}x improvement',
        fontsize=12, color='red', fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='red', alpha=0.8))

plt.tight_layout()
plt.savefig(f"{OUT}/fig9_improvement_progression.png")
plt.close()
print("  Saved fig9_improvement_progression.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 10: Qualitative Examples Table (V5 Hybrid — Best Model)
# ══════════════════════════════════════════════════════════════════════
print("Figure 10: Qualitative Examples Table...")

correct_ex = [p for p in hybrid_v5_preds if p['exact_match']][:5]
near_ex = []
for p in hybrid_v5_preds:
    if not p['exact_match']:
        ref_w  = set(p['reference'].lower().split())
        pred_w = set(p['prediction'].lower().split())
        if ref_w & pred_w and len(ref_w) <= 3:
            near_ex.append(p)
    if len(near_ex) >= 3:
        break
wrong_ex = [p for p in hybrid_v5_preds
            if not p['exact_match'] and len(p['reference'].split()) == 1
            and len(p['prediction'].split()) == 1][:3]

table_data = []
for p in correct_ex:
    table_data.append([p['source'], p['reference'], p['prediction'],
                       'YES', p.get('method', '\u2014')])
for p in near_ex:
    table_data.append([p['source'], p['reference'], p['prediction'],
                       'PARTIAL', p.get('method', '\u2014')])
for p in wrong_ex:
    table_data.append([p['source'], p['reference'], p['prediction'],
                       'NO', p.get('method', '\u2014')])

fig, ax = plt.subplots(figsize=(14, max(4, len(table_data) * 0.45 + 1)))
ax.axis('off')

cols = ['Source (Romanized)', 'Reference', 'Prediction', 'Match', 'Method']
tbl = ax.table(cellText=table_data, colLabels=cols, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.0, 1.5)

for j in range(len(cols)):
    tbl[(0, j)].set_facecolor('#4472C4')
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')

for i in range(1, len(table_data) + 1):
    sym = table_data[i - 1][3]
    if sym == 'YES':
        bg = '#E2EFDA'
    elif sym == 'PARTIAL':
        bg = '#FFF2CC'
    else:
        bg = '#FCE4EC'
    for j in range(len(cols)):
        tbl[(i, j)].set_facecolor(bg)

ax.set_title('Qualitative Translation Examples (V5 Hybrid: ByT5 + Enhanced Retrieval \u2014 Best Model)',
             fontsize=13, pad=20)
plt.tight_layout()
plt.savefig(f"{OUT}/fig10_qualitative_examples.png")
plt.close()
print("  Saved fig10_qualitative_examples.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 11: Multilingual & Cross-Version Impact Analysis
# ══════════════════════════════════════════════════════════════════════
print("Figure 11: Multilingual Impact Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 11a: Data volume V3 vs V4 vs V5
dv_labels = ['V3 (EN only)', 'V4 (EN+Tamil)', 'V5 (ByT5)']
dv_vals   = [split_v3['total_train'], split_v4['total_train'], split_v5['total_train']]
pct_inc_v4 = (dv_vals[1] - dv_vals[0]) / dv_vals[0] * 100

bb = axes[0].bar(dv_labels, dv_vals, color=['#ED7D31', '#9B59B6', '#2ECC71'],
                 edgecolor='black', linewidth=0.5, width=0.50)
for bar, val in zip(bb, dv_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 250,
                 f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].annotate(f'+{pct_inc_v4:.0f}% vs V3',
                 xy=(1, dv_vals[1]), xytext=(0.3, dv_vals[1]*0.65),
                 fontsize=10, color='green', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
axes[0].set_ylabel('Training Examples')
axes[0].set_title('Training Data Volume')
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# 11b: Metric % change V4 Hybrid → V5 Hybrid
met_names = ['Neural\nEM', 'Neural\nBLEU', 'Neural\nchrF',
             'Hybrid\nEM', 'Hybrid\nBLEU', 'Hybrid\nchrF']
v4_v = [v4['test_metrics']['exact_match_accuracy'],
        v4['test_metrics']['corpus_bleu'],
        v4['test_metrics']['avg_chrf'],
        hybrid_v4['test_metrics']['exact_match_accuracy'],
        hybrid_v4['test_metrics']['corpus_bleu'],
        hybrid_v4['test_metrics']['avg_chrf']]
v5_v = [v5['test_metrics']['exact_match_accuracy'],
        v5['test_metrics']['corpus_bleu'],
        v5['test_metrics']['avg_chrf'],
        hybrid_v5['test_metrics']['exact_match_accuracy'],
        hybrid_v5['test_metrics']['corpus_bleu'],
        hybrid_v5['test_metrics']['avg_chrf']]
pct_ch = [(v5_v[i] - v4_v[i]) / max(v4_v[i], 0.01) * 100 for i in range(len(v4_v))]

bar_c = ['#70AD47' if p >= 0 else '#C00000' for p in pct_ch]
bb = axes[1].bar(met_names, pct_ch, color=bar_c, edgecolor='black', linewidth=0.5, width=0.6)
for bar, val in zip(bb, pct_ch):
    off = 1.5 if val >= 0 else -4
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + off,
                 f'{val:+.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[1].axhline(y=0, color='black', linewidth=0.8)
axes[1].set_ylabel('Change (%)')
axes[1].set_title('V4 \u2192 V5: Metric Changes (ByT5 Byte-level Impact)')
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig11_multilingual_impact.png")
plt.close()
print("  Saved fig11_multilingual_impact.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 12: V5 ByT5 vs V4 T5 — Byte-level vs Subword Comparison
# ══════════════════════════════════════════════════════════════════════
print("Figure 12: ByT5 vs T5 Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 12a: Architecture comparison — model size
arch_labels = ['V4 T5-small\n(Subword)', 'V5 ByT5-small\n(Byte-level)']
arch_params = [60, 300]  # million params
arch_colors = ['#9B59B6', '#E74C3C']

bb = axes[0].bar(arch_labels, arch_params, color=arch_colors,
                 edgecolor='black', linewidth=0.5, width=0.50)
for bar, val in zip(bb, arch_params):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{val}M', ha='center', va='bottom', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Parameters (Millions)')
axes[0].set_title('Model Size: Subword vs Byte-level')
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# 12b: Neural EM / Hybrid EM side by side
comp_labels_12 = ['V4\nNeural', 'V5\nNeural', 'V4\nHybrid', 'V5\nHybrid',
                  'V4+Enh.\nRetrieval']
comp_em_12 = [v4['test_metrics']['exact_match_accuracy'],
              v5['test_metrics']['exact_match_accuracy'],
              hybrid_v4['test_metrics']['exact_match_accuracy'],
              hybrid_v5['test_metrics']['exact_match_accuracy'],
              hybrid_v4_enhanced['test_metrics']['exact_match_accuracy']]
comp_colors_12 = ['#9B59B6', '#E74C3C', '#70AD47', '#2ECC71', '#3498DB']

bb = axes[1].bar(comp_labels_12, comp_em_12, color=comp_colors_12,
                 edgecolor='black', linewidth=0.5, width=0.55)
for bar, val in zip(bb, comp_em_12):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Exact Match (%)')
axes[1].set_title('V4 (T5) vs V5 (ByT5): Accuracy Comparison')
axes[1].set_ylim(0, max(comp_em_12) * 1.35)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Highlight best
best_12 = int(np.argmax(comp_em_12))
bb[best_12].set_edgecolor('#C00000')
bb[best_12].set_linewidth(2.5)

# 12c: Training time vs accuracy scatter
train_times = [22, 7, 36, 58, 119]  # approx mins per version
train_ems = [v1['exact_match_accuracy'], v2['exact_match_accuracy'],
             v3['test_metrics']['exact_match_accuracy'],
             v4['test_metrics']['exact_match_accuracy'],
             v5['test_metrics']['exact_match_accuracy']]
train_labels_12 = ['V1', 'V2', 'V3', 'V4', 'V5']
train_colors_12 = ['#4472C4', '#5B9BD5', '#ED7D31', '#9B59B6', '#E74C3C']

for t, e, l, c in zip(train_times, train_ems, train_labels_12, train_colors_12):
    axes[2].scatter(t, e, c=c, s=150, zorder=5, edgecolors='black', linewidth=0.8)
    axes[2].annotate(l, (t, e), textcoords="offset points", xytext=(8, 5),
                     fontsize=11, fontweight='bold', color=c)
axes[2].set_xlabel('Training Time (minutes)')
axes[2].set_ylabel('Neural EM (%)')
axes[2].set_title('Training Time vs Accuracy')
axes[2].grid(alpha=0.3, linestyle='--')
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

plt.suptitle('V5 ByT5 (Byte-level) vs V4 T5 (Subword) \u2014 Architecture Comparison',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/fig12_byt5_vs_t5_comparison.png")
plt.close()
print("  Saved fig12_byt5_vs_t5_comparison.png")


# ══════════════════════════════════════════════════════════════════════
# LaTeX TABLES
# ══════════════════════════════════════════════════════════════════════
print("\nGenerating LaTeX tables...")

v4_ret = hybrid_v4.get('method_distribution', {}).get('retrieval', 0)
v4_neu = hybrid_v4.get('method_distribution', {}).get('neural', 0)
v4_ret_pct = v4_ret / (v4_ret + v4_neu) * 100 if (v4_ret + v4_neu) > 0 else 0
v4_neu_pct = v4_neu / (v4_ret + v4_neu) * 100 if (v4_ret + v4_neu) > 0 else 0

v5_ret_pct = v5_ret / (v5_ret + v5_neu) * 100 if (v5_ret + v5_neu) > 0 else 0
v5_neu_pct = v5_neu / (v5_ret + v5_neu) * 100 if (v5_ret + v5_neu) > 0 else 0

latex = r"""\begin{table}[htbp]
\centering
\caption{Performance Comparison of Sourashtra-English Translation Models}
\label{tab:results}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Params} & \textbf{EM (%%)} & \textbf{BLEU} & \textbf{chrF} & \textbf{CER} & \textbf{Time} \\
\midrule
V1: Char-GRU Seq2Seq & $\sim$1M & %.2f & %.2f & %.2f & %.2f & $\sim$22 min \\
V2: Transformer + BPE & $\sim$5M & %.2f & %.2f & %.2f & %.2f & $\sim$7 min \\
V3: T5-small (EN only) & 60M & %.2f & %.2f & %.2f & %.2f & $\sim$36 min \\
V3: Hybrid (T5+Retrieval) & 60M & %.2f & %.2f & %.2f & -- & -- \\
V4: T5+Tamil (Multilingual) & 60M & %.2f & %.2f & %.2f & %.2f & $\sim$58 min \\
V4: Hybrid (T5+Tamil+Ret.) & 60M & %.2f & %.2f & %.2f & %.2f & -- \\
V5: ByT5-small (Byte-level) & 300M & %.2f & %.2f & %.2f & %.2f & $\sim$119 min \\
\textbf{V5: Hybrid (ByT5+Enh.Ret.)} & \textbf{300M} & \textbf{%.2f} & \textbf{%.2f} & \textbf{%.2f} & \textbf{%.2f} & -- \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Dataset Statistics and Training Configuration}
\label{tab:dataset}
\begin{tabular}{lr}
\toprule
\textbf{Statistic} & \textbf{Value} \\
\midrule
Total word pairs & %s \\
Training set (base SR$\rightarrow$EN) & %s \\
V3 training (EN + augm.) & %s \\
V4 training (EN + Tamil + augm.) & %s \\
V5 training (ByT5 + Tamil + augm.) & %s \\
Validation set & %s \\
Test set & %s \\
English sentence augmentation & %s \\
Tamil $\rightarrow$ English pairs & %s \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Hybrid System Configuration (V4 vs V5)}
\label{tab:hybrid}
\begin{tabular}{lrr}
\toprule
\textbf{Component} & \textbf{V4 Hybrid} & \textbf{V5 Hybrid} \\
\midrule
Neural model & T5-small (60M) & ByT5-small (300M) \\
Tokenization & Subword (SentencePiece) & Byte-level (UTF-8) \\
Training precision & FP16 & BF16 \\
Retrieval method & Char n-gram Jaccard & Jaccard+Lev+Prefix \\
Confidence threshold & %.2f & %.2f \\
Retrieval coverage & %.1f%%%% & %.1f%%%% \\
Neural coverage & %.1f%%%% & %.1f%%%% \\
Hybrid EM accuracy & %.2f%%%% & \textbf{%.2f%%%%} \\
Overall improvement (V1$\rightarrow$best) & %.1f$\times$ & \textbf{%.1f$\times$} \\
\bottomrule
\end{tabular}
\end{table}
""" % (
    # Table 1
    v1['exact_match_accuracy'], v1['corpus_bleu'], v1['avg_chrf'], v1['avg_cer'],
    v2['exact_match_accuracy'], v2['corpus_bleu'], v2['avg_chrf'], v2['avg_cer'],
    v3['test_metrics']['exact_match_accuracy'], v3['test_metrics']['corpus_bleu'],
    v3['test_metrics']['avg_chrf'], v3['test_metrics']['avg_cer'],
    hybrid_v3['test_metrics']['exact_match_accuracy'],
    hybrid_v3['test_metrics']['corpus_bleu'],
    hybrid_v3['test_metrics']['avg_chrf'],
    v4['test_metrics']['exact_match_accuracy'], v4['test_metrics']['corpus_bleu'],
    v4['test_metrics']['avg_chrf'], v4['test_metrics']['avg_cer'],
    hybrid_v4['test_metrics']['exact_match_accuracy'],
    hybrid_v4['test_metrics']['corpus_bleu'],
    hybrid_v4['test_metrics']['avg_chrf'], hybrid_v4['test_metrics']['avg_cer'],
    v5['test_metrics']['exact_match_accuracy'], v5['test_metrics']['corpus_bleu'],
    v5['test_metrics']['avg_chrf'], v5['test_metrics']['avg_cer'],
    hybrid_v5['test_metrics']['exact_match_accuracy'],
    hybrid_v5['test_metrics']['corpus_bleu'],
    hybrid_v5['test_metrics']['avg_chrf'], hybrid_v5['test_metrics']['avg_cer'],
    # Table 2
    f"{split_v1['total_pairs']:,}",
    f"{split_v1['train_size']:,}",
    f"{split_v3['total_train']:,}",
    f"{split_v4['total_train']:,}",
    f"{split_v5['total_train']:,}",
    f"{split_v1['val_size']:,}",
    f"{split_v1['test_size']:,}",
    f"{split_v3['sentence_augmented']:,}",
    f"{split_v4['task_ta_en']:,}",
    # Table 3
    hybrid_v4['best_threshold'],
    hybrid_v5['best_threshold'],
    v4_ret_pct, v5_ret_pct,
    v4_neu_pct, v5_neu_pct,
    hybrid_v4['test_metrics']['exact_match_accuracy'],
    hybrid_v5['test_metrics']['exact_match_accuracy'],
    hybrid_v4['test_metrics']['exact_match_accuracy'] / v1['exact_match_accuracy'],
    hybrid_v5['test_metrics']['exact_match_accuracy'] / v1['exact_match_accuracy'],
)

with open(f"{OUT}/latex_tables.tex", 'w', encoding='utf-8') as f:
    f.write(latex)
print("  Saved latex_tables.tex")


# ══════════════════════════════════════════════════════════════════════
# RESULTS REPORT
# ══════════════════════════════════════════════════════════════════════
print("\nGenerating results report...")

best_em = hybrid_v5['test_metrics']['exact_match_accuracy']
best_bleu = hybrid_v5['test_metrics']['corpus_bleu']
best_chrf = hybrid_v5['test_metrics']['avg_chrf']

report = f"""# Sourashtra-English Translation: Complete Results Report

## Model Performance Summary (V1 → V5)

| Model | Params | EM (%) | BLEU | chrF | Training |
|-------|--------|--------|------|------|----------|
| V1: Char-GRU Seq2Seq | ~1M | {v1['exact_match_accuracy']:.2f} | {v1['corpus_bleu']:.2f} | {v1['avg_chrf']:.2f} | ~22 min |
| V2: Transformer + BPE | ~5M | {v2['exact_match_accuracy']:.2f} | {v2['corpus_bleu']:.2f} | {v2['avg_chrf']:.2f} | ~7 min |
| V3: T5-small (EN only) | 60M | {v3['test_metrics']['exact_match_accuracy']:.2f} | {v3['test_metrics']['corpus_bleu']:.2f} | {v3['test_metrics']['avg_chrf']:.2f} | ~36 min |
| V3: Hybrid (T5+Retrieval) | 60M | {hybrid_v3['test_metrics']['exact_match_accuracy']:.2f} | {hybrid_v3['test_metrics']['corpus_bleu']:.2f} | {hybrid_v3['test_metrics']['avg_chrf']:.2f} | — |
| V4: T5+Tamil (Multilingual) | 60M | {v4['test_metrics']['exact_match_accuracy']:.2f} | {v4['test_metrics']['corpus_bleu']:.2f} | {v4['test_metrics']['avg_chrf']:.2f} | ~58 min |
| V4: Hybrid (T5+Tamil+Ret.) | 60M | {hybrid_v4['test_metrics']['exact_match_accuracy']:.2f} | {hybrid_v4['test_metrics']['corpus_bleu']:.2f} | {hybrid_v4['test_metrics']['avg_chrf']:.2f} | — |
| V4+Enhanced Retrieval | 60M | {hybrid_v4_enhanced['test_metrics']['exact_match_accuracy']:.2f} | {hybrid_v4_enhanced['test_metrics']['corpus_bleu']:.2f} | {hybrid_v4_enhanced['test_metrics']['avg_chrf']:.2f} | — |
| V5: ByT5-small (Byte-level) | 300M | {v5['test_metrics']['exact_match_accuracy']:.2f} | {v5['test_metrics']['corpus_bleu']:.2f} | {v5['test_metrics']['avg_chrf']:.2f} | ~119 min |
| **V5: Hybrid (ByT5+Enh.Ret.)** | **300M** | **{best_em:.2f}** | **{best_bleu:.2f}** | **{best_chrf:.2f}** | — |

## Key Findings

1. **Best Model: V5 Hybrid** achieves **{best_em:.2f}% EM** — the highest across all versions
2. **Overall improvement**: {best_em/v1['exact_match_accuracy']:.1f}x from V1 to V5 Hybrid
3. **ByT5 byte-level advantage**: V5 Neural EM ({v5['test_metrics']['exact_match_accuracy']:.2f}%) surpasses all V4 variants including V4 Hybrid ({hybrid_v4['test_metrics']['exact_match_accuracy']:.2f}%)
4. **Enhanced retrieval**: Jaccard(0.45) + Levenshtein(0.45) + Prefix(0.10) scoring
   - V4+Enhanced Retrieval: {hybrid_v4_enhanced['test_metrics']['exact_match_accuracy']:.2f}% (vs V4 Hybrid {hybrid_v4['test_metrics']['exact_match_accuracy']:.2f}%)
   - V5 Hybrid: {best_em:.2f}% (best overall)
5. **V5 architecture**: ByT5-small (300M params), BF16 training, ~119 min
   - Byte-level tokenization eliminates subword segmentation issues
   - 5x more parameters than T5-small but significantly better accuracy
6. **Tamil cross-lingual transfer** (V4): Adding {split_v4['task_ta_en']:,} Tamil→EN pairs (+{(split_v4['total_train']-split_v3['total_train'])/split_v3['total_train']*100:.0f}% training data)
   - Hybrid EM improved: {hybrid_v3['test_metrics']['exact_match_accuracy']:.2f}% → {hybrid_v4['test_metrics']['exact_match_accuracy']:.2f}% (+{hybrid_v4['test_metrics']['exact_match_accuracy']-hybrid_v3['test_metrics']['exact_match_accuracy']:.2f}%)
7. **Retrieval distribution**: V5 Hybrid uses retrieval for {v5_ret}/{v5_ret+v5_neu} ({v5_ret_pct:.1f}%) predictions vs V4's {v4_ret}/{v4_ret+v4_neu} ({v4_ret_pct:.1f}%)

## Figures Generated

| # | File | Description |
|---|------|-------------|
| 1 | fig1_model_comparison_em.png | Main result — 8 models EM comparison |
| 2 | fig2_multi_metric_comparison.png | EM / BLEU / chrF for all 8 models |
| 3 | fig3_training_loss_curves.png | Loss curves (V1/V2/V3/V4/V5) |
| 4 | fig4_v3_v4_v5_em_progression.png | V3 vs V4 vs V5 validation EM over epochs |
| 5 | fig5_hybrid_analysis.png | V3 vs V4 vs V5 hybrid breakdown |
| 6 | fig6_category_performance.png | V3 vs V4 vs V5 per-category accuracy |
| 7 | fig7_dataset_statistics.png | Dataset composition + multilingual |
| 8 | fig8_error_analysis.png | Error type analysis (V5 Hybrid — best) |
| 9 | fig9_improvement_progression.png | V1→V5 improvement timeline |
| 10 | fig10_qualitative_examples.png | Example translations (V5 Hybrid) |
| 11 | fig11_multilingual_impact.png | V4→V5 metric impact analysis |
| 12 | fig12_byt5_vs_t5_comparison.png | ByT5 vs T5 architecture comparison |
| — | latex_tables.tex | IEEE LaTeX tables (3 tables, V5 included) |
"""

with open(f"{OUT}/RESULTS_REPORT.md", 'w', encoding='utf-8') as f:
    f.write(report)
print("  Saved RESULTS_REPORT.md")


# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"Output directory: {OUT}/")
print(f"Total files: 12 figures + LaTeX tables + report = 14 files")
print(f"\nBest model: V5 Hybrid = {best_em:.2f}% EM")
print(f"Overall improvement: {best_em/v1['exact_match_accuracy']:.1f}x (V1 → V5)")
