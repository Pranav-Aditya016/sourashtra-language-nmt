# Sourashtra-English Translation: Complete Results Report

## Model Performance Summary (V1 → V5)

| Model | Params | EM (%) | BLEU | chrF | Training |
|-------|--------|--------|------|------|----------|
| V1: Char-GRU Seq2Seq | ~1M | 0.42 | 0.00 | 8.81 | ~22 min |
| V2: Transformer + BPE | ~5M | 2.56 | 0.00 | 11.78 | ~7 min |
| V3: T5-small (EN only) | 60M | 6.01 | 4.72 | 18.47 | ~36 min |
| V3: Hybrid (T5+Retrieval) | 60M | 7.47 | 5.36 | 20.86 | — |
| V4: T5+Tamil (Multilingual) | 60M | 5.80 | 3.57 | 18.27 | ~58 min |
| V4: Hybrid (T5+Tamil+Ret.) | 60M | 7.68 | 5.28 | 21.01 | — |
| V4+Enhanced Retrieval | 60M | 7.84 | 4.64 | 21.18 | — |
| V5: ByT5-small (Byte-level) | 300M | 9.25 | 4.70 | 22.32 | ~119 min |
| **V5: Hybrid (ByT5+Enh.Ret.)** | **300M** | **9.61** | **4.57** | **22.95** | — |

## Key Findings

1. **Best Model: V5 Hybrid** achieves **9.61% EM** — the highest across all versions
2. **Overall improvement**: 22.9x from V1 to V5 Hybrid
3. **ByT5 byte-level advantage**: V5 Neural EM (9.25%) surpasses all V4 variants including V4 Hybrid (7.68%)
4. **Enhanced retrieval**: Jaccard(0.45) + Levenshtein(0.45) + Prefix(0.10) scoring
   - V4+Enhanced Retrieval: 7.84% (vs V4 Hybrid 7.68%)
   - V5 Hybrid: 9.61% (best overall)
5. **V5 architecture**: ByT5-small (300M params), BF16 training, ~119 min
   - Byte-level tokenization eliminates subword segmentation issues
   - 5x more parameters than T5-small but significantly better accuracy
6. **Tamil cross-lingual transfer** (V4): Adding 9,568 Tamil→EN pairs (+85% training data)
   - Hybrid EM improved: 7.47% → 7.68% (+0.21%)
7. **Retrieval distribution**: V5 Hybrid uses retrieval for 462/1914 (24.1%) predictions vs V4's 1134/1914 (59.2%)

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
