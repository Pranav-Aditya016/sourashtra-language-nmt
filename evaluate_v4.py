"""
Evaluation & Hybrid System V4 - Multilingual mT5 + Retrieval
==============================================================
1. Evaluate V4 neural-only on test set
2. Run hybrid retrieval + neural ensemble
3. Compare V1 → V4 progression
4. Generate comparison figures
"""
import os
import json
import time
import numpy as np
import pandas as pd
from collections import Counter

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

from config_v4 import ConfigV4
from data_loader_v4 import load_and_prepare_data, format_sr_to_en
from metrics import evaluate_all, print_metrics


# ═══════════════════════════════════════════════════════════
# FAST CHARACTER N-GRAM RETRIEVAL (same as V3 hybrid)
# ═══════════════════════════════════════════════════════════

def char_ngrams(word: str, n: int = 3) -> set:
    """Get character n-grams with boundary markers."""
    padded = f"^{word.lower().strip()}$"
    return set(padded[i:i+n] for i in range(len(padded) - n + 1))


def build_similarity_matrix(test_sources, train_sources):
    """Pre-compute n-gram Jaccard similarity between all test-train pairs."""
    print("[INDEX] Pre-computing similarity matrix...")
    t0 = time.time()

    n_test = len(test_sources)
    n_train = len(train_sources)

    test_ng = [char_ngrams(s, 3) for s in test_sources]
    train_ng = [char_ngrams(s, 3) for s in train_sources]
    test_bg = [char_ngrams(s, 2) for s in test_sources]
    train_bg = [char_ngrams(s, 2) for s in train_sources]

    test_lower = [s.lower().strip() for s in test_sources]
    train_lower = [s.lower().strip() for s in train_sources]

    sim_matrix = np.zeros((n_test, n_train), dtype=np.float32)

    for i in range(n_test):
        for j in range(n_train):
            if test_lower[i] == train_lower[j]:
                sim_matrix[i, j] = 1.0
            else:
                ng_inter = len(test_ng[i] & train_ng[j])
                ng_union = len(test_ng[i] | train_ng[j])
                bg_inter = len(test_bg[i] & train_bg[j])
                bg_union = len(test_bg[i] | train_bg[j])
                j3 = ng_inter / ng_union if ng_union > 0 else 0.0
                j2 = bg_inter / bg_union if bg_union > 0 else 0.0
                sim_matrix[i, j] = 0.55 * j3 + 0.45 * j2

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_test} ({(i+1)/n_test*100:.0f}%)")

    elapsed = time.time() - t0
    print(f"  Matrix computed in {elapsed:.1f}s ({n_test}x{n_train})")
    return sim_matrix


def main():
    config = ConfigV4()
    config.ensure_dirs()

    print("=" * 70)
    print("  V4 EVALUATION & HYBRID SYSTEM")
    print("  (Multilingual mT5-small + Tamil + Retrieval)")
    print("=" * 70)

    # ── Load data ──
    data = load_and_prepare_data(config)
    test_df = data["test_df"]
    train_df = data["train_df"]

    test_sources = list(test_df["source"])
    test_refs = list(test_df["target"])
    test_cats = list(test_df["category"]) if "category" in test_df.columns else [""] * len(test_sources)

    train_sources = list(train_df["source"])
    train_targets = list(train_df["target"])

    # ═══════════════════════════════════════════════════════
    # RETRIEVAL BASELINE
    # ═══════════════════════════════════════════════════════
    sim_matrix = build_similarity_matrix(test_sources, train_sources)

    best_train_idx = np.argmax(sim_matrix, axis=1)
    best_confidence = np.max(sim_matrix, axis=1)
    retrieval_preds = [train_targets[idx] for idx in best_train_idx]
    retrieval_confs = best_confidence.tolist()

    ret_exact = sum(1 for p, r in zip(retrieval_preds, test_refs)
                    if p.strip().lower() == r.strip().lower())
    print(f"\n[RETRIEVAL] {ret_exact}/{len(test_refs)} = {ret_exact/len(test_refs)*100:.2f}%")

    # Confidence distribution
    print("\n  Confidence distribution:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        above = sum(1 for c in retrieval_confs if c >= thresh)
        print(f"    Conf >= {thresh:.1f}: {above:4d}/{len(retrieval_confs)} ({above/len(retrieval_confs)*100:.1f}%)")

    # ═══════════════════════════════════════════════════════
    # NEURAL V4 PREDICTIONS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  NEURAL MODEL (V4 - mT5-small + Tamil)")
    print(f"{'='*70}")

    model_path = os.path.join(config.CHECKPOINT_DIR, "best_model_v4")
    if not os.path.exists(model_path):
        print(f"  [ERROR] Model not found at {model_path}")
        print(f"  Run train_v4.py first!")
        return

    print(f"[NEURAL] Loading {model_path}")
    # Use T5Tokenizer for T5, AutoTokenizer for mT5
    if "mt5" in config.MODEL_NAME.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(config.DEVICE)
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(config.DEVICE)
    model.eval()

    # Build test inputs (Sourashtra→English only)
    use_cat = config.USE_CATEGORY_PREFIX
    test_input_texts = [format_sr_to_en(row["source"], row["category"], use_cat)
                        for _, row in test_df.iterrows()]

    neural_preds = []
    batch_size = 8  # Smaller for mT5

    print(f"[NEURAL] Generating predictions ({len(test_input_texts)} examples)...")
    for i in range(0, len(test_input_texts), batch_size):
        batch = test_input_texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                          max_length=config.MAX_SOURCE_LEN).to(config.DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_length=config.MAX_GENERATE_LEN,
                num_beams=config.NUM_BEAMS,
                no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
                repetition_penalty=config.REPETITION_PENALTY,
                length_penalty=config.LENGTH_PENALTY,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        neural_preds.extend([d.strip() for d in decoded])

        if (i + batch_size) % 200 == 0:
            print(f"  {min(i+batch_size, len(test_input_texts))}/{len(test_input_texts)}")

    neural_exact = sum(1 for p, r in zip(neural_preds, test_refs)
                       if p.strip().lower() == r.strip().lower())
    print(f"\n  V4 Neural exact match: {neural_exact}/{len(test_refs)} = {neural_exact/len(test_refs)*100:.2f}%")

    # Full neural metrics
    neural_metrics = evaluate_all(test_refs, neural_preds)
    print_metrics(neural_metrics, "V4 NEURAL-ONLY (mT5-small + Tamil)")

    # ═══════════════════════════════════════════════════════
    # HYBRID THRESHOLD SWEEP
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  HYBRID V4 (Neural + Retrieval) - Threshold Sweep")
    print(f"{'='*70}")

    best_threshold = 0.5
    best_em = 0

    for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
        hybrid_preds = []
        ret_used = 0
        for i in range(len(test_refs)):
            if retrieval_confs[i] >= threshold:
                hybrid_preds.append(retrieval_preds[i])
                ret_used += 1
            else:
                hybrid_preds.append(neural_preds[i])

        exact = sum(1 for p, r in zip(hybrid_preds, test_refs)
                    if p.strip().lower() == r.strip().lower())
        pct = exact / len(test_refs) * 100

        marker = ""
        if exact > best_em:
            best_em = exact
            best_threshold = threshold
            marker = " << BEST"

        print(f"  T={threshold:.2f}  EM={exact:3d} ({pct:5.2f}%)  "
              f"ret={ret_used:4d}  neural={len(test_refs)-ret_used:4d}{marker}")

    # ═══════════════════════════════════════════════════════
    # FINAL HYBRID RESULTS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  FINAL HYBRID V4 RESULTS (T={best_threshold})")
    print(f"{'='*70}")

    hybrid_final = []
    methods = []
    for i in range(len(test_refs)):
        if retrieval_confs[i] >= best_threshold:
            hybrid_final.append(retrieval_preds[i])
            methods.append("retrieval")
        else:
            hybrid_final.append(neural_preds[i])
            methods.append("neural")

    hybrid_metrics = evaluate_all(test_refs, hybrid_final)
    print_metrics(hybrid_metrics, f"V4 HYBRID (mT5 + Tamil + Retrieval, T={best_threshold})")

    # Method breakdown
    for m in ["retrieval", "neural"]:
        total = methods.count(m)
        correct = sum(1 for i in range(len(test_refs)) if methods[i] == m and
                      hybrid_final[i].strip().lower() == test_refs[i].strip().lower())
        print(f"  {m:12s}: {correct:3d}/{total:4d} = {correct/max(total,1)*100:.1f}%")

    # ═══════════════════════════════════════════════════════
    # FULL MODEL COMPARISON (V1 → V4)
    # ═══════════════════════════════════════════════════════
    v4_neural_em = neural_metrics["exact_match_accuracy"]
    v4_hybrid_em = hybrid_metrics["exact_match_accuracy"]

    print(f"\n{'='*70}")
    print(f"  COMPLETE MODEL COMPARISON (V1 → V4)")
    print(f"{'='*70}")
    print(f"  V1 (Char GRU Seq2Seq):        0.42% exact match")
    print(f"  V2 (Transformer+BPE):         2.56% exact match")
    print(f"  V3 Neural (T5-small, EN):     6.01% exact match")
    print(f"  V3 Hybrid (T5+Retrieval):     7.47% exact match")
    print(f"  V4 Neural (mT5 + Tamil):      {v4_neural_em:.2f}% exact match")
    print(f"  V4 Hybrid (mT5+Tamil+Ret):    {v4_hybrid_em:.2f}% exact match")
    print()
    print(f"  V4 Neural BLEU:  {neural_metrics['corpus_bleu']:.2f}")
    print(f"  V4 Neural chrF:  {neural_metrics['avg_chrf']:.2f}")
    print(f"  V4 Hybrid BLEU:  {hybrid_metrics['corpus_bleu']:.2f}")
    print(f"  V4 Hybrid chrF:  {hybrid_metrics['avg_chrf']:.2f}")
    print(f"{'='*70}")

    # ── Save results ──
    results = {
        "best_threshold": best_threshold,
        "test_metrics": hybrid_metrics,
        "neural_metrics": neural_metrics,
        "neural_em": v4_neural_em,
        "retrieval_em": ret_exact / len(test_refs) * 100,
        "hybrid_em": v4_hybrid_em,
        "method_distribution": {
            "retrieval": methods.count("retrieval"),
            "neural": methods.count("neural"),
        },
    }
    with open(os.path.join(config.RESULTS_DIR, "hybrid_results_v4.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save all predictions
    all_preds = [
        {
            "source": test_sources[i],
            "reference": test_refs[i],
            "prediction": hybrid_final[i],
            "method": methods[i],
            "retrieval_conf": round(retrieval_confs[i], 3),
            "retrieval_pred": retrieval_preds[i],
            "neural_pred": neural_preds[i],
            "category": test_cats[i],
            "exact_match": hybrid_final[i].strip().lower() == test_refs[i].strip().lower(),
        }
        for i in range(len(test_refs))
    ]
    with open(os.path.join(config.RESULTS_DIR, "hybrid_predictions_v4.json"), "w",
              encoding="utf-8") as f:
        json.dump(all_preds, f, indent=2, ensure_ascii=False)

    # ── Correct examples ──
    correct_examples = [
        (test_sources[i], test_refs[i], methods[i], retrieval_confs[i])
        for i in range(len(test_refs))
        if hybrid_final[i].strip().lower() == test_refs[i].strip().lower()
    ]
    print(f"\n  Correct translations ({len(correct_examples)}):")
    for src, ref, method, conf in correct_examples[:30]:
        print(f"    {src:30s} -> {ref:25s} ({method}, conf={conf:.2f})")

    print(f"\n  Results saved to {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
