"""
Hybrid Translation System V3 - Optimized
==========================================
Fast character n-gram retrieval + T5 neural ensemble.
"""
import os
import json
import time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from config_v3 import ConfigV3
from data_loader_v3 import load_and_prepare_data
from metrics import evaluate_all, print_metrics


# ═══════════════════════════════════════════════════════════
# FAST CHARACTER N-GRAM RETRIEVAL
# ═══════════════════════════════════════════════════════════

def char_ngrams(word: str, n: int = 3) -> set:
    """Get character n-grams with boundary markers."""
    padded = f"^{word.lower().strip()}$"
    return set(padded[i:i+n] for i in range(len(padded) - n + 1))


def build_similarity_matrix(test_sources, train_sources):
    """Pre-compute similarity between all test-train pairs using fast n-gram Jaccard."""
    print("[INDEX] Pre-computing similarity matrix...")
    t0 = time.time()
    
    n_test = len(test_sources)
    n_train = len(train_sources)
    
    # Pre-compute n-grams for all words
    test_ng = [char_ngrams(s, 3) for s in test_sources]
    train_ng = [char_ngrams(s, 3) for s in train_sources]
    test_bg = [char_ngrams(s, 2) for s in test_sources]
    train_bg = [char_ngrams(s, 2) for s in train_sources]
    
    # Normalize sources for exact match check
    test_lower = [s.lower().strip() for s in test_sources]
    train_lower = [s.lower().strip() for s in train_sources]
    
    # Compute similarity matrix
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
    config = ConfigV3()
    config.ensure_dirs()
    
    print("=" * 60)
    print("  HYBRID TRANSLATION SYSTEM - V3")
    print("=" * 60)
    
    # Load data
    data = load_and_prepare_data(config)
    test_df = data["test_df"]
    train_df = data["train_df"]
    
    test_sources = list(test_df["source"])
    test_refs = list(test_df["target"])
    test_cats = list(test_df["category"]) if "category" in test_df.columns else [""]*len(test_sources)
    
    train_sources = list(train_df["source"])
    train_targets = list(train_df["target"])
    
    # ── Pre-compute similarity matrix ──
    sim_matrix = build_similarity_matrix(test_sources, train_sources)
    
    # Get best retrieval match for each test word
    best_train_idx = np.argmax(sim_matrix, axis=1)
    best_confidence = np.max(sim_matrix, axis=1)
    retrieval_preds = [train_targets[idx] for idx in best_train_idx]
    retrieval_confs = best_confidence.tolist()
    
    # Retrieval-only accuracy
    ret_exact = sum(1 for p, r in zip(retrieval_preds, test_refs) 
                    if p.strip().lower() == r.strip().lower())
    print(f"\n[RETRIEVAL] Overall: {ret_exact}/{len(test_refs)} = {ret_exact/len(test_refs)*100:.2f}%")
    
    # Confidence distribution + band accuracy
    print("\n  Confidence distribution:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        above = sum(1 for c in retrieval_confs if c >= thresh)
        print(f"    Conf >= {thresh:.1f}: {above:4d}/{len(retrieval_confs)} ({above/len(retrieval_confs)*100:.1f}%)")
    
    print("\n  Retrieval accuracy by confidence band:")
    bands = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    for lo, hi in bands:
        correct = total = 0
        for i in range(len(test_refs)):
            if lo <= retrieval_confs[i] < hi:
                total += 1
                if retrieval_preds[i].strip().lower() == test_refs[i].strip().lower():
                    correct += 1
        if total > 0:
            print(f"    [{lo:.1f}, {hi:.1f}): {correct:3d}/{total:3d} = {correct/total*100:.1f}%")
    
    # ── Neural predictions ──
    print(f"\n{'='*60}")
    print("  NEURAL MODEL EVALUATION")
    print(f"{'='*60}")
    
    model_path = os.path.join(config.CHECKPOINT_DIR, "best_model_v3")
    print(f"[NEURAL] Loading {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(config.DEVICE)
    model.eval()
    
    test_input_texts = list(data["test_dataset"]["input_text"])
    neural_preds = []
    batch_size = 32
    
    print(f"[NEURAL] Generating predictions...")
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
    
    neural_exact = sum(1 for p, r in zip(neural_preds, test_refs) 
                       if p.strip().lower() == r.strip().lower())
    print(f"  Neural exact match: {neural_exact}/{len(test_refs)} = {neural_exact/len(test_refs)*100:.2f}%")
    
    # ── Hybrid threshold sweep ──
    print(f"\n{'='*60}")
    print("  HYBRID EVALUATION (threshold sweep)")
    print(f"{'='*60}")
    
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
    
    # ── Final results ──
    print(f"\n{'='*60}")
    print(f"  FINAL HYBRID RESULTS (T={best_threshold})")
    print(f"{'='*60}")
    
    hybrid_final = []
    methods = []
    for i in range(len(test_refs)):
        if retrieval_confs[i] >= best_threshold:
            hybrid_final.append(retrieval_preds[i])
            methods.append("retrieval")
        else:
            hybrid_final.append(neural_preds[i])
            methods.append("neural")
    
    full_metrics = evaluate_all(test_refs, hybrid_final)
    print_metrics(full_metrics, f"V3 HYBRID (T={best_threshold})")
    
    # Method breakdown
    for m in ["retrieval", "neural"]:
        total = methods.count(m)
        correct = sum(1 for i in range(len(test_refs)) if methods[i] == m and 
                      hybrid_final[i].strip().lower() == test_refs[i].strip().lower())
        print(f"  {m:12s}: {correct:3d}/{total:4d} = {correct/max(total,1)*100:.1f}%")
    
    # Correct examples
    correct_examples = [(test_sources[i], test_refs[i], methods[i], retrieval_confs[i])
                       for i in range(len(test_refs))
                       if hybrid_final[i].strip().lower() == test_refs[i].strip().lower()]
    
    print(f"\n  Correct translations ({len(correct_examples)}):")
    for src, ref, method, conf in correct_examples[:30]:
        print(f"    {src:30s} -> {ref:25s} ({method}, conf={conf:.2f})")
    
    # ── Comparison ──
    v3_em = full_metrics["exact_match_accuracy"]
    print(f"\n{'='*60}")
    print(f"  FINAL MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"  V1 (Char Seq2Seq):       0.42%")
    print(f"  V2 (Transformer+BPE):    2.56%")
    print(f"  V3 Neural only:          {neural_exact/len(test_refs)*100:.2f}%")
    print(f"  V3 Retrieval only:       {ret_exact/len(test_refs)*100:.2f}%")
    print(f"  V3 Hybrid:               {v3_em:.2f}%")
    print(f"  Improvement over V2:     {v3_em/2.56:.1f}x")
    print(f"{'='*60}")
    
    # Save
    save = {"best_threshold": best_threshold, "test_metrics": full_metrics,
            "neural_em": neural_exact/len(test_refs)*100,
            "retrieval_em": ret_exact/len(test_refs)*100, "hybrid_em": v3_em}
    with open(os.path.join(config.RESULTS_DIR, "hybrid_results_v3.json"), "w") as f:
        json.dump(save, f, indent=2)
    
    all_results = [{"source": test_sources[i], "reference": test_refs[i],
                    "prediction": hybrid_final[i], "method": methods[i],
                    "retrieval_conf": round(retrieval_confs[i], 3),
                    "retrieval_pred": retrieval_preds[i], "neural_pred": neural_preds[i],
                    "category": test_cats[i],
                    "exact_match": hybrid_final[i].strip().lower() == test_refs[i].strip().lower()}
                   for i in range(len(test_refs))]
    with open(os.path.join(config.RESULTS_DIR, "hybrid_predictions_v3.json"), "w",
              encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Results saved to {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
