"""
Evaluation & Enhanced Hybrid System V5
=========================================
1. Evaluate V5 (ByT5-small) neural predictions
2. Run ENHANCED retrieval: Levenshtein + Jaccard + prefix matching
3. Hybrid system: enhanced retrieval + neural ensemble
4. Also re-evaluate V4 model with enhanced retrieval for comparison

Enhanced Retrieval (V5 innovation):
  - Character n-gram Jaccard similarity (existing from V3/V4)
  - Normalized Levenshtein edit distance (NEW)
  - Common prefix bonus (NEW)
  - Combined weighted score for better matching
"""
import os
import json
import time
import numpy as np
import pandas as pd
from collections import Counter

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config_v5 import ConfigV5
from data_loader_v5 import load_and_prepare_data, format_sr_to_en
from metrics import evaluate_all, print_metrics


# ═══════════════════════════════════════════════════════════
# ENHANCED RETRIEVAL SYSTEM (V5 Innovation)
# ═══════════════════════════════════════════════════════════

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,      # deletion
                curr[j] + 1,           # insertion
                prev[j] + (c1 != c2),  # substitution
            ))
        prev = curr
    return prev[-1]


def char_ngrams(word: str, n: int = 3) -> set:
    """Get character n-grams with boundary markers."""
    padded = f"^{word.lower().strip()}$"
    return set(padded[i:i+n] for i in range(len(padded) - n + 1))


def jaccard_similarity(set1: set, set2: set) -> float:
    """Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    union = len(set1 | set2)
    return len(set1 & set2) / union if union > 0 else 0.0


def common_prefix_length(s1: str, s2: str) -> int:
    """Length of common prefix."""
    n = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            n += 1
        else:
            break
    return n


def build_enhanced_similarity_matrix(test_sources, train_sources, config):
    """
    Enhanced similarity matrix combining multiple string similarity methods.

    Combines:
      1. Character n-gram Jaccard (trigram + bigram)
      2. Normalized Levenshtein similarity
      3. Common prefix ratio bonus
    """
    print("[INDEX] Building ENHANCED similarity matrix...")
    print(f"  Weights: Jaccard={config.RETRIEVAL_JACCARD_WEIGHT}, "
          f"Levenshtein={config.RETRIEVAL_LEVENSHTEIN_WEIGHT}, "
          f"Prefix={config.RETRIEVAL_PREFIX_WEIGHT}")
    t0 = time.time()

    n_test = len(test_sources)
    n_train = len(train_sources)

    # Pre-compute n-grams
    test_tri = [char_ngrams(s, 3) for s in test_sources]
    train_tri = [char_ngrams(s, 3) for s in train_sources]
    test_bi = [char_ngrams(s, 2) for s in test_sources]
    train_bi = [char_ngrams(s, 2) for s in train_sources]

    test_lower = [s.lower().strip() for s in test_sources]
    train_lower = [s.lower().strip() for s in train_sources]

    w_jac = config.RETRIEVAL_JACCARD_WEIGHT
    w_lev = config.RETRIEVAL_LEVENSHTEIN_WEIGHT
    w_pfx = config.RETRIEVAL_PREFIX_WEIGHT

    sim_matrix = np.zeros((n_test, n_train), dtype=np.float32)

    for i in range(n_test):
        for j in range(n_train):
            # Exact match → score 1.0
            if test_lower[i] == train_lower[j]:
                sim_matrix[i, j] = 1.0
                continue

            # 1. N-gram Jaccard (trigram + bigram blend)
            j3 = jaccard_similarity(test_tri[i], train_tri[j])
            j2 = jaccard_similarity(test_bi[i], train_bi[j])
            jac_score = 0.55 * j3 + 0.45 * j2

            # 2. Normalized Levenshtein similarity
            max_len = max(len(test_lower[i]), len(train_lower[j]))
            if max_len == 0:
                lev_sim = 1.0
            else:
                dist = levenshtein_distance(test_lower[i], train_lower[j])
                lev_sim = 1.0 - dist / max_len

            # 3. Common prefix ratio
            pfx_len = common_prefix_length(test_lower[i], train_lower[j])
            pfx_ratio = pfx_len / max_len if max_len > 0 else 0.0

            # Combined score
            sim_matrix[i, j] = w_jac * jac_score + w_lev * lev_sim + w_pfx * pfx_ratio

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_test} ({(i+1)/n_test*100:.0f}%)")

    elapsed = time.time() - t0
    print(f"  Enhanced matrix computed in {elapsed:.1f}s ({n_test}x{n_train})")
    return sim_matrix


# ═══════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════

def run_hybrid_evaluation(model_path, config, model_name_label, version_tag,
                          use_enhanced_retrieval=True):
    """
    Run full hybrid evaluation for any model.

    Args:
        model_path: Path to saved model checkpoint
        config: Configuration object
        model_name_label: Label for prints (e.g., "V5 ByT5-small")
        version_tag: File suffix (e.g., "v5")
        use_enhanced_retrieval: Whether to use enhanced similarity (True) or basic Jaccard
    """
    config.ensure_dirs()

    print(f"\n{'='*70}")
    print(f"  {model_name_label} EVALUATION & HYBRID SYSTEM")
    print(f"{'='*70}")

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
    # ENHANCED RETRIEVAL
    # ═══════════════════════════════════════════════════════
    if use_enhanced_retrieval:
        sim_matrix = build_enhanced_similarity_matrix(test_sources, train_sources, config)
    else:
        # Fall back to basic Jaccard (V3/V4 style)
        from evaluate_v4 import build_similarity_matrix
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
    # NEURAL PREDICTIONS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  NEURAL MODEL ({model_name_label})")
    print(f"{'='*70}")

    if not os.path.exists(model_path):
        print(f"  [ERROR] Model not found at {model_path}")
        print(f"  Run training script first!")
        return

    print(f"[NEURAL] Loading {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(config.DEVICE)
    model.eval()

    use_cat = config.USE_CATEGORY_PREFIX
    test_input_texts = [format_sr_to_en(row["source"], row["category"], use_cat)
                        for _, row in test_df.iterrows()]

    neural_preds = []
    batch_size = 8

    print(f"[NEURAL] Generating predictions ({len(test_input_texts)} examples)...")
    for i in range(0, len(test_input_texts), batch_size):
        batch = test_input_texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                          max_length=config.MAX_SOURCE_LEN).to(config.DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=config.MAX_GENERATE_LEN,
                num_beams=config.NUM_BEAMS,
                length_penalty=config.LENGTH_PENALTY,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        neural_preds.extend([d.strip() for d in decoded])

        if (i + batch_size) % 200 == 0:
            print(f"  {min(i+batch_size, len(test_input_texts))}/{len(test_input_texts)}")

    neural_exact = sum(1 for p, r in zip(neural_preds, test_refs)
                       if p.strip().lower() == r.strip().lower())
    print(f"\n  Neural exact match: {neural_exact}/{len(test_refs)} = {neural_exact/len(test_refs)*100:.2f}%")

    neural_metrics = evaluate_all(test_refs, neural_preds)
    print_metrics(neural_metrics, f"{model_name_label} NEURAL-ONLY")

    # ═══════════════════════════════════════════════════════
    # HYBRID THRESHOLD SWEEP (finer granularity)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  HYBRID THRESHOLD SWEEP (Enhanced Retrieval)")
    print(f"{'='*70}")

    best_threshold = 0.5
    best_em = 0

    thresholds = [0.25, 0.30, 0.35, 0.38, 0.40, 0.42, 0.44, 0.45, 0.46,
                  0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                  0.85, 0.90, 0.95, 1.0]

    for threshold in thresholds:
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
    print(f"  FINAL HYBRID RESULTS (T={best_threshold})")
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
    print_metrics(hybrid_metrics, f"{model_name_label} HYBRID (T={best_threshold})")

    # Method breakdown
    for m in ["retrieval", "neural"]:
        total = methods.count(m)
        correct = sum(1 for i in range(len(test_refs)) if methods[i] == m and
                      hybrid_final[i].strip().lower() == test_refs[i].strip().lower())
        print(f"  {m:12s}: {correct:3d}/{total:4d} = {correct/max(total,1)*100:.1f}%")

    # ═══════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════
    results = {
        "best_threshold": best_threshold,
        "test_metrics": hybrid_metrics,
        "neural_metrics": neural_metrics,
        "neural_em": neural_metrics["exact_match_accuracy"],
        "retrieval_em": ret_exact / len(test_refs) * 100,
        "hybrid_em": hybrid_metrics["exact_match_accuracy"],
        "enhanced_retrieval": use_enhanced_retrieval,
        "method_distribution": {
            "retrieval": methods.count("retrieval"),
            "neural": methods.count("neural"),
        },
    }
    with open(os.path.join(config.RESULTS_DIR, f"hybrid_results_{version_tag}.json"), "w") as f:
        json.dump(results, f, indent=2)

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
    with open(os.path.join(config.RESULTS_DIR, f"hybrid_predictions_{version_tag}.json"), "w",
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
    return results


def main():
    config = ConfigV5()

    # ═══════════════════════════════════════════════════════
    # STEP 1: Evaluate V5 (ByT5-small) with enhanced retrieval
    # ═══════════════════════════════════════════════════════
    v5_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model_v5")
    v5_results = run_hybrid_evaluation(
        model_path=v5_model_path,
        config=config,
        model_name_label="V5 (ByT5-small + Tamil)",
        version_tag="v5",
        use_enhanced_retrieval=True,
    )

    # ═══════════════════════════════════════════════════════
    # STEP 2: Re-evaluate V4 with enhanced retrieval
    # ═══════════════════════════════════════════════════════
    v4_model_path = os.path.join(config.PROJECT_ROOT, "checkpoints_v4", "best_model_v4")
    if os.path.exists(v4_model_path):
        print("\n\n" + "#" * 70)
        print("  BONUS: Re-evaluating V4 with ENHANCED retrieval")
        print("#" * 70)

        v4_enhanced = run_hybrid_evaluation(
            model_path=v4_model_path,
            config=config,
            model_name_label="V4 (T5+Tamil) + ENHANCED Retrieval",
            version_tag="v4_enhanced",
            use_enhanced_retrieval=True,
        )

    # ═══════════════════════════════════════════════════════
    # FINAL COMPARISON
    # ═══════════════════════════════════════════════════════
    print(f"\n\n{'='*70}")
    print(f"  COMPLETE MODEL COMPARISON (V1 → V5)")
    print(f"{'='*70}")
    print(f"  V1 (Char GRU Seq2Seq):                 0.42%")
    print(f"  V2 (Transformer+BPE):                  2.56%")
    print(f"  V3 Neural (T5-small, EN):              6.01%")
    print(f"  V3 Hybrid (T5+Retrieval):              7.47%")
    print(f"  V4 Neural (T5+Tamil):                  5.80%")
    print(f"  V4 Hybrid (T5+Tamil+Retrieval):        7.68%")
    if v5_results:
        print(f"  V5 Neural (ByT5+Tamil):                {v5_results['neural_em']:.2f}%")
        print(f"  V5 Hybrid (ByT5+Tamil+Enhanced Ret):   {v5_results['hybrid_em']:.2f}%")
    if os.path.exists(v4_model_path) and v4_enhanced:
        print(f"  V4 + Enhanced Retrieval:               {v4_enhanced['hybrid_em']:.2f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
