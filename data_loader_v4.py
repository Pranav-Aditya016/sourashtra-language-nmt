"""
Data Pipeline V4 - Multilingual Multi-Task for mT5 Fine-tuning
================================================================
Creates training data with three translation tasks:
  1. Sourashtra (Roman) → English  (12,758 pairs — primary, same as V3)
  2. Tamil → English               (12,497 pairs — NEW cross-lingual)
  3. Sourashtra (Roman) → Tamil     (12,497 pairs — NEW bridge task)
  + Sentence augmentation in all languages

CRITICAL: Test split is identical to V1/V2/V3 (seed=42, 15%)
          Test evaluation is Sourashtra → English ONLY for fair comparison.
          Tamil tasks are used as TRAINING augmentation only.
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset


# ──────────────────────────────────────────────────────────────
# Task Formatting
# ──────────────────────────────────────────────────────────────

def format_sr_to_en(source_text, category=None, use_category=True):
    """Format: Sourashtra → English task (primary)."""
    if use_category and category and str(category) != "nan":
        return f"translate Sourashtra [{category}] to English: {source_text.strip()}"
    return f"translate Sourashtra to English: {source_text.strip()}"


def format_ta_to_en(tamil_text, category=None, use_category=True):
    """Format: Tamil → English task (cross-lingual)."""
    if use_category and category and str(category) != "nan":
        return f"translate Tamil [{category}] to English: {tamil_text.strip()}"
    return f"translate Tamil to English: {tamil_text.strip()}"


def format_sr_to_ta(source_text, category=None, use_category=True):
    """Format: Sourashtra → Tamil task (bridge)."""
    if use_category and category and str(category) != "nan":
        return f"translate Sourashtra [{category}] to Tamil: {source_text.strip()}"
    return f"translate Sourashtra to Tamil: {source_text.strip()}"


# ──────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────

def load_and_prepare_data(config):
    """
    Load multilingual data, split identically to V1/V2/V3, build multi-task sets.

    Returns:
        dict with: train_dataset, val_dataset, test_dataset (HF Dataset),
                   test_df, train_df, val_df, split_info
    """
    print("\n" + "=" * 70)
    print("  [V4] LOADING MULTILINGUAL DATA")
    print("=" * 70)

    # ── 1. Load primary translation file (Roman Sourashtra → English) ─
    trans_df = pd.read_csv(config.TRANSLATION_FILE)
    trans_df["source"] = trans_df["source"].astype(str).str.strip()
    trans_df["target"] = trans_df["target"].astype(str).str.strip()
    trans_df = trans_df[(trans_df["source"].str.len() > 0) &
                        (trans_df["target"].str.len() > 0)].reset_index(drop=True)
    print(f"  Primary (Sourashtra → English) pairs: {len(trans_df):,}")

    # ── 2. Load unified dataset for Tamil + categories ─────────────
    unified_df = pd.read_csv(config.UNIFIED_FILE)
    unified_df["roman_readable"] = unified_df["roman_readable"].astype(str).str.strip()
    unified_df["meaning_english"] = unified_df["meaning_english"].astype(str).str.strip()
    unified_df["meaning_tamil"] = unified_df["meaning_tamil"].astype(str).str.strip()

    # Build lookups: roman_readable → (category, tamil_meaning)
    cat_lookup = {}
    tamil_lookup = {}  # roman_readable → tamil meaning
    for _, row in unified_df.iterrows():
        rr = row["roman_readable"]
        if pd.isna(rr) or not str(rr).strip():
            continue
        rr = str(rr).strip().lower()
        cat_lookup[rr] = row.get("category", "")
        tamil = row.get("meaning_tamil", "")
        if pd.notna(tamil) and str(tamil).strip():
            tamil_lookup[rr] = str(tamil).strip()

    # Add categories and Tamil meanings to trans_df
    trans_df["category"] = trans_df["source"].str.lower().map(cat_lookup).fillna("")
    trans_df["tamil_meaning"] = trans_df["source"].str.lower().map(tamil_lookup).fillna("")

    cat_count = (trans_df["category"] != "").sum()
    tamil_count = (trans_df["tamil_meaning"] != "").sum()
    print(f"  With category info: {cat_count:,} / {len(trans_df):,}")
    print(f"  With Tamil meaning: {tamil_count:,} / {len(trans_df):,}")

    # ── 3. Split IDENTICALLY to V1/V2/V3 ──────────────────────────
    train_val_df, test_df = train_test_split(
        trans_df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=config.VAL_SIZE / (1 - config.TEST_SIZE),
        random_state=config.RANDOM_SEED,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"\n  Split: Train={len(train_df):,}  Val={len(val_df):,}  Test={len(test_df):,}")

    # ── 4. Build MULTI-TASK training examples ──────────────────────
    train_inputs = []
    train_targets = []
    train_tasks = []     # Track which task each example belongs to
    use_cat = config.USE_CATEGORY_PREFIX

    # Count per task
    count_sr_en = 0
    count_ta_en = 0
    count_sr_ta = 0
    count_sent = 0
    count_ta_sent = 0

    # ── 4a. Task 1: Sourashtra → English (primary) ────────────────
    if config.TASK_SOURASHTRA_TO_ENGLISH:
        for _, row in train_df.iterrows():
            inp = format_sr_to_en(row["source"], row["category"], use_cat)
            train_inputs.append(inp)
            train_targets.append(row["target"].strip())
            train_tasks.append("sr_en")
            count_sr_en += 1
        print(f"\n  Task 1 [Sourashtra→English]: {count_sr_en:,} examples")

    # ── 4b. Task 2: Tamil → English (cross-lingual transfer) ──────
    if config.TASK_TAMIL_TO_ENGLISH:
        for _, row in train_df.iterrows():
            tamil = row["tamil_meaning"]
            if tamil and str(tamil) != "nan" and str(tamil).strip():
                inp = format_ta_to_en(tamil, row["category"], use_cat)
                train_inputs.append(inp)
                train_targets.append(row["target"].strip())
                train_tasks.append("ta_en")
                count_ta_en += 1
        print(f"  Task 2 [Tamil→English]:      {count_ta_en:,} examples")

    # ── 4c. Task 3: Sourashtra → Tamil (bridge task) ──────────────
    if config.TASK_SOURASHTRA_TO_TAMIL:
        for _, row in train_df.iterrows():
            tamil = row["tamil_meaning"]
            if tamil and str(tamil) != "nan" and str(tamil).strip():
                inp = format_sr_to_ta(row["source"], row["category"], use_cat)
                train_inputs.append(inp)
                train_targets.append(str(tamil).strip())
                train_tasks.append("sr_ta")
                count_sr_ta += 1
        print(f"  Task 3 [Sourashtra→Tamil]:   {count_sr_ta:,} examples")

    # ── 4d. Sentence augmentation (English) ───────────────────────
    if config.AUGMENT_WITH_SENTENCES:
        corpus_df = pd.read_csv(config.CORPUS_FILE)
        sent_df = corpus_df.dropna(
            subset=["alias_english_sentence", "english_sentence"]
        )
        sent_df = sent_df[
            (sent_df["alias_english_sentence"].str.strip().str.len() > 0) &
            (sent_df["english_sentence"].str.strip().str.len() > 0)
        ]

        # Data leakage protection
        test_sources = set(test_df["source"].str.lower().tolist())
        val_sources = set(val_df["source"].str.lower().tolist())

        for _, row in sent_df.iterrows():
            src_sent = str(row["alias_english_sentence"]).strip()
            tgt_sent = str(row["english_sentence"]).strip()
            cat = str(row.get("category", "")).strip()
            alias_eng = str(row.get("alias_english", "")).strip().lower()

            if alias_eng in test_sources or alias_eng in val_sources:
                continue

            inp = format_sr_to_en(src_sent, cat, use_cat)
            train_inputs.append(inp)
            train_targets.append(tgt_sent)
            train_tasks.append("sr_en_sent")
            count_sent += 1

        print(f"\n  Sentence augmentation (EN):  +{count_sent:,} examples")

    # ── 4e. Tamil sentence augmentation ───────────────────────────
    if config.AUGMENT_TAMIL_SENTENCES:
        corpus_df = pd.read_csv(config.CORPUS_FILE)
        # Check if Tamil sentences exist in corpus
        ta_cols = ["tamil_sentence", "example_sentence_tamil"]
        ta_sent_col = None
        for col in ta_cols:
            if col in corpus_df.columns:
                ta_sent_col = col
                break

        if ta_sent_col is None:
            # Try from unified dataset
            u_sent = unified_df[
                unified_df["example_sentence_tamil"].notna() &
                unified_df["example_sentence_english"].notna()
            ]
            if len(u_sent) > 0:
                test_sources = set(test_df["source"].str.lower().tolist())
                val_sources = set(val_df["source"].str.lower().tolist())

                for _, row in u_sent.iterrows():
                    rr = str(row.get("roman_readable", "")).strip().lower()
                    if rr in test_sources or rr in val_sources:
                        continue
                    ta_sent = str(row["example_sentence_tamil"]).strip()
                    en_sent = str(row["example_sentence_english"]).strip()
                    cat = str(row.get("category", "")).strip()
                    if ta_sent and en_sent:
                        inp = format_ta_to_en(ta_sent, cat, use_cat)
                        train_inputs.append(inp)
                        train_targets.append(en_sent)
                        train_tasks.append("ta_en_sent")
                        count_ta_sent += 1

            print(f"  Sentence augment (Tamil):    +{count_ta_sent:,} examples")

    # ── Summary ────────────────────────────────────────────────────
    total_train = len(train_inputs)
    print(f"\n  {'=' * 50}")
    print(f"  TOTAL TRAINING EXAMPLES: {total_train:,}")
    print(f"    Sourashtra→English (words):     {count_sr_en:,}")
    print(f"    Tamil→English (words):          {count_ta_en:,}")
    print(f"    Sourashtra→Tamil (words):       {count_sr_ta:,}")
    print(f"    Sentence augment (EN):          {count_sent:,}")
    print(f"    Sentence augment (Tamil):       {count_ta_sent:,}")
    print(f"  Augmentation ratio: {total_train / count_sr_en:.2f}x")
    print(f"  {'=' * 50}")

    # ── 5. Build validation examples (Sourashtra→English ONLY) ─────
    #    Validation uses only the primary task for fair comparison
    val_inputs = [format_sr_to_en(row["source"], row["category"], use_cat)
                  for _, row in val_df.iterrows()]
    val_targets = [row["target"].strip() for _, row in val_df.iterrows()]

    # ── 6. Build test examples (Sourashtra→English ONLY) ──────────
    test_inputs = [format_sr_to_en(row["source"], row["category"], use_cat)
                   for _, row in test_df.iterrows()]
    test_targets = [row["target"].strip() for _, row in test_df.iterrows()]

    # ── 7. Create HuggingFace Datasets ─────────────────────────────
    train_dataset = Dataset.from_dict({
        "input_text": train_inputs,
        "target_text": train_targets,
        "task": train_tasks,
    })
    val_dataset = Dataset.from_dict({
        "input_text": val_inputs,
        "target_text": val_targets,
    })
    test_dataset = Dataset.from_dict({
        "input_text": test_inputs,
        "target_text": test_targets,
    })

    # Shuffle training data (mixes tasks together)
    train_dataset = train_dataset.shuffle(seed=config.RANDOM_SEED)

    # ── 8. Save split info ─────────────────────────────────────────
    config.ensure_dirs()
    split_info = {
        "task_sr_en": count_sr_en,
        "task_ta_en": count_ta_en,
        "task_sr_ta": count_sr_ta,
        "sent_augment_en": count_sent,
        "sent_augment_ta": count_ta_sent,
        "total_train": total_train,
        "val_size": len(val_df),
        "test_size": len(test_df),
        "augmentation_ratio": round(total_train / max(count_sr_en, 1), 2),
        "languages": ["sourashtra", "english", "tamil"],
    }
    with open(os.path.join(config.RESULTS_DIR, "split_info_v4.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n[OK] Multilingual data ready for mT5 fine-tuning!")

    # Show task examples
    print("\n  Sample training inputs (multi-task):")
    task_examples = {}
    for i in range(len(train_inputs)):
        t = train_tasks[i]
        if t not in task_examples:
            task_examples[t] = (train_inputs[i], train_targets[i])
    for task, (inp, tgt) in task_examples.items():
        print(f"    [{task}]")
        print(f"    IN:  {inp}")
        print(f"    OUT: {tgt}")
        print()

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "split_info": split_info,
    }
