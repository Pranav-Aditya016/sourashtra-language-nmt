"""
Data Pipeline V5-Reverse — English/Tamil → Sourashtra (Roman)
==============================================================
Same data as V5 but with source/target SWAPPED:
  Original V5:  input="translate Sourashtra to English: paal"   target="milk"
  Reverse V5:   input="translate English to Sourashtra: milk"    target="paal"

Tasks:
  1. English → Sourashtra (Roman)   (primary)
  2. Tamil → Sourashtra (Roman)     (cross-lingual)
  + Sentence augmentation

CRITICAL: Test split is identical to V1–V5 (seed=42, 15%)
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset


# ──────────────────────────────────────────────────────────────
# Task Formatting (REVERSED from V5)
# ──────────────────────────────────────────────────────────────

def format_en_to_sr(english_text, category=None, use_category=True):
    """Format: English → Sourashtra task."""
    if use_category and category and str(category) != "nan":
        return f"translate English [{category}] to Sourashtra: {english_text.strip()}"
    return f"translate English to Sourashtra: {english_text.strip()}"


def format_ta_to_sr(tamil_text, category=None, use_category=True):
    """Format: Tamil → Sourashtra task."""
    if use_category and category and str(category) != "nan":
        return f"translate Tamil [{category}] to Sourashtra: {tamil_text.strip()}"
    return f"translate Tamil to Sourashtra: {tamil_text.strip()}"


# ──────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────

def load_and_prepare_data(config):
    """Load data with source/target swapped for reverse translation."""
    print("\n" + "=" * 70)
    print("  [V5-REVERSE] LOADING DATA (English/Tamil → Sourashtra)")
    print("=" * 70)

    # ── 1. Load primary translation file ──
    # Original: source=Sourashtra, target=English
    # We SWAP: source=English, target=Sourashtra
    trans_df = pd.read_csv(config.TRANSLATION_FILE)
    trans_df["source"] = trans_df["source"].astype(str).str.strip()   # Sourashtra Roman
    trans_df["target"] = trans_df["target"].astype(str).str.strip()   # English
    trans_df = trans_df[(trans_df["source"].str.len() > 0) &
                        (trans_df["target"].str.len() > 0)].reset_index(drop=True)
    print(f"  Parallel pairs (English ↔ Sourashtra): {len(trans_df):,}")

    # ── 2. Load unified dataset for Tamil + categories ──
    unified_df = pd.read_csv(config.UNIFIED_FILE)
    unified_df["roman_readable"] = unified_df["roman_readable"].astype(str).str.strip()
    unified_df["meaning_english"] = unified_df["meaning_english"].astype(str).str.strip()
    unified_df["meaning_tamil"] = unified_df["meaning_tamil"].astype(str).str.strip()

    cat_lookup = {}
    tamil_lookup = {}
    for _, row in unified_df.iterrows():
        rr = row["roman_readable"]
        if pd.isna(rr) or not str(rr).strip():
            continue
        rr = str(rr).strip().lower()
        cat_lookup[rr] = row.get("category", "")
        tamil = row.get("meaning_tamil", "")
        if pd.notna(tamil) and str(tamil).strip():
            tamil_lookup[rr] = str(tamil).strip()

    trans_df["category"] = trans_df["source"].str.lower().map(cat_lookup).fillna("")
    trans_df["tamil_meaning"] = trans_df["source"].str.lower().map(tamil_lookup).fillna("")

    cat_count = (trans_df["category"] != "").sum()
    tamil_count = (trans_df["tamil_meaning"] != "").sum()
    print(f"  With category info: {cat_count:,} / {len(trans_df):,}")
    print(f"  With Tamil meaning: {tamil_count:,} / {len(trans_df):,}")

    # ── 3. Split IDENTICALLY to V1/V2/V3/V4/V5 ──
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

    # ── 4. Build MULTI-TASK training examples (REVERSED) ──
    train_inputs, train_targets, train_tasks = [], [], []
    use_cat = config.USE_CATEGORY_PREFIX
    count_en_sr = count_ta_sr = count_sent = count_ta_sent = 0

    # Task 1: English → Sourashtra (REVERSED from V5)
    if config.TASK_ENGLISH_TO_SOURASHTRA:
        for _, row in train_df.iterrows():
            # Input = English word, Target = Sourashtra Roman
            train_inputs.append(format_en_to_sr(row["target"], row["category"], use_cat))
            train_targets.append(row["source"].strip())  # Sourashtra Roman
            train_tasks.append("en_sr")
            count_en_sr += 1
        print(f"\n  Task 1 [English→Sourashtra]: {count_en_sr:,} examples")

    # Task 2: Tamil → Sourashtra
    if config.TASK_TAMIL_TO_SOURASHTRA:
        for _, row in train_df.iterrows():
            tamil = row["tamil_meaning"]
            if tamil and str(tamil) != "nan" and str(tamil).strip():
                train_inputs.append(format_ta_to_sr(tamil, row["category"], use_cat))
                train_targets.append(row["source"].strip())  # Sourashtra Roman
                train_tasks.append("ta_sr")
                count_ta_sr += 1
        print(f"  Task 2 [Tamil→Sourashtra]:   {count_ta_sr:,} examples")

    # Sentence augmentation (English → Sourashtra)
    if config.AUGMENT_WITH_SENTENCES:
        corpus_df = pd.read_csv(config.CORPUS_FILE)
        sent_df = corpus_df.dropna(subset=["alias_english_sentence", "english_sentence"])
        sent_df = sent_df[
            (sent_df["alias_english_sentence"].str.strip().str.len() > 0) &
            (sent_df["english_sentence"].str.strip().str.len() > 0)
        ]
        test_sources_set = set(test_df["source"].str.lower())
        val_sources_set = set(val_df["source"].str.lower())

        for _, row in sent_df.iterrows():
            src_sent = str(row["alias_english_sentence"]).strip()  # Sourashtra romanized sentence
            tgt_sent = str(row["english_sentence"]).strip()        # English sentence
            cat = str(row.get("category", "")).strip()
            alias = str(row.get("alias_english", "")).strip().lower()
            if alias in test_sources_set or alias in val_sources_set:
                continue
            # REVERSED: English sentence → Sourashtra sentence
            train_inputs.append(format_en_to_sr(tgt_sent, cat, use_cat))
            train_targets.append(src_sent)
            train_tasks.append("en_sr_sent")
            count_sent += 1
        print(f"\n  Sentence augmentation (EN→SR): +{count_sent:,} examples")

    # Tamil sentence augmentation (Tamil → Sourashtra)
    if config.AUGMENT_TAMIL_SENTENCES:
        u_sent = unified_df[
            unified_df["example_sentence_tamil"].notna() &
            unified_df["roman_readable"].notna()
        ]
        if len(u_sent) > 0:
            for _, row in u_sent.iterrows():
                rr = str(row.get("roman_readable", "")).strip().lower()
                if rr in test_sources_set or rr in val_sources_set:
                    continue
                ta_sent = str(row["example_sentence_tamil"]).strip()
                sr_word = str(row["roman_readable"]).strip()
                cat = str(row.get("category", "")).strip()
                if ta_sent and sr_word:
                    train_inputs.append(format_ta_to_sr(ta_sent, cat, use_cat))
                    train_targets.append(sr_word)
                    train_tasks.append("ta_sr_sent")
                    count_ta_sent += 1
        print(f"  Sentence augment (TA→SR):      +{count_ta_sent:,} examples")

    total_train = len(train_inputs)
    print(f"\n  {'=' * 50}")
    print(f"  TOTAL TRAINING EXAMPLES: {total_train:,}")
    print(f"    English→Sourashtra (words):  {count_en_sr:,}")
    print(f"    Tamil→Sourashtra (words):    {count_ta_sr:,}")
    print(f"    Sentence augment (EN→SR):    {count_sent:,}")
    print(f"    Sentence augment (TA→SR):    {count_ta_sent:,}")
    print(f"  Augmentation ratio: {total_train / max(count_en_sr, 1):.2f}x")
    print(f"  {'=' * 50}")

    # ── Validation / Test (English→Sourashtra ONLY) ──
    val_inputs = [format_en_to_sr(row["target"], row["category"], use_cat)
                  for _, row in val_df.iterrows()]
    val_targets = [row["source"].strip() for _, row in val_df.iterrows()]
    test_inputs = [format_en_to_sr(row["target"], row["category"], use_cat)
                   for _, row in test_df.iterrows()]
    test_targets = [row["source"].strip() for _, row in test_df.iterrows()]

    # ── HuggingFace Datasets ──
    train_dataset = Dataset.from_dict({
        "input_text": train_inputs, "target_text": train_targets, "task": train_tasks,
    }).shuffle(seed=config.RANDOM_SEED)
    val_dataset = Dataset.from_dict({"input_text": val_inputs, "target_text": val_targets})
    test_dataset = Dataset.from_dict({"input_text": test_inputs, "target_text": test_targets})

    # ── Save split info ──
    config.ensure_dirs()
    split_info = {
        "task_en_sr": count_en_sr, "task_ta_sr": count_ta_sr,
        "sent_augment_en_sr": count_sent, "sent_augment_ta_sr": count_ta_sent,
        "total_train": total_train,
        "val_size": len(val_df), "test_size": len(test_df),
        "augmentation_ratio": round(total_train / max(count_en_sr, 1), 2),
        "languages": ["english", "tamil", "sourashtra"],
        "direction": "English/Tamil → Sourashtra",
        "model": "byt5-small",
    }
    with open(os.path.join(config.RESULTS_DIR, "split_info_v5_reverse.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n[OK] Reverse data ready for ByT5 fine-tuning!")

    return {
        "train_dataset": train_dataset, "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "split_info": split_info,
    }
