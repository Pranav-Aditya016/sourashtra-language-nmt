"""
Data Pipeline V3 - Augmented Data for T5 Fine-tuning
======================================================
Creates training data with:
  1. Primary: roman_readable → English (12,758 pairs, same as V1/V2)
  2. Augment: harvard_kyoto → English (from unified dataset, train only)
  3. Augment: corpus sentence pairs (~2,334 extra)
  4. Category prefix conditioning

CRITICAL: Test split is identical to V1/V2 (seed=42, 15%)
          so results are directly comparable.
          Augmentation is applied ONLY to training data.
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset


def format_input(source_text, category=None, use_category=True):
    """
    Format source text as T5 input.

    Examples:
      "translate Sourashtra to English: akaraati"
      "translate Sourashtra [Education] to English: akaraati"
    """
    if use_category and category and str(category) != "nan":
        return f"translate Sourashtra [{category}] to English: {source_text.strip()}"
    return f"translate Sourashtra to English: {source_text.strip()}"


def load_and_prepare_data(config):
    """
    Load all data, split identically to V1/V2, augment training set.

    Returns:
        dict with: train_dataset, val_dataset, test_dataset (HF Dataset),
                   test_df, train_df, val_df, split_info
    """
    print("\n[LOAD] Loading data for V3...")

    # ── 1. Load primary translation file (same as V1/V2) ─────────
    trans_df = pd.read_csv(config.TRANSLATION_FILE)
    trans_df["source"] = trans_df["source"].astype(str).str.strip()
    trans_df["target"] = trans_df["target"].astype(str).str.strip()
    trans_df = trans_df[(trans_df["source"].str.len() > 0) &
                        (trans_df["target"].str.len() > 0)].reset_index(drop=True)
    print(f"  Primary translation pairs: {len(trans_df):,}")

    # ── 2. Load unified dataset for categories + HK romanization ──
    unified_df = pd.read_csv(config.UNIFIED_FILE)
    # Create lookup: roman_readable → (category, harvard_kyoto)
    unified_df["roman_readable"] = unified_df["roman_readable"].astype(str).str.strip()
    unified_df["meaning_english"] = unified_df["meaning_english"].astype(str).str.strip()

    # Build category lookup
    cat_lookup = {}
    hk_lookup = {}
    for _, row in unified_df.iterrows():
        rr = row["roman_readable"]
        if pd.isna(rr) or not str(rr).strip():
            continue
        rr = str(rr).strip().lower()
        cat_lookup[rr] = row.get("category", "")
        hk = row.get("havard_kyoto", "")
        if pd.notna(hk) and str(hk).strip():
            hk_lookup[rr] = str(hk).strip()

    # Add categories to trans_df
    trans_df["category"] = trans_df["source"].str.lower().map(cat_lookup).fillna("")
    cat_count = (trans_df["category"] != "").sum()
    print(f"  Pairs with category info: {cat_count:,} / {len(trans_df):,}")

    # ── 3. Split IDENTICALLY to V1/V2 ────────────────────────────
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

    print(f"  Split: Train={len(train_df):,}  Val={len(val_df):,}  Test={len(test_df):,}")

    # ── 4. Build training examples ───────────────────────────────
    train_inputs = []
    train_targets = []
    use_cat = config.USE_CATEGORY_PREFIX

    # 4a. Primary training data (roman_readable → English)
    for _, row in train_df.iterrows():
        inp = format_input(row["source"], row["category"], use_cat)
        train_inputs.append(inp)
        train_targets.append(row["target"].strip())

    primary_count = len(train_inputs)
    print(f"  Primary training examples: {primary_count:,}")

    # 4b. Harvard-Kyoto augmentation (train only)
    hk_added = 0
    if config.AUGMENT_WITH_HK:
        for _, row in train_df.iterrows():
            src_lower = row["source"].lower()
            if src_lower in hk_lookup:
                hk_text = hk_lookup[src_lower]
                # Only add if HK is different from roman_readable
                if hk_text.lower() != row["source"].lower():
                    inp = format_input(hk_text, row["category"], use_cat)
                    train_inputs.append(inp)
                    train_targets.append(row["target"].strip())
                    hk_added += 1
        print(f"  Harvard-Kyoto augmentation: +{hk_added:,} examples")

    # 4c. Corpus sentence pairs (train only)
    sent_added = 0
    if config.AUGMENT_WITH_SENTENCES:
        corpus_df = pd.read_csv(config.CORPUS_FILE)
        # Filter for rows that have both sentence columns
        sent_df = corpus_df.dropna(
            subset=["alias_english_sentence", "english_sentence"]
        )
        sent_df = sent_df[
            (sent_df["alias_english_sentence"].str.strip().str.len() > 0) &
            (sent_df["english_sentence"].str.strip().str.len() > 0)
        ]

        # Get test sources to ensure no data leakage
        test_sources = set(test_df["source"].str.lower().tolist())
        val_sources = set(val_df["source"].str.lower().tolist())

        for _, row in sent_df.iterrows():
            src_sent = str(row["alias_english_sentence"]).strip()
            tgt_sent = str(row["english_sentence"]).strip()
            cat = str(row.get("category", "")).strip()

            # Skip if source contains any test/val word (be conservative)
            src_words = set(src_sent.lower().split())
            # Only skip if an exact source match is in test/val
            alias_eng = str(row.get("alias_english", "")).strip().lower()
            if alias_eng in test_sources or alias_eng in val_sources:
                continue

            inp = format_input(src_sent, cat, use_cat)
            train_inputs.append(inp)
            train_targets.append(tgt_sent)
            sent_added += 1

        print(f"  Sentence augmentation: +{sent_added:,} examples")

    total_train = len(train_inputs)
    print(f"  Total training examples: {total_train:,} "
          f"({total_train/primary_count:.1f}x augmentation)")

    # ── 5. Build validation examples ─────────────────────────────
    val_inputs = [format_input(row["source"], row["category"], use_cat)
                  for _, row in val_df.iterrows()]
    val_targets = [row["target"].strip() for _, row in val_df.iterrows()]

    # ── 6. Build test examples ───────────────────────────────────
    test_inputs = [format_input(row["source"], row["category"], use_cat)
                   for _, row in test_df.iterrows()]
    test_targets = [row["target"].strip() for _, row in test_df.iterrows()]

    # ── 7. Create HuggingFace Datasets ───────────────────────────
    train_dataset = Dataset.from_dict({
        "input_text": train_inputs,
        "target_text": train_targets,
    })
    val_dataset = Dataset.from_dict({
        "input_text": val_inputs,
        "target_text": val_targets,
    })
    test_dataset = Dataset.from_dict({
        "input_text": test_inputs,
        "target_text": test_targets,
    })

    # Shuffle training data
    train_dataset = train_dataset.shuffle(seed=config.RANDOM_SEED)

    # ── 8. Save split info ───────────────────────────────────────
    config.ensure_dirs()
    split_info = {
        "primary_train": primary_count,
        "hk_augmented": hk_added,
        "sentence_augmented": sent_added,
        "total_train": total_train,
        "val_size": len(val_df),
        "test_size": len(test_df),
        "augmentation_ratio": round(total_train / primary_count, 2),
    }
    with open(os.path.join(config.RESULTS_DIR, "split_info_v3.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n[OK] Data ready for T5 fine-tuning!")

    # Show some examples
    print("\n  Sample training inputs:")
    for i in range(min(5, len(train_inputs))):
        print(f"    IN:  {train_inputs[i]}")
        print(f"    OUT: {train_targets[i]}")
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
