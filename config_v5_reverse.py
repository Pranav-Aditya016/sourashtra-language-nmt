"""
Sourashtra Translation - V5 Reverse (English/Tamil → Sourashtra Roman)
=======================================================================
Same architecture as V5 (ByT5-small) but with REVERSED translation direction.

V5 Original: Sourashtra → English  (best accuracy: 9.25% EM)
V5 Reverse:  English/Tamil → Sourashtra (Roman script)

Uses the same parallel data (12,758 pairs) with source/target swapped.
"""
import os
import torch


class ConfigV5Reverse:
    # -- Paths ---------------------------------------------------------
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "cleaned_data")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints_v5_reverse")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_v5_reverse")
    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs_v5_reverse")

    # -- Data Files ----------------------------------------------------
    TRANSLATION_FILE = os.path.join(DATA_DIR, "translation_roman_english.csv")
    UNIFIED_FILE = os.path.join(DATA_DIR, "unified_full_dataset.csv")
    CORPUS_FILE = os.path.join(DATA_DIR, "cleaned_corpus.csv")

    # -- Pre-trained Model ---------------------------------------------
    MODEL_NAME = "google/byt5-small"   # 300M params, byte-level T5

    # -- Multilingual Tasks (REVERSED direction) -----------------------
    TASK_ENGLISH_TO_SOURASHTRA = True
    TASK_TAMIL_TO_SOURASHTRA = True

    TASK_WEIGHT_EN_SR = 1.0
    TASK_WEIGHT_TA_SR = 0.8

    # -- Data Augmentation ---------------------------------------------
    AUGMENT_WITH_SENTENCES = True
    AUGMENT_TAMIL_SENTENCES = True
    USE_CATEGORY_PREFIX = True

    # -- Data Split (SAME as V1/V2/V3/V4/V5 for fair comparison) -------
    TEST_SIZE = 0.15
    VAL_SIZE = 0.10
    RANDOM_SEED = 42

    # -- Training Hyperparameters (same as V5) -------------------------
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION = 4
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
    WARMUP_RATIO = 0.06
    WEIGHT_DECAY = 0.0
    MAX_SOURCE_LEN = 128     # English/Tamil input in bytes
    MAX_TARGET_LEN = 128     # Sourashtra Roman output (can be longer than English)
    FP16 = False
    BF16 = True
    GRADIENT_CHECKPOINTING = True
    MAX_GRAD_NORM = 1.0
    OPTIM = "adafactor"
    LR_SCHEDULER = "cosine"

    # -- Generation / Decoding -----------------------------------------
    NUM_BEAMS = 4
    MAX_GENERATE_LEN = 128   # Sourashtra romanized can be longer
    NO_REPEAT_NGRAM_SIZE = 0
    REPETITION_PENALTY = 1.0
    LENGTH_PENALTY = 1.0

    # -- Early Stopping ------------------------------------------------
    PATIENCE = 20
    EVAL_STEPS_PER_EPOCH = 1
    METRIC_FOR_BEST = "exact_match"

    # -- Device --------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.CHECKPOINT_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def summary(cls):
        print("\n" + "=" * 70)
        print("  V5-REVERSE CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"  Model:              {cls.MODEL_NAME}")
        print(f"  Architecture:       ByT5-small (byte-level, ~300M params)")
        print(f"  Direction:          English/Tamil → Sourashtra (REVERSE)")
        print(f"  Tasks:")
        if cls.TASK_ENGLISH_TO_SOURASHTRA:
            print(f"    English → Sourashtra  (weight={cls.TASK_WEIGHT_EN_SR})")
        if cls.TASK_TAMIL_TO_SOURASHTRA:
            print(f"    Tamil → Sourashtra    (weight={cls.TASK_WEIGHT_TA_SR})")
        print(f"  Batch size:         {cls.BATCH_SIZE} × {cls.GRADIENT_ACCUMULATION} = {cls.BATCH_SIZE * cls.GRADIENT_ACCUMULATION}")
        print(f"  Epochs:             {cls.NUM_EPOCHS}")
        print(f"  BF16:               {cls.BF16}")
        print(f"  Grad checkpointing: {cls.GRADIENT_CHECKPOINTING}")
        print(f"  Max source len:     {cls.MAX_SOURCE_LEN} bytes")
        print(f"  Max target len:     {cls.MAX_TARGET_LEN} bytes")
        print(f"  Device:             {cls.DEVICE}")
        print("=" * 70)
