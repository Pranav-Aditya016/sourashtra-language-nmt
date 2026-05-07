"""
Sourashtra Translation - V4 Architecture (Multilingual mT5 + Tamil)
====================================================================
WHY V4 exists (V3 got 6.01% exact match with English-only T5):
  - T5-small is English-centric → it tokenizes Tamil as byte sequences
  - Sourashtra and Tamil share phonological roots, grammar, and vocabulary
  - Adding Tamil as a second source language gives the model cross-lingual
    transfer: patterns learned from Tamil↔English help Sourashtra→English
  - Nearly 2x more training data (12.5K Tamil pairs + 12.7K English pairs)

V4 Solution: Fine-tune T5-small (60M params) with multilingual multi-task data
  - Same proven T5-small model as V3 (apples-to-apples comparison)
  - Multi-task training: Sourashtra→English, Tamil→English, Sourashtra→Tamil
  - T5's SentencePiece handles Tamil via byte-fallback tokenization
  - ~3x training data with all three tasks + sentence augmentation
  - Cross-lingual transfer: Tamil signals reinforce Sourashtra patterns

Research hypothesis:
  Tamil and Sourashtra share Dravidian roots. By jointly learning Tamil↔English
  alongside Sourashtra→English, the model creates a shared semantic space that
  bridges all three languages, improving Sourashtra translation accuracy.
  V4 isolates the DATA variable (same model as V3, only adding Tamil).
"""
import os
import torch


class ConfigV4:
    # -- Paths ---------------------------------------------------------
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "cleaned_data")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints_v4")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_v4")
    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs_v4")

    # -- Data Files ----------------------------------------------------
    TRANSLATION_FILE = os.path.join(DATA_DIR, "translation_roman_english.csv")
    UNIFIED_FILE = os.path.join(DATA_DIR, "unified_full_dataset.csv")
    CORPUS_FILE = os.path.join(DATA_DIR, "cleaned_corpus.csv")

    # -- Pre-trained Model ---------------------------------------------
    MODEL_NAME = "t5-small"            # 60M params (same as V3 for fair comparison)
    # For mT5-small (300M, native Tamil): use "google/mt5-small" + BF16 + batch=4
    # mT5 gives better Tamil tokenization but trains ~10x slower on 8GB VRAM

    # -- Multilingual Tasks (V4 core feature) --------------------------
    TASK_SOURASHTRA_TO_ENGLISH = True    # Primary task (same as V3)
    TASK_TAMIL_TO_ENGLISH = True         # NEW: leverage Tamil knowledge
    TASK_SOURASHTRA_TO_TAMIL = False     # DISABLED: generating Tamil with T5
    #   causes FP16 NaN (T5 tokenizes Tamil as byte sequences → overflow)
    #   Tamil→English is safe because Tamil is only in the ENCODER input

    # Task weights for balanced training (higher = more sampling)
    TASK_WEIGHT_SR_EN = 1.0              # Primary task gets full weight
    TASK_WEIGHT_TA_EN = 0.8              # Tamil→English: strong signal
    TASK_WEIGHT_SR_TA = 0.6              # Sourashtra→Tamil: auxiliary

    # -- Data Augmentation ---------------------------------------------
    AUGMENT_WITH_SENTENCES = True        # Use corpus sentence pairs
    AUGMENT_TAMIL_SENTENCES = True       # Also augment Tamil sentence pairs
    USE_CATEGORY_PREFIX = True           # Category conditioning

    # -- Data Split (SAME as V1/V2/V3 for fair comparison) -------------
    TEST_SIZE = 0.15
    VAL_SIZE = 0.10
    RANDOM_SEED = 42

    # -- Training Hyperparameters --------------------------------------
    BATCH_SIZE = 32                      # Same as V3 (T5-small fits easily)
    GRADIENT_ACCUMULATION = 2            # Effective batch = 64
    NUM_EPOCHS = 40                      # Same as V3
    LEARNING_RATE = 1e-3                 # Adafactor default for T5
    WARMUP_RATIO = 0.0                   # No warmup (Adafactor handles it)
    WEIGHT_DECAY = 0.0                   # No regularization
    MAX_SOURCE_LEN = 64                  # Sufficient for words + prefixes
    MAX_TARGET_LEN = 32                  # Most targets are short
    FP16 = True                          # FP16 works fine with T5-small
    BF16 = False                         # Only needed for mT5
    GRADIENT_CHECKPOINTING = False       # Not needed for T5-small
    MAX_GRAD_NORM = 1.0                   # Gradient clipping for stability
    OPTIM = "adafactor"                  # T5 optimizer
    LR_SCHEDULER = "cosine"              # Smooth convergence

    # -- Generation / Decoding -----------------------------------------
    NUM_BEAMS = 5                        # Beam search width
    MAX_GENERATE_LEN = 32                # Most targets are 1-3 words
    NO_REPEAT_NGRAM_SIZE = 3             # Prevent repetition
    REPETITION_PENALTY = 1.5             # Penalize repeated tokens
    LENGTH_PENALTY = 1.0                 # Neutral length preference

    # -- Early Stopping ------------------------------------------------
    PATIENCE = 40                        # Effectively disabled
    EVAL_STEPS_PER_EPOCH = 1             # Evaluate once per epoch
    METRIC_FOR_BEST = "exact_match"      # Track exact match

    # -- Device --------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.CHECKPOINT_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def summary(cls):
        print("\n" + "=" * 70)
        print("  V4 CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"  Model:              {cls.MODEL_NAME}")
        print(f"  Tasks:")
        if cls.TASK_SOURASHTRA_TO_ENGLISH:
            print(f"    Sourashtra → English  (weight={cls.TASK_WEIGHT_SR_EN})")
        if cls.TASK_TAMIL_TO_ENGLISH:
            print(f"    Tamil → English       (weight={cls.TASK_WEIGHT_TA_EN})")
        if cls.TASK_SOURASHTRA_TO_TAMIL:
            print(f"    Sourashtra → Tamil    (weight={cls.TASK_WEIGHT_SR_TA})")
        print(f"  Batch size:         {cls.BATCH_SIZE} × {cls.GRADIENT_ACCUMULATION} = {cls.BATCH_SIZE * cls.GRADIENT_ACCUMULATION}")
        print(f"  Epochs:             {cls.NUM_EPOCHS}")
        print(f"  FP16:               {cls.FP16}")
        print(f"  Grad checkpointing: {cls.GRADIENT_CHECKPOINTING}")
        print(f"  Device:             {cls.DEVICE}")
        print(f"  Category prefix:    {cls.USE_CATEGORY_PREFIX}")
        print(f"  Sentence augment:   {cls.AUGMENT_WITH_SENTENCES}")
        print(f"  Tamil sentences:    {cls.AUGMENT_TAMIL_SENTENCES}")
        print("=" * 70)
