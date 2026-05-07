"""
Sourashtra Translation - V5 Architecture (ByT5-small + Tamil + Enhanced Retrieval)
====================================================================================
WHY V5 exists (V4 Hybrid got 7.68% exact match):
  - T5-small tokenizes romanized Sourashtra into subwords (e.g. "paal" → ["pa","al"])
    This LOSES character-level patterns critical for transliteration
  - ByT5-small operates at the RAW BYTE level — every character is a separate token
  - For transliteration tasks, byte-level processing captures exact character
    correspondences (e.g. "aa"→long vowel, "th"→aspirate) that subword models miss
  - Same parameter count as mT5-small (~300M) but 4x fewer decoder layers
    → faster generation, less memory for decoding

V5 Solution: ByT5-small + Tamil multi-task + enhanced retrieval
  - ByT5-small: byte-level Transformer (no tokenizer vocabulary)
  - Same Tamil cross-lingual strategy as V4 (2x training data)
  - Enhanced retrieval: Levenshtein distance + Jaccard + combined scoring
  - Gradient checkpointing to fit 300M params in 8GB VRAM
"""
import os
import torch


class ConfigV5:
    # -- Paths ---------------------------------------------------------
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "cleaned_data")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints_v5")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_v5")
    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs_v5")

    # -- Data Files ----------------------------------------------------
    TRANSLATION_FILE = os.path.join(DATA_DIR, "translation_roman_english.csv")
    UNIFIED_FILE = os.path.join(DATA_DIR, "unified_full_dataset.csv")
    CORPUS_FILE = os.path.join(DATA_DIR, "cleaned_corpus.csv")

    # -- Pre-trained Model ---------------------------------------------
    MODEL_NAME = "google/byt5-small"   # 300M params, byte-level T5
    # ByT5 architecture: 12 encoder layers, 4 decoder layers
    # d_model = 1472, no SentencePiece — pure UTF-8 byte encoding

    # -- Multilingual Tasks (same as V4) -------------------------------
    TASK_SOURASHTRA_TO_ENGLISH = True
    TASK_TAMIL_TO_ENGLISH = True
    TASK_SOURASHTRA_TO_TAMIL = False   # Still disabled (FP16 Tamil gen issue)

    TASK_WEIGHT_SR_EN = 1.0
    TASK_WEIGHT_TA_EN = 0.8
    TASK_WEIGHT_SR_TA = 0.0

    # -- Data Augmentation ---------------------------------------------
    AUGMENT_WITH_SENTENCES = True
    AUGMENT_TAMIL_SENTENCES = True
    USE_CATEGORY_PREFIX = True

    # -- Data Split (SAME as V1/V2/V3/V4) ------------------------------
    TEST_SIZE = 0.15
    VAL_SIZE = 0.10
    RANDOM_SEED = 42

    # -- Training Hyperparameters (tuned for ByT5 on 8GB) --------------
    BATCH_SIZE = 16                      # Smaller: ByT5 is 5x T5-small
    GRADIENT_ACCUMULATION = 4            # Effective batch = 64
    NUM_EPOCHS = 20                      # ByT5 converges faster
    LEARNING_RATE = 1e-3                 # Adafactor default
    WARMUP_RATIO = 0.06                  # Brief warmup for stability
    WEIGHT_DECAY = 0.0
    MAX_SOURCE_LEN = 128                 # Bytes are longer than subwords
    MAX_TARGET_LEN = 64                  # English targets in bytes
    FP16 = False
    BF16 = True                          # BF16 required for ByT5 (FP16 causes NaN with d_model=1472)
    GRADIENT_CHECKPOINTING = True        # REQUIRED for 300M params in 8GB
    MAX_GRAD_NORM = 1.0
    OPTIM = "adafactor"
    LR_SCHEDULER = "cosine"

    # -- Generation / Decoding -----------------------------------------
    NUM_BEAMS = 4                        # Slightly fewer for speed
    MAX_GENERATE_LEN = 64               # Byte-level needs more tokens
    NO_REPEAT_NGRAM_SIZE = 0             # Disabled for byte-level
    REPETITION_PENALTY = 1.0             # Neutral for byte-level
    LENGTH_PENALTY = 1.0

    # -- Early Stopping ------------------------------------------------
    PATIENCE = 20
    EVAL_STEPS_PER_EPOCH = 1
    METRIC_FOR_BEST = "exact_match"

    # -- Enhanced Retrieval (V5 innovation) ----------------------------
    RETRIEVAL_JACCARD_WEIGHT = 0.45      # Weight for n-gram Jaccard
    RETRIEVAL_LEVENSHTEIN_WEIGHT = 0.45  # Weight for edit distance sim
    RETRIEVAL_PREFIX_WEIGHT = 0.10       # Weight for prefix match bonus

    # -- Device --------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.CHECKPOINT_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def summary(cls):
        print("\n" + "=" * 70)
        print("  V5 CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"  Model:              {cls.MODEL_NAME}")
        print(f"  Architecture:       ByT5-small (byte-level, ~300M params)")
        print(f"  Tasks:")
        if cls.TASK_SOURASHTRA_TO_ENGLISH:
            print(f"    Sourashtra → English  (weight={cls.TASK_WEIGHT_SR_EN})")
        if cls.TASK_TAMIL_TO_ENGLISH:
            print(f"    Tamil → English       (weight={cls.TASK_WEIGHT_TA_EN})")
        print(f"  Batch size:         {cls.BATCH_SIZE} × {cls.GRADIENT_ACCUMULATION} = {cls.BATCH_SIZE * cls.GRADIENT_ACCUMULATION}")
        print(f"  Epochs:             {cls.NUM_EPOCHS}")
        print(f"  FP16:               {cls.FP16}")
        print(f"  Grad checkpointing: {cls.GRADIENT_CHECKPOINTING}")
        print(f"  Max source len:     {cls.MAX_SOURCE_LEN} bytes")
        print(f"  Max target len:     {cls.MAX_TARGET_LEN} bytes")
        print(f"  Device:             {cls.DEVICE}")
        print(f"  Enhanced retrieval: Jaccard({cls.RETRIEVAL_JACCARD_WEIGHT}) + "
              f"Levenshtein({cls.RETRIEVAL_LEVENSHTEIN_WEIGHT}) + "
              f"Prefix({cls.RETRIEVAL_PREFIX_WEIGHT})")
        print("=" * 70)
