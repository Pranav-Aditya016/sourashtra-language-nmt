"""
Sourashtra Translation - V3 Architecture (Pre-trained T5 Fine-tuning)
======================================================================
WHY V3 exists (V2 got only 2.56% exact match):
  - V1/V2 train from scratch — model has NO prior knowledge of English
  - 92% of source words appear only once (hapax) → can't generalize
  - Sourashtra and English have zero morphological correspondence
  - 12K pairs is far too little to learn a full translation model

V3 Solution: Fine-tune T5-small (60M params)
  - T5 already knows English perfectly (trained on 750GB of text)
  - It only needs to learn: Roman Sourashtra input → English meaning
  - Data augmentation: use Harvard-Kyoto romanization + corpus sentences
  - Category-conditioned inputs help the model disambiguate

Expected improvement: 20-50% exact match (vs 2.56% in V2)
"""
import os
import torch


class ConfigV3:
    # -- Paths ---------------------------------------------------------
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "cleaned_data")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints_v3")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_v3")
    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs_v3")

    # -- Data Files ----------------------------------------------------
    TRANSLATION_FILE = os.path.join(DATA_DIR, "translation_roman_english.csv")
    UNIFIED_FILE = os.path.join(DATA_DIR, "unified_full_dataset.csv")
    CORPUS_FILE = os.path.join(DATA_DIR, "cleaned_corpus.csv")

    # -- Pre-trained Model ---------------------------------------------
    MODEL_NAME = "t5-small"          # 60M params, fits easily in 8GB VRAM
    # MODEL_NAME = "google/flan-t5-small"  # Alternative: instruction-tuned

    # -- Data Augmentation ---------------------------------------------
    AUGMENT_WITH_HK = False          # Disabled: different romanization adds noise
    AUGMENT_WITH_SENTENCES = True    # Use corpus sentence pairs (~2.3K extra)
    USE_CATEGORY_PREFIX = True       # Add category info to input

    # -- Data Split (SAME as V1/V2 for fair comparison) ----------------
    TEST_SIZE = 0.15
    VAL_SIZE = 0.10
    RANDOM_SEED = 42

    # -- Training Hyperparameters --------------------------------------
    BATCH_SIZE = 32                  # T5-small fits 32 on 8GB GPU
    GRADIENT_ACCUMULATION = 2        # Effective batch = 64
    NUM_EPOCHS = 40                  # Peak was ~30-43 epochs
    LEARNING_RATE = 1e-3             # T5 default with Adafactor
    WARMUP_RATIO = 0.0               # No warmup for Adafactor
    WEIGHT_DECAY = 0.0               # No regularization — we want memorization
    MAX_SOURCE_LEN = 64              # Max tokens for source
    MAX_TARGET_LEN = 32              # Most targets are short
    FP16 = True                      # Mixed precision for speed + memory
    OPTIM = "adafactor"              # Optimizer T5 was designed for
    LR_SCHEDULER = "cosine"          # Cosine decay for stable convergence

    # -- Generation / Decoding -----------------------------------------
    NUM_BEAMS = 5                    # Beam search width
    MAX_GENERATE_LEN = 32            # Most targets are 1-3 words
    NO_REPEAT_NGRAM_SIZE = 3         # Prevent repetition loops
    REPETITION_PENALTY = 1.5         # Penalize repeated tokens
    LENGTH_PENALTY = 1.0             # Neutral length preference

    # -- Early Stopping ------------------------------------------------
    PATIENCE = 40                    # Effectively disabled — let it overfit
    EVAL_STEPS_PER_EPOCH = 1         # Evaluate once per epoch (faster)
    METRIC_FOR_BEST = "exact_match"  # Track exact match for model selection

    # -- Device --------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.CHECKPOINT_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def summary(cls):
        print("=" * 60)
        print("SOURASHTRA TRANSLATION V3 - T5 Fine-tuning")
        print("=" * 60)
        print(f"  Device:             {cls.DEVICE}")
        print(f"  Model:              {cls.MODEL_NAME}")
        print(f"  HK Augmentation:    {cls.AUGMENT_WITH_HK}")
        print(f"  Sentence Augment:   {cls.AUGMENT_WITH_SENTENCES}")
        print(f"  Category Prefix:    {cls.USE_CATEGORY_PREFIX}")
        print(f"  Batch size:         {cls.BATCH_SIZE} x {cls.GRADIENT_ACCUMULATION} accum")
        print(f"  Epochs:             {cls.NUM_EPOCHS}")
        print(f"  Learning rate:      {cls.LEARNING_RATE}")
        print(f"  FP16:               {cls.FP16}")
        print(f"  Beam search:        {cls.NUM_BEAMS}")
        print(f"  Early stopping:     patience={cls.PATIENCE}")
        print("=" * 60)
