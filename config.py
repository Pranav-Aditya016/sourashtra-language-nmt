"""
Configuration for Sourashtra Translation Model
================================================
Centralized config for all experiments.
"""
import torch
import os

class Config:
    # ── Project Paths ──────────────────────────────────────────────
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "cleaned_data")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

    # ── Data Files ─────────────────────────────────────────────────
    TRANSLATION_FILE = os.path.join(DATA_DIR, "translation_roman_english.csv")
    UNIFIED_FILE = os.path.join(DATA_DIR, "unified_full_dataset.csv")
    EXAMPLE_SENTENCES_FILE = os.path.join(DATA_DIR, "example_sentences.csv")

    # ── Data Split ─────────────────────────────────────────────────
    TEST_SIZE = 0.15
    VAL_SIZE = 0.10       # 10% of train for validation
    RANDOM_SEED = 42

    # ── Character-level Seq2Seq Model ──────────────────────────────
    # Using character-level because:
    # - Source has 11K unique word tokens (too sparse for 12K samples)
    # - Most entries are 1-2 words (char-level captures morphology)
    # - Better generalization for unseen Sourashtra words
    EMBEDDING_DIM = 128
    ENCODER_HIDDEN_DIM = 256
    DECODER_HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    ATTENTION_TYPE = "bahdanau"  # "bahdanau" or "luong"

    # ── Training ───────────────────────────────────────────────────
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 1.0
    TEACHER_FORCING_START = 1.0   # Start with full teacher forcing
    TEACHER_FORCING_END = 0.3     # Anneal down to 30%
    TEACHER_FORCING_DECAY = 0.97  # Multiply each epoch
    PATIENCE = 15                 # Early stopping patience
    SCHEDULER_PATIENCE = 5       # LR scheduler patience
    SCHEDULER_FACTOR = 0.5       # LR reduction factor

    # ── Sequence Lengths ───────────────────────────────────────────
    MAX_SOURCE_LEN = 80    # Max chars in source (roman_readable)
    MAX_TARGET_LEN = 120   # Max chars in target (english meaning)

    # ── Device ─────────────────────────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Logging ────────────────────────────────────────────────────
    LOG_INTERVAL = 50       # Log every N batches
    EVAL_SAMPLES = 10       # Number of samples to show during eval
    SAVE_EVERY = 5          # Save checkpoint every N epochs

    @classmethod
    def ensure_dirs(cls):
        """Create all necessary directories."""
        for d in [cls.CHECKPOINT_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def summary(cls):
        """Print configuration summary."""
        print("=" * 60)
        print("SOURASHTRA TRANSLATION MODEL - CONFIGURATION")
        print("=" * 60)
        print(f"  Device:           {cls.DEVICE}")
        print(f"  Data file:        {cls.TRANSLATION_FILE}")
        print(f"  Embedding dim:    {cls.EMBEDDING_DIM}")
        print(f"  Hidden dim:       {cls.ENCODER_HIDDEN_DIM}")
        print(f"  Num layers:       {cls.NUM_LAYERS}")
        print(f"  Dropout:          {cls.DROPOUT}")
        print(f"  Batch size:       {cls.BATCH_SIZE}")
        print(f"  Epochs:           {cls.NUM_EPOCHS}")
        print(f"  Learning rate:    {cls.LEARNING_RATE}")
        print(f"  Max source len:   {cls.MAX_SOURCE_LEN}")
        print(f"  Max target len:   {cls.MAX_TARGET_LEN}")
        print(f"  Teacher forcing:  {cls.TEACHER_FORCING_START} → {cls.TEACHER_FORCING_END}")
        print(f"  Early stopping:   patience={cls.PATIENCE}")
        print("=" * 60)
