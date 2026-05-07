"""
Sourashtra Translation - V2 Architecture
==========================================
Fixes from V1 (char-level Seq2Seq - 0% BLEU):
  - V1 failed because char-level has no cross-lingual correspondence
    ("gaay" chars don't map to "cow" chars)
  - V2 uses BPE subword tokenization (captures Sourashtra morphology)
  - V2 uses Transformer architecture (better for low-resource MT)
  - V2 uses label smoothing + proper warmup scheduling

Architecture: Small Transformer Encoder-Decoder
  - BPE subword tokenization (shared or separate)
  - Positional encoding
  - Multi-head self-attention + cross-attention
  - Label smoothing for better generalization

Why this works better:
  1. BPE tokenization creates ~500-2000 subword units that REUSE
     across words (e.g., "gaay" and "gaaygo'run" share "gaay")
  2. Transformer attention sees global context simultaneously
  3. Label smoothing prevents overconfident memorization
  4. Warmup schedule stabilizes early training
"""
import os
import torch

class ConfigV2:
    # -- Paths ---------------------------------------------------------
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "cleaned_data")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints_v2")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_v2")
    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs_v2")
    TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "tokenizers")

    # -- Data ----------------------------------------------------------
    TRANSLATION_FILE = os.path.join(DATA_DIR, "translation_roman_english.csv")
    UNIFIED_FILE = os.path.join(DATA_DIR, "unified_full_dataset.csv")

    # -- Data Split ----------------------------------------------------
    TEST_SIZE = 0.15
    VAL_SIZE = 0.10
    RANDOM_SEED = 42

    # -- BPE Tokenizer -------------------------------------------------
    SRC_VOCAB_SIZE = 1000    # Sourashtra subwords
    TGT_VOCAB_SIZE = 2000    # English subwords (more variability)
    BPE_CHARACTER_COVERAGE = 1.0
    BPE_MODEL_TYPE = "bpe"   # "bpe" or "unigram"

    # -- Transformer Model ---------------------------------------------
    D_MODEL = 256            # Model dimension
    N_HEADS = 8              # Attention heads
    N_ENCODER_LAYERS = 3     # Encoder layers
    N_DECODER_LAYERS = 3     # Decoder layers
    D_FF = 512               # Feed-forward dimension
    DROPOUT = 0.3            # Dropout rate (higher for small data)
    MAX_SEQ_LEN = 64         # Max sequence length in subword tokens

    # -- Training ------------------------------------------------------
    BATCH_SIZE = 128
    NUM_EPOCHS = 150
    LEARNING_RATE = 5e-4     # Peak LR (after warmup)
    WARMUP_STEPS = 500       # Linear warmup steps
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    LABEL_SMOOTHING = 0.1    # Prevents overconfident predictions
    PATIENCE = 20            # Early stopping patience

    # -- Device --------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Logging -------------------------------------------------------
    EVAL_EVERY = 5           # Evaluate every N epochs
    SAVE_EVERY = 10          # Save checkpoint every N epochs

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.CHECKPOINT_DIR, cls.RESULTS_DIR, cls.LOGS_DIR, cls.TOKENIZER_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def summary(cls):
        print("=" * 60)
        print("SOURASHTRA TRANSLATION MODEL V2 - CONFIGURATION")
        print("=" * 60)
        print(f"  Device:             {cls.DEVICE}")
        print(f"  Architecture:       Transformer ({cls.N_ENCODER_LAYERS}E/{cls.N_DECODER_LAYERS}D)")
        print(f"  d_model:            {cls.D_MODEL}")
        print(f"  Attention heads:    {cls.N_HEADS}")
        print(f"  Feed-forward dim:   {cls.D_FF}")
        print(f"  Dropout:            {cls.DROPOUT}")
        print(f"  Src BPE vocab:      {cls.SRC_VOCAB_SIZE}")
        print(f"  Tgt BPE vocab:      {cls.TGT_VOCAB_SIZE}")
        print(f"  Batch size:         {cls.BATCH_SIZE}")
        print(f"  Epochs:             {cls.NUM_EPOCHS}")
        print(f"  Learning rate:      {cls.LEARNING_RATE}")
        print(f"  Warmup steps:       {cls.WARMUP_STEPS}")
        print(f"  Label smoothing:    {cls.LABEL_SMOOTHING}")
        print(f"  Early stopping:     patience={cls.PATIENCE}")
        print("=" * 60)
