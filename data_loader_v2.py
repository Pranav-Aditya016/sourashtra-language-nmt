"""
BPE Tokenizer + Data Pipeline for V2 Transformer Model
========================================================
Uses SentencePiece for BPE subword tokenization.

Why BPE for Sourashtra:
  - Creates reusable subword units from Sourashtra morphology
  - "pushTi kEr" and "pushTi" share the "pushTi" subword
  - Reduces vocabulary from 11K unique words to ~1K subwords
  - Handles unseen words through subword decomposition
"""
import os
import json
import tempfile
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sentencepiece as spm


# =========================================================
# BPE Tokenizer Wrapper
# =========================================================

class BPETokenizer:
    """SentencePiece BPE tokenizer with special token handling."""

    PAD_ID = 0
    SOS_ID = 1  # <s> in sentencepiece
    EOS_ID = 2  # </s> in sentencepiece
    UNK_ID = 3

    def __init__(self, model_path=None):
        self.sp = spm.SentencePieceProcessor()
        if model_path and os.path.exists(model_path):
            self.sp.Load(model_path)

    @classmethod
    def train(cls, texts, vocab_size, model_prefix, model_type="bpe",
              character_coverage=1.0):
        """Train a new BPE tokenizer on the given texts."""
        # Write texts to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                         delete=False, encoding="utf-8") as f:
            for text in texts:
                f.write(text.lower().strip() + "\n")
            temp_path = f.name

        try:
            spm.SentencePieceTrainer.Train(
                input=temp_path,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
                pad_id=cls.PAD_ID,
                bos_id=cls.SOS_ID,
                eos_id=cls.EOS_ID,
                unk_id=cls.UNK_ID,
                pad_piece="<PAD>",
                bos_piece="<SOS>",
                eos_piece="<EOS>",
                unk_piece="<UNK>",
                # Normalization settings for preserving special chars
                normalization_rule_name="identity",
                remove_extra_whitespaces=False,
                # Training params
                num_threads=4,
                train_extremely_large_corpus=False,
            )
        finally:
            os.unlink(temp_path)

        tokenizer = cls(model_prefix + ".model")
        print(f"  Trained BPE tokenizer: vocab_size={tokenizer.vocab_size()}")
        return tokenizer

    def vocab_size(self):
        return self.sp.GetPieceSize()

    def encode(self, text, max_len=64, add_sos=True, add_eos=True):
        """Encode text to token IDs with optional SOS/EOS."""
        ids = self.sp.Encode(text.lower().strip())

        if add_sos:
            ids = [self.SOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]

        actual_len = len(ids)

        # Pad or truncate
        if len(ids) < max_len:
            ids += [self.PAD_ID] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
            if add_eos:
                ids[-1] = self.EOS_ID
            actual_len = max_len

        return ids, actual_len

    def decode(self, ids):
        """Decode token IDs back to text."""
        # Filter out special tokens
        clean_ids = []
        for id_ in ids:
            if isinstance(id_, torch.Tensor):
                id_ = id_.item()
            if id_ in (self.PAD_ID, self.SOS_ID, self.EOS_ID):
                if id_ == self.EOS_ID:
                    break
                continue
            clean_ids.append(id_)
        return self.sp.Decode(clean_ids)

    def tokenize(self, text):
        """Tokenize to subword pieces (for visualization)."""
        return self.sp.EncodeAsPieces(text.lower().strip())


# =========================================================
# Dataset
# =========================================================

class TranslationDatasetV2(Dataset):
    """BPE-tokenized translation dataset."""

    def __init__(self, sources, targets, src_tokenizer, tgt_tokenizer, max_len):
        self.sources = sources
        self.targets = targets
        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        src_text = self.sources[idx]
        tgt_text = self.targets[idx]

        src_ids, src_len = self.src_tok.encode(src_text, self.max_len,
                                                add_sos=False, add_eos=True)
        tgt_ids, tgt_len = self.tgt_tok.encode(tgt_text, self.max_len,
                                                add_sos=True, add_eos=True)

        return {
            "source": torch.tensor(src_ids, dtype=torch.long),
            "target": torch.tensor(tgt_ids, dtype=torch.long),
            "source_len": src_len,
            "target_len": tgt_len,
            "source_text": src_text,
            "target_text": tgt_text,
        }


# =========================================================
# Full Data Pipeline
# =========================================================

def load_data_v2(config):
    """
    Load data, train BPE tokenizers, create splits and loaders.

    Returns dict with: train_loader, val_loader, test_loader,
                       src_tokenizer, tgt_tokenizer, train_df, val_df, test_df
    """
    print("\n[LOAD] Loading data...")
    df = pd.read_csv(config.TRANSLATION_FILE)
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df = df[(df["source"].str.len() > 0) & (df["target"].str.len() > 0)].reset_index(drop=True)
    print(f"  Total pairs: {len(df):,}")

    # -- Train/Val/Test Split ----------------------------------------
    train_val_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=config.VAL_SIZE / (1 - config.TEST_SIZE),
        random_state=config.RANDOM_SEED
    )
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # -- Train BPE Tokenizers (on training data only) ----------------
    print("\n[INFO] Training BPE tokenizers...")
    config.ensure_dirs()

    src_model_prefix = os.path.join(config.TOKENIZER_DIR, "src_bpe")
    tgt_model_prefix = os.path.join(config.TOKENIZER_DIR, "tgt_bpe")

    print("  Source (Roman Sourashtra):")
    src_tokenizer = BPETokenizer.train(
        train_df["source"].tolist(),
        vocab_size=config.SRC_VOCAB_SIZE,
        model_prefix=src_model_prefix,
        model_type=config.BPE_MODEL_TYPE,
        character_coverage=config.BPE_CHARACTER_COVERAGE,
    )

    print("  Target (English):")
    tgt_tokenizer = BPETokenizer.train(
        train_df["target"].tolist(),
        vocab_size=config.TGT_VOCAB_SIZE,
        model_prefix=tgt_model_prefix,
        model_type=config.BPE_MODEL_TYPE,
        character_coverage=config.BPE_CHARACTER_COVERAGE,
    )

    # Show tokenization examples (replace SentencePiece U+2581 for Windows cp1252)
    print("\n  BPE tokenization examples:")
    for s, t in zip(train_df["source"].head(5), train_df["target"].head(5)):
        src_pieces = src_tokenizer.tokenize(s)
        tgt_pieces = tgt_tokenizer.tokenize(t)
        src_safe = [p.replace('\u2581', '_') for p in src_pieces]
        tgt_safe = [p.replace('\u2581', '_') for p in tgt_pieces]
        print(f"    '{s}' -> {src_safe}")
        print(f"    '{t}' -> {tgt_safe}")
        print()

    # -- Create Datasets & Loaders -----------------------------------
    train_dataset = TranslationDatasetV2(
        train_df["source"].tolist(), train_df["target"].tolist(),
        src_tokenizer, tgt_tokenizer, config.MAX_SEQ_LEN
    )
    val_dataset = TranslationDatasetV2(
        val_df["source"].tolist(), val_df["target"].tolist(),
        src_tokenizer, tgt_tokenizer, config.MAX_SEQ_LEN
    )
    test_dataset = TranslationDatasetV2(
        test_df["source"].tolist(), test_df["target"].tolist(),
        src_tokenizer, tgt_tokenizer, config.MAX_SEQ_LEN
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=pin
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=pin
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=pin
    )

    # -- Save split info ---------------------------------------------
    split_info = {
        "total_pairs": len(df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "src_vocab_size": src_tokenizer.vocab_size(),
        "tgt_vocab_size": tgt_tokenizer.vocab_size(),
    }
    with open(os.path.join(config.RESULTS_DIR, "split_info_v2.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"[OK] Data ready! Src vocab: {src_tokenizer.vocab_size()}, "
          f"Tgt vocab: {tgt_tokenizer.vocab_size()}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "src_tokenizer": src_tokenizer,
        "tgt_tokenizer": tgt_tokenizer,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }
