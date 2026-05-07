"""
Data loading and preprocessing for Sourashtra Translation
==========================================================
Character-level tokenization with proper train/val/test splits.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import json
import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Character Vocabulary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CharVocab:
    """Character-level vocabulary for Sourashtra↔English translation."""

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        self.char2idx = {
            self.PAD_TOKEN: 0,
            self.SOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.char_freq = Counter()

    @property
    def pad_idx(self):
        return self.char2idx[self.PAD_TOKEN]

    @property
    def sos_idx(self):
        return self.char2idx[self.SOS_TOKEN]

    @property
    def eos_idx(self):
        return self.char2idx[self.EOS_TOKEN]

    @property
    def unk_idx(self):
        return self.char2idx[self.UNK_TOKEN]

    def __len__(self):
        return len(self.char2idx)

    def build(self, texts):
        """Build vocabulary from list of strings."""
        for text in texts:
            self.char_freq.update(text.lower())

        idx = len(self.char2idx)
        for char in sorted(self.char_freq.keys()):
            if char not in self.char2idx:
                self.char2idx[char] = idx
                self.idx2char[idx] = char
                idx += 1

        print(f"  Vocab size: {len(self)} ({len(self.char_freq)} unique chars)")

    def encode(self, text, max_len, add_sos=False, add_eos=True):
        """
        Encode text → list of indices.
        Returns: (indices_tensor, actual_length)
        """
        chars = list(text.lower())
        indices = []
        if add_sos:
            indices.append(self.sos_idx)
        indices += [self.char2idx.get(c, self.unk_idx) for c in chars]
        if add_eos:
            indices.append(self.eos_idx)

        actual_len = len(indices)

        # Pad or truncate
        if len(indices) < max_len:
            indices += [self.pad_idx] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
            actual_len = max_len

        return indices, actual_len

    def decode(self, indices):
        """Decode list of indices → text string."""
        chars = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            if idx == self.eos_idx:
                break
            if idx not in (self.pad_idx, self.sos_idx):
                chars.append(self.idx2char.get(idx, "?"))
        return "".join(chars)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TranslationDataset(Dataset):
    """Character-level translation dataset."""

    def __init__(self, sources, targets, src_vocab, tgt_vocab,
                 max_src_len, max_tgt_len):
        self.sources = sources
        self.targets = targets
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        src = self.sources[idx]
        tgt = self.targets[idx]

        # Encode source (no SOS, add EOS)
        src_indices, src_len = self.src_vocab.encode(
            src, self.max_src_len, add_sos=False, add_eos=True
        )
        # Encode target (add SOS and EOS for teacher forcing)
        tgt_indices, tgt_len = self.tgt_vocab.encode(
            tgt, self.max_tgt_len, add_sos=True, add_eos=True
        )

        return {
            "source": torch.tensor(src_indices, dtype=torch.long),
            "target": torch.tensor(tgt_indices, dtype=torch.long),
            "source_len": src_len,
            "target_len": tgt_len,
            "source_text": src,
            "target_text": tgt,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Loading Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_data(config):
    """
    Load and prepare all data splits.

    Returns:
        dict with keys: train_loader, val_loader, test_loader,
                        src_vocab, tgt_vocab, test_df
    """
    print("\n[LOAD] Loading data...")
    df = pd.read_csv(config.TRANSLATION_FILE)
    print(f"  Total pairs: {len(df):,}")

    # Clean: strip whitespace, drop any empty
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df = df[(df["source"].str.len() > 0) & (df["target"].str.len() > 0)].reset_index(drop=True)
    print(f"  After cleaning: {len(df):,}")

    # ── Split: Train (75%) / Val (10%) / Test (15%) ────────────
    train_val_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=config.VAL_SIZE / (1 - config.TEST_SIZE),
        random_state=config.RANDOM_SEED
    )

    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # ── Build Vocabularies (on train only to prevent leaking) ──
    print("\n[INFO] Building vocabularies...")
    src_vocab = CharVocab()
    tgt_vocab = CharVocab()

    print("  Source (Roman Sourashtra):")
    src_vocab.build(train_df["source"].tolist())
    print("  Target (English):")
    tgt_vocab.build(train_df["target"].tolist())

    # ── Create Datasets ────────────────────────────────────────
    train_dataset = TranslationDataset(
        train_df["source"].tolist(), train_df["target"].tolist(),
        src_vocab, tgt_vocab, config.MAX_SOURCE_LEN, config.MAX_TARGET_LEN
    )
    val_dataset = TranslationDataset(
        val_df["source"].tolist(), val_df["target"].tolist(),
        src_vocab, tgt_vocab, config.MAX_SOURCE_LEN, config.MAX_TARGET_LEN
    )
    test_dataset = TranslationDataset(
        test_df["source"].tolist(), test_df["target"].tolist(),
        src_vocab, tgt_vocab, config.MAX_SOURCE_LEN, config.MAX_TARGET_LEN
    )

    # ── Create DataLoaders ─────────────────────────────────────
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=pin, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=pin
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=pin
    )

    # ── Save split info for reproducibility ────────────────────
    split_info = {
        "total_pairs": len(df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "random_seed": config.RANDOM_SEED,
    }
    config.ensure_dirs()
    with open(os.path.join(config.RESULTS_DIR, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n[OK] Data ready! Src vocab: {len(src_vocab)}, Tgt vocab: {len(tgt_vocab)}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


def collate_fn(batch):
    """Custom collate that sorts by source length (for packed sequences)."""
    batch.sort(key=lambda x: x["source_len"], reverse=True)
    return {
        "source": torch.stack([x["source"] for x in batch]),
        "target": torch.stack([x["target"] for x in batch]),
        "source_len": [x["source_len"] for x in batch],
        "target_len": [x["target_len"] for x in batch],
        "source_text": [x["source_text"] for x in batch],
        "target_text": [x["target_text"] for x in batch],
    }
