"""
Transformer Seq2Seq Model for Sourashtra Translation (V2)
==========================================================
Small Transformer encoder-decoder designed for low-resource MT.

Architecture choices for 12K-pair dictionary-style dataset:
  - Small model (3 layers, d_model=256) to prevent overfitting
  - High dropout (0.3) for regularization
  - Label smoothing in loss function
  - Sinusoidal positional encoding (no learned positions needed)

The Transformer handles this task better than GRU because:
  1. Self-attention captures relationships between subwords
  2. Cross-attention aligns source and target subwords
  3. Parallel processing during training (faster on GPU)
  4. Better gradient flow for longer sequences
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Positional Encoding
# =========================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# =========================================================
# Transformer Model
# =========================================================

class TransformerTranslator(nn.Module):
    """
    Transformer Encoder-Decoder for translation.

    Uses PyTorch's built-in nn.Transformer with custom
    embedding layers and output projection.
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256,
                 n_heads=8, n_encoder_layers=3, n_decoder_layers=3,
                 d_ff=512, dropout=0.3, max_seq_len=64, pad_idx=0):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,  # Using batch_first=True
            norm_first=True,   # Pre-LayerNorm (more stable training)
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_src_key_padding_mask(self, src):
        """Create source padding mask. True = masked position."""
        return src == self.pad_idx  # (batch, src_len)

    def _make_tgt_key_padding_mask(self, tgt):
        """Create target padding mask."""
        return tgt == self.pad_idx  # (batch, tgt_len)

    def _make_tgt_causal_mask(self, tgt_len, device):
        """Create causal (look-ahead) mask for decoder."""
        return nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)

    def forward(self, src, tgt):
        """
        Forward pass.

        Args:
            src: (batch, src_len) source token IDs
            tgt: (batch, tgt_len) target token IDs (shifted right, starts with SOS)
        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        # Masks
        src_key_padding_mask = self._make_src_key_padding_mask(src)
        tgt_key_padding_mask = self._make_tgt_key_padding_mask(tgt)
        tgt_mask = self._make_tgt_causal_mask(tgt.size(1), tgt.device)

        # Embeddings + positional encoding
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        # Transformer
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # Project to vocabulary
        logits = self.output_proj(output)  # (batch, tgt_len, tgt_vocab)
        return logits

    @torch.no_grad()
    def translate(self, src, sos_idx, eos_idx, max_len=64):
        """
        Greedy decoding for inference.

        Args:
            src: (1, src_len) single source sequence
            sos_idx, eos_idx: special token indices
            max_len: maximum output length
        Returns:
            decoded_ids: list of predicted token IDs
        """
        self.eval()
        device = src.device

        # Encode source
        src_key_padding_mask = self._make_src_key_padding_mask(src)
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask
        )

        # Start decoding
        decoded_ids = [sos_idx]
        tgt_tensor = torch.tensor([decoded_ids], dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_emb = self.pos_encoder(
                self.tgt_embedding(tgt_tensor) * math.sqrt(self.d_model)
            )
            tgt_mask = self._make_tgt_causal_mask(tgt_tensor.size(1), device)

            output = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

            logits = self.output_proj(output[:, -1, :])  # Last position
            next_token = logits.argmax(dim=-1).item()

            if next_token == eos_idx:
                break

            decoded_ids.append(next_token)
            tgt_tensor = torch.tensor([decoded_ids], dtype=torch.long, device=device)

        return decoded_ids[1:]  # Remove SOS

    @torch.no_grad()
    def translate_beam(self, src, sos_idx, eos_idx, max_len=64, beam_width=5):
        """
        Beam search decoding for better translations.

        Returns:
            best_sequence: list of token IDs (best beam)
        """
        self.eval()
        device = src.device

        # Encode source
        src_key_padding_mask = self._make_src_key_padding_mask(src)
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask
        )

        # Initialize beams: (score, token_ids)
        beams = [(0.0, [sos_idx])]
        complete = []

        for _ in range(max_len):
            candidates = []
            for score, seq in beams:
                if seq[-1] == eos_idx:
                    complete.append((score, seq))
                    continue

                tgt_tensor = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_emb = self.pos_encoder(
                    self.tgt_embedding(tgt_tensor) * math.sqrt(self.d_model)
                )
                tgt_mask = self._make_tgt_causal_mask(len(seq), device)

                output = self.transformer.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                )

                logits = self.output_proj(output[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                # Top-k next tokens
                topk_scores, topk_ids = log_probs.topk(beam_width)
                for k in range(beam_width):
                    new_score = score + topk_scores[k].item()
                    new_seq = seq + [topk_ids[k].item()]
                    candidates.append((new_score, new_seq))

            if not candidates:
                break

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        # Return best complete sequence, or best beam
        all_seqs = complete + beams
        if not all_seqs:
            return []
        best = max(all_seqs, key=lambda x: x[0] / max(len(x[1]), 1))
        result = best[1][1:]  # Remove SOS
        # Remove EOS if present
        if result and result[-1] == eos_idx:
            result = result[:-1]
        return result


def build_model_v2(config, src_vocab_size, tgt_vocab_size):
    """Factory function to create the Transformer model."""
    model = TransformerTranslator(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_encoder_layers=config.N_ENCODER_LAYERS,
        n_decoder_layers=config.N_DECODER_LAYERS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        max_seq_len=config.MAX_SEQ_LEN,
        pad_idx=0,
    ).to(config.DEVICE)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] Transformer model built!")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Src vocab:            {src_vocab_size}")
    print(f"  Tgt vocab:            {tgt_vocab_size}")

    return model
