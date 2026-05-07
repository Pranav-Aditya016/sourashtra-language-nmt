"""
Seq2Seq Model with Bahdanau Attention for Sourashtra Translation
=================================================================
Character-level Encoder-Decoder with:
  - Bidirectional GRU Encoder
  - Bahdanau (Additive) Attention
  - GRU Decoder with attention context
  - Bridge layer to transform encoder hidden → decoder hidden

Architecture chosen because:
  - GRU is faster than LSTM, comparable performance for small data
  - Bidirectional encoder captures full input context
  - Bahdanau attention works well for character-level models
  - Character-level handles morphological variations in Sourashtra
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Bidirectional GRU Encoder.

    Input:  (batch, src_len) of character indices
    Output: (encoder_outputs, hidden)
      - encoder_outputs: (batch, src_len, hidden_dim * 2)
      - hidden: (num_layers, batch, hidden_dim * 2)  [after bridge]
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch, src_len)
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, emb)
        outputs, hidden = self.rnn(embedded)
        # outputs: (batch, src_len, hidden*2)
        # hidden: (num_layers*2, batch, hidden)  [fwd + bwd interleaved]
        return outputs, hidden


class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) Attention.

    score(s_t, h_i) = v^T * tanh(W_s * s_t + W_h * h_i)
    """

    def __init__(self, decoder_hidden_dim, encoder_hidden_dim):
        super().__init__()
        self.W_s = nn.Linear(decoder_hidden_dim, decoder_hidden_dim, bias=False)
        self.W_h = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=False)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: (batch, dec_hidden)
            encoder_outputs: (batch, src_len, enc_hidden)
            mask: (batch, src_len) - True for padded positions
        Returns:
            context: (batch, enc_hidden)
            attention_weights: (batch, src_len)
        """
        # (batch, 1, dec_hidden) + (batch, src_len, dec_hidden)
        score = self.v(
            torch.tanh(
                self.W_s(decoder_hidden).unsqueeze(1) +
                self.W_h(encoder_outputs)
            )
        ).squeeze(-1)  # (batch, src_len)

        if mask is not None:
            score = score.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(score, dim=-1)  # (batch, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        # context: (batch, enc_hidden)
        return context, attn_weights


class Decoder(nn.Module):
    """
    GRU Decoder with Bahdanau Attention.

    At each step:
      1. Embed the previous token
      2. Compute attention over encoder outputs
      3. Concatenate [embedded, context] → GRU input
      4. GRU step
      5. Linear → vocab logits
    """

    def __init__(self, vocab_size, embedding_dim, decoder_hidden_dim,
                 encoder_hidden_dim, num_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = BahdanauAttention(decoder_hidden_dim, encoder_hidden_dim)
        self.rnn = nn.GRU(
            embedding_dim + encoder_hidden_dim, decoder_hidden_dim,
            num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_out = nn.Linear(decoder_hidden_dim + encoder_hidden_dim + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, encoder_outputs, mask=None):
        """
        Single decoding step.

        Args:
            input_token: (batch, 1) - previous token indices
            hidden: (num_layers, batch, dec_hidden)
            encoder_outputs: (batch, src_len, enc_hidden)
            mask: (batch, src_len) - True for pad positions
        Returns:
            prediction: (batch, vocab_size) - logits
            hidden: (num_layers, batch, dec_hidden)
            attn_weights: (batch, src_len)
        """
        embedded = self.dropout(self.embedding(input_token))  # (batch, 1, emb)

        # Attention using top decoder layer hidden state
        query = hidden[-1]  # (batch, dec_hidden)
        context, attn_weights = self.attention(query, encoder_outputs, mask)

        # GRU input = [embedded, context]
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        # (batch, 1, emb + enc_hidden)

        output, hidden = self.rnn(rnn_input, hidden)
        # output: (batch, 1, dec_hidden)

        # Prediction: concat output, context, embedded → linear
        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=-1)
        )  # (batch, vocab_size)

        return prediction, hidden, attn_weights


class BridgeLayer(nn.Module):
    """
    Transform bidirectional encoder hidden states into decoder initial hidden.
    Combines forward and backward directions.
    """

    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # Encoder has 2*num_layers (bidir), decoder has num_layers
        self.bridges = nn.ModuleList([
            nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, encoder_hidden):
        """
        Args:
            encoder_hidden: (num_layers * 2, batch, hidden_dim)
        Returns:
            decoder_hidden: (num_layers, batch, decoder_hidden_dim)
        """
        # Reshape: (num_layers, 2, batch, hidden) → concatenate directions
        batch_size = encoder_hidden.shape[1]
        hidden_dim = encoder_hidden.shape[2]
        # encoder_hidden is (num_layers*2, batch, hidden)
        # Even indices are forward, odd indices are backward
        decoder_hiddens = []
        for i in range(self.num_layers):
            fwd = encoder_hidden[2 * i]       # (batch, hidden)
            bwd = encoder_hidden[2 * i + 1]   # (batch, hidden)
            combined = torch.cat([fwd, bwd], dim=-1)  # (batch, hidden*2)
            decoder_hiddens.append(torch.tanh(self.bridges[i](combined)))

        return torch.stack(decoder_hiddens, dim=0)  # (num_layers, batch, dec_hidden)


class Seq2SeqAttn(nn.Module):
    """
    Full Seq2Seq model: Encoder + Bridge + Decoder with Attention.
    """

    def __init__(self, encoder, decoder, bridge, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bridge = bridge
        self.device = device

    def create_mask(self, src, pad_idx=0):
        """Create mask for padded positions (True = pad)."""
        return src == pad_idx

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch, src_len)
            tgt: (batch, tgt_len) - includes <SOS> at start
            teacher_forcing_ratio: probability of using true target
        Returns:
            outputs: (batch, tgt_len, tgt_vocab_size) - logits
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.vocab_size

        # Storage for outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size, device=self.device)

        # Encode
        encoder_outputs, encoder_hidden = self.encoder(src)

        # Bridge: transform encoder hidden → decoder hidden
        decoder_hidden = self.bridge(encoder_hidden)

        # Padding mask for attention
        mask = self.create_mask(src)

        # First decoder input is <SOS>
        decoder_input = tgt[:, 0].unsqueeze(1)  # (batch, 1)

        for t in range(1, tgt_len):
            prediction, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, mask
            )
            outputs[:, t] = prediction

            # Teacher forcing decision
            if torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                decoder_input = prediction.argmax(dim=-1).unsqueeze(1)

        return outputs

    def translate(self, src, sos_idx, eos_idx, max_len=120):
        """
        Translate a single source sequence (greedy decoding).

        Args:
            src: (1, src_len) - single source tensor
            sos_idx: SOS token index
            eos_idx: EOS token index
            max_len: maximum output length
        Returns:
            decoded_indices: list of predicted indices
            attention_weights: (output_len, src_len) attention matrix
        """
        self.eval()
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encoder(src)
            decoder_hidden = self.bridge(encoder_hidden)
            mask = self.create_mask(src)

            decoder_input = torch.tensor([[sos_idx]], device=self.device)
            decoded_indices = []
            attention_matrix = []

            for _ in range(max_len):
                prediction, decoder_hidden, attn_weights = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, mask
                )
                top1 = prediction.argmax(dim=-1)
                token_idx = top1.item()

                if token_idx == eos_idx:
                    break

                decoded_indices.append(token_idx)
                attention_matrix.append(attn_weights.squeeze(0).cpu())
                decoder_input = top1.unsqueeze(0)

            if attention_matrix:
                attention_matrix = torch.stack(attention_matrix)
            else:
                attention_matrix = torch.zeros(1, src.shape[1])

        return decoded_indices, attention_matrix


def build_model(config, src_vocab_size, tgt_vocab_size):
    """Factory function to build the full Seq2Seq model."""
    enc_hidden = config.ENCODER_HIDDEN_DIM
    dec_hidden = config.DECODER_HIDDEN_DIM

    encoder = Encoder(
        vocab_size=src_vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=enc_hidden,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    )

    decoder = Decoder(
        vocab_size=tgt_vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        decoder_hidden_dim=dec_hidden,
        encoder_hidden_dim=enc_hidden * 2,   # bidirectional
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    )

    bridge = BridgeLayer(
        encoder_hidden_dim=enc_hidden,
        decoder_hidden_dim=dec_hidden,
        num_layers=config.NUM_LAYERS,
    )

    model = Seq2SeqAttn(encoder, decoder, bridge, config.DEVICE)
    model = model.to(config.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] Model built!")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Encoder params:       {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Decoder params:       {sum(p.numel() for p in decoder.parameters()):,}")
    print(f"  Bridge params:        {sum(p.numel() for p in bridge.parameters()):,}")

    return model
