#!/usr/bin/env python3
"""
Sourashtra Translation Model - Quick Start Training Script
==========================================================
This script trains a simple Seq2Seq model for Roman Sourashtra → English translation
Perfect for validating your pipeline before moving to larger models!

Requirements:
    pip install torch pandas numpy scikit-learn tqdm
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle
from tqdm import tqdm

# Configuration
class Config:
    # Data
    data_file = "cleaned_data/translation_roman_english.csv"
    
    # Model hyperparameters
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2
    dropout = 0.3
    
    # Training
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.001
    max_seq_length = 50
    
    # Vocabulary
    min_word_freq = 2
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpoints
    save_dir = "model_checkpoints"

config = Config()

# ==========================================
# 1. DATA PREPARATION
# ==========================================

class Vocabulary:
    """Build vocabulary from text"""
    def __init__(self, min_freq=2):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word_freq = Counter()
        self.min_freq = min_freq
        
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        for text in texts:
            words = text.lower().split()
            self.word_freq.update(words)
        
        # Add words that appear at least min_freq times
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        
    def encode(self, text, max_len=None):
        """Convert text to indices"""
        words = text.lower().split()
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        if max_len:
            if len(indices) < max_len:
                indices += [self.word2idx['<PAD>']] * (max_len - len(indices))
            else:
                indices = indices[:max_len]
        
        return indices
    
    def decode(self, indices):
        """Convert indices back to text"""
        words = []
        for idx in indices:
            if idx == self.word2idx['<EOS>']:
                break
            if idx != self.word2idx['<PAD>'] and idx != self.word2idx['<SOS>']:
                words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)

class TranslationDataset(Dataset):
    """Dataset for translation pairs"""
    def __init__(self, source_texts, target_texts, source_vocab, target_vocab, max_len):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        source = self.source_texts[idx]
        target = self.target_texts[idx]
        
        # Encode source
        source_indices = self.source_vocab.encode(source, self.max_len)
        
        # Encode target (add SOS and EOS)
        target_words = target.lower().split()
        target_indices = [self.target_vocab.word2idx['<SOS>']]
        target_indices += [self.target_vocab.word2idx.get(word, self.target_vocab.word2idx['<UNK>']) 
                          for word in target_words]
        target_indices.append(self.target_vocab.word2idx['<EOS>'])
        
        # Pad target
        if len(target_indices) < self.max_len + 2:
            target_indices += [self.target_vocab.word2idx['<PAD>']] * (self.max_len + 2 - len(target_indices))
        else:
            target_indices = target_indices[:self.max_len + 2]
        
        return (
            torch.tensor(source_indices, dtype=torch.long),
            torch.tensor(target_indices, dtype=torch.long)
        )

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================

class Encoder(nn.Module):
    """Encoder with bidirectional GRU"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, 
                         batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(x))  # [batch_size, seq_len, embedding_dim]
        outputs, hidden = self.gru(embedded)  # outputs: [batch_size, seq_len, hidden_dim*2]
        return outputs, hidden

class Attention(nn.Module):
    """Bahdanau Attention"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        
    def forward(self, query, keys):
        # query: [batch_size, hidden_dim*2]
        # keys: [batch_size, seq_len, hidden_dim*2]
        
        # Expand query to match keys
        query = query.unsqueeze(1)  # [batch_size, 1, hidden_dim*2]
        
        # Calculate attention scores
        scores = self.V(torch.tanh(self.W1(query) + self.W2(keys)))  # [batch_size, seq_len, 1]
        scores = scores.squeeze(2)  # [batch_size, seq_len]
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len]
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), keys)  # [batch_size, 1, hidden_dim*2]
        context = context.squeeze(1)  # [batch_size, hidden_dim*2]
        
        return context, attention_weights

class Decoder(nn.Module):
    """Decoder with attention"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(embedding_dim + hidden_dim * 2, hidden_dim * 2, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden, encoder_outputs):
        # x: [batch_size, 1]
        # hidden: [num_layers*2, batch_size, hidden_dim] (from bidirectional encoder)
        # encoder_outputs: [batch_size, seq_len, hidden_dim*2]
        
        embedded = self.dropout(self.embedding(x))  # [batch_size, 1, embedding_dim]
        
        # Use last layer of hidden state for attention
        query = hidden[-1]  # [batch_size, hidden_dim*2]
        context, attention_weights = self.attention(query, encoder_outputs)
        
        # Concatenate embedded input and context
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # [batch_size, 1, embedding_dim + hidden_dim*2]
        
        output, hidden = self.gru(rnn_input, hidden)
        
        # Predict next word
        prediction = self.fc(output.squeeze(1))  # [batch_size, vocab_size]
        
        return prediction, hidden, attention_weights

class Seq2Seq(nn.Module):
    """Sequence to Sequence Model"""
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # source: [batch_size, source_len]
        # target: [batch_size, target_len]
        
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        
        # Encode source
        encoder_outputs, hidden = self.encoder(source)
        
        # First input to decoder is SOS token
        decoder_input = target[:, 0].unsqueeze(1)
        
        for t in range(1, target_len):
            # Decode
            prediction, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t] = prediction
            
            # Teacher forcing
            use_teacher_forcing = np.random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            decoder_input = target[:, t].unsqueeze(1) if use_teacher_forcing else top1.unsqueeze(1)
        
        return outputs

# ==========================================
# 3. TRAINING FUNCTIONS
# ==========================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    for source, target in tqdm(dataloader, desc="Training"):
        source = source.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(source, target)
        
        # Calculate loss (ignore padding)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        target = target[:, 1:].reshape(-1)
        
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for source, target in tqdm(dataloader, desc="Evaluating"):
            source = source.to(device)
            target = target.to(device)
            
            output = model(source, target, teacher_forcing_ratio=0)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            target = target[:, 1:].reshape(-1)
            
            loss = criterion(output, target)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def translate(model, sentence, source_vocab, target_vocab, device, max_len=50):
    """Translate a sentence"""
    model.eval()
    
    with torch.no_grad():
        # Encode source
        source_indices = source_vocab.encode(sentence, max_len)
        source_tensor = torch.tensor(source_indices, dtype=torch.long).unsqueeze(0).to(device)
        
        encoder_outputs, hidden = model.encoder(source_tensor)
        
        # Start with SOS token
        decoder_input = torch.tensor([target_vocab.word2idx['<SOS>']], dtype=torch.long).unsqueeze(0).to(device)
        
        decoded_words = []
        
        for _ in range(max_len):
            prediction, hidden, _ = model.decoder(decoder_input, hidden, encoder_outputs)
            top1 = prediction.argmax(1)
            
            if top1.item() == target_vocab.word2idx['<EOS>']:
                break
            
            decoded_words.append(target_vocab.idx2word[top1.item()])
            decoder_input = top1.unsqueeze(0)
        
        return ' '.join(decoded_words)

# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================

def main():
    print("="*80)
    print("SOURASHTRA TRANSLATION MODEL - TRAINING")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from {config.data_file}...")
    df = pd.read_csv(config.data_file)
    print(f"Total pairs: {len(df):,}")
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Train: {len(train_df):,}, Validation: {len(val_df):,}")
    
    # Build vocabularies
    print("\nBuilding vocabularies...")
    source_vocab = Vocabulary(min_freq=config.min_word_freq)
    target_vocab = Vocabulary(min_freq=config.min_word_freq)
    
    source_vocab.build_vocab(df['source'].tolist())
    target_vocab.build_vocab(df['target'].tolist())
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TranslationDataset(
        train_df['source'].tolist(),
        train_df['target'].tolist(),
        source_vocab,
        target_vocab,
        config.max_seq_length
    )
    
    val_dataset = TranslationDataset(
        val_df['source'].tolist(),
        val_df['target'].tolist(),
        source_vocab,
        target_vocab,
        config.max_seq_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Create model
    print("\nInitializing model...")
    encoder = Encoder(
        len(source_vocab.word2idx),
        config.embedding_dim,
        config.hidden_dim,
        config.num_layers,
        config.dropout
    )
    
    decoder = Decoder(
        len(target_vocab.word2idx),
        config.embedding_dim,
        config.hidden_dim,
        config.num_layers,
        config.dropout
    )
    
    model = Seq2Seq(encoder, decoder, config.device).to(config.device)
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=target_vocab.word2idx['<PAD>'])
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on: {config.device}")
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        val_loss = evaluate(model, val_loader, criterion, config.device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pt')
            print("✓ Saved best model!")
        
        # Test translation
        if (epoch + 1) % 5 == 0:
            print("\nSample translations:")
            test_sentences = [
                train_df['source'].iloc[0],
                train_df['source'].iloc[100],
                val_df['source'].iloc[0]
            ]
            
            for sent in test_sentences[:3]:
                translation = translate(model, sent, source_vocab, target_vocab, config.device)
                print(f"  Source: {sent}")
                print(f"  Translation: {translation}")
                print()
    
    # Save vocabularies
    print("\nSaving vocabularies...")
    with open('source_vocab.pkl', 'wb') as f:
        pickle.dump(source_vocab, f)
    with open('target_vocab.pkl', 'wb') as f:
        pickle.dump(target_vocab, f)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print("Saved files:")
    print("  - best_model.pt (model checkpoint)")
    print("  - source_vocab.pkl (source vocabulary)")
    print("  - target_vocab.pkl (target vocabulary)")

if __name__ == "__main__":
    main()
