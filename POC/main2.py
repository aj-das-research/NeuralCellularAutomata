import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import numpy as np
from collections import Counter
import math
from torch.nn.utils.rnn import pad_sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import numpy as np
from collections import Counter
import math
from torch.nn.utils.rnn import pad_sequence
import random
import os
import pandas as pd


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads each batch of source and target sequences to the same length, 
    ensuring compatibility for stacking in DataLoader.
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences in each batch to the same length
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)  # Pad source sequences
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)  # Pad target sequences
    
    return src_batch, tgt_batch


# ============ Tokenizer Class ============

class SimpleTokenizer:
    """Simple word-level tokenizer"""
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.token2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2token = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.word_counts = Counter()
        
    def fit(self, texts: List[str]):
        """Build vocabulary from texts"""
        for text in texts:
            words = text.strip().lower().split()
            self.word_counts.update(words)
            
        for word, _ in self.word_counts.most_common(self.vocab_size - 4):
            if word not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[word] = idx
                self.idx2token[idx] = word
                
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Convert text to token indices"""
        words = text.strip().lower().split()
        tokens = [self.token2idx.get(word, self.token2idx['<unk>']) for word in words]
        tokens = [self.token2idx['<sos>']] + tokens + [self.token2idx['<eos>']]
        
        if max_length:
            tokens = tokens[:max_length]
            
        return tokens
        
    def decode(self, tokens: List[int]) -> str:
        """Convert token indices back to text"""
        words = []
        for token in tokens:
            word = self.idx2token.get(token, '<unk>')
            if word in ['<pad>', '<sos>', '<eos>']:
                continue
            words.append(word)
        return ' '.join(words)


# ============ Dataset Class ============

class SimpleTranslationDataset(Dataset):
    """Simple parallel translation dataset"""
    def __init__(self, source_texts: List[str], target_texts: List[str],
                 src_tokenizer: SimpleTokenizer, tgt_tokenizer: SimpleTokenizer, max_len: int = 128):
        assert len(source_texts) == len(target_texts)
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        
    def __len__(self) -> int:
        return len(self.source_texts)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_text = self.source_texts[idx]
        tgt_text = self.target_texts[idx]
        
        src_tokens = self.src_tokenizer.encode(src_text, self.max_len)
        tgt_tokens = self.tgt_tokenizer.encode(tgt_text, self.max_len)
        
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)


# ============ Neural Cellular Automata Cell Class ============

class NCACell(nn.Module):
    """Enhanced NCA cell with perception and update rules."""
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Perception network
        self.perception = nn.Sequential(
            nn.Conv1d(state_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, state_dim, kernel_size=1)  # Ensure output matches state_dim
        )
        
        # Update network
        self.update = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh()
        )
        
        self.update_rate = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Reshape for 1D conv: (batch_size, state_dim, seq_len)
        x_conv = x.transpose(1, 2)
        dx = self.perception(x_conv).transpose(1, 2)  # Back to (batch_size, seq_len, state_dim)
        
        # Apply update transformation
        dx = self.update(dx)
        
        # Ensure that dx is broadcastable to x
        mask = torch.rand_like(x) < torch.sigmoid(self.update_rate)
        x_new = x + dx * mask.float()
        
        # print(f"NCACell forward: x shape {x.shape}, dx shape {dx.shape}, mask shape {mask.shape}")
        return x_new



# ============ Neural Cellular Automata Self-Attention Class ============

class NCASelfAttention(nn.Module):
    """Self-attention mechanism with NCA-based preprocessing."""
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        self.nca_cell = NCACell(hidden_dim, hidden_dim * 2)
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # NCA preprocessing
        x = self.nca_cell(x)
        
        # Linear transformations and split into heads
        q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.squeeze(1).squeeze(1)  # Shape: (batch_size, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, v)  # (batch_size, n_heads, seq_len, head_dim)
        
        # Concatenate heads and apply final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.fc(context)
        
        return output


# ============ Transformer Encoder Layer with NCA ============

class NCAEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = NCASelfAttention(hidden_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        _x = self.self_attn(self.norm1(x), mask)
        x = x + self.dropout(_x)
        
        # Feedforward with residual connection
        _x = self.ff(self.norm2(x))
        x = x + self.dropout(_x)
        
        return x


# ============ Transformer Model with NCA ============

class NCATransformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 hidden_dim: int = 512, n_heads: int = 8, n_layers: int = 6, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # Embedding layers
        self.src_embed = nn.Embedding(src_vocab_size, hidden_dim)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, hidden_dim)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            NCAEncoderLayer(hidden_dim, n_heads, ff_dim, dropout) for _ in range(n_layers)
        ])
        
        # Decoder layers (can replicate similar to encoder for simplicity)
        self.decoder_layers = nn.ModuleList([
            NCAEncoderLayer(hidden_dim, n_heads, ff_dim, dropout) for _ in range(n_layers)
        ])
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.dropout(self.src_embed(src))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x
    
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.dropout(self.tgt_embed(tgt))
        for layer in self.decoder_layers:
            x = layer(x, tgt_mask)
        return self.output_proj(x)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(tgt, memory, tgt_mask)


# ============ Mask Creation ============

def create_mask(src, tgt, pad_idx=0):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)
    return src_mask, tgt_mask


# ============ Training & Evaluation Functions ============

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for i, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        src_mask, tgt_mask = create_mask(src, tgt[:, :-1])
        
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1], src_mask, tgt_mask)
        output = output.view(-1, output.size(-1))
        target = tgt[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask, tgt_mask = create_mask(src, tgt[:, :-1])
            output = model(src, tgt[:, :-1], src_mask, tgt_mask)
            output = output.view(-1, output.size(-1))
            target = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# ============ Usage Example for Training and Evaluation ============

# def create_sample_dataset() -> Tuple[List[str], List[str]]:
#     """Create a small sample translation dataset"""
#     en_texts = [
#         "hello how are you",
#         "what is your name",
#         "i love programming",
#         "the weather is nice today",
#         "can you help me please",
#         "where are you going",
#         "i am learning python",
#         "this is a good day",
#         "thank you very much",
#         "see you tomorrow"
#     ]
    
#     de_texts = [
#         "hallo wie geht es dir",
#         "wie heißt du",
#         "ich liebe programmierung",
#         "das wetter ist heute schön",
#         "kannst du mir bitte helfen",
#         "wohin gehst du",
#         "ich lerne python",
#         "das ist ein guter tag",
#         "vielen dank",
#         "bis morgen"
#     ]
    
#     en_texts *= 100  # To create a larger dataset
#     de_texts *= 100
#     return en_texts, de_texts

# Update the create_sample_dataset function
def create_sample_dataset() -> Tuple[List[str], List[str]]:
    """Load dataset from a CSV file containing English and French sentence pairs."""
    # Path to your CSV file
    csv_file_path = 'en-fr.csv'  # Replace with the actual path to your CSV file
    
    # Check if the file exists
    if not os.path.exists(csv_file_path):
        print(f"CSV file not found at path: {csv_file_path}")
        exit(1)
    
    # Load the dataset using pandas
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        exit(1)
    
    # Check if required columns are present
    if 'en' not in df.columns or 'fr' not in df.columns:
        print("CSV file must contain 'en' and 'fr' columns.")
        exit(1)
    
    # Extract English and French sentences
    en_texts = df['en'].astype(str).tolist()
    fr_texts = df['fr'].astype(str).tolist()
    
    # Shuffle the dataset
    combined = list(zip(en_texts, fr_texts))
    random.shuffle(combined)
    en_texts[:], fr_texts[:] = zip(*combined)
    
    # Optionally, limit the dataset size for testing
    en_texts = en_texts[:1000]
    print(f"sample data point: {en_texts[0]}")
    fr_texts = fr_texts[:1000]

    
    return en_texts, fr_texts


def train_model():
    print("Starting model training...")
    
    # Configuration settings
    print("Setting up configuration...")
    config = {
        'hidden_dim': 256,
        'n_heads': 8,
        'n_layers': 3,
        'ff_dim': 512,
        'max_len': 64,
        'dropout': 0.1,
        'batch_size': 2,
        'num_epochs': 10,
        'learning_rate': 0.0005,
        'pad_idx': 0
    }
    
    # Setting the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset from CSV
    print("Loading dataset from CSV...")
    src_texts, tgt_texts = create_sample_dataset()
    print(f"Dataset loaded with {len(src_texts)} samples")
    
    # Initialize tokenizers and fit on sample data
    print("Initializing and fitting tokenizers...")
    src_tokenizer = SimpleTokenizer(vocab_size=5000)
    tgt_tokenizer = SimpleTokenizer(vocab_size=5000)
    src_tokenizer.fit(src_texts)
    tgt_tokenizer.fit(tgt_texts)
    print(f"Source vocabulary size: {len(src_tokenizer.token2idx)}")
    print(f"Target vocabulary size: {len(tgt_tokenizer.token2idx)}")
    
    # Split data into training and validation sets
    print("Splitting data into train/validation sets...")
    train_size = int(0.9 * len(src_texts))
    train_src, train_tgt = src_texts[:train_size], tgt_texts[:train_size]
    val_src, val_tgt = src_texts[train_size:], tgt_texts[train_size:]
    print(f"Training samples: {len(train_src)}, Validation samples: {len(val_src)}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SimpleTranslationDataset(train_src, train_tgt, src_tokenizer, tgt_tokenizer, config['max_len'])
    val_dataset = SimpleTranslationDataset(val_src, val_tgt, src_tokenizer, tgt_tokenizer, config['max_len'])
    
    # DataLoaders for batching
    print("Setting up data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Model initialization
    print("Initializing model...")
    model = NCATransformer(
        src_vocab_size=len(src_tokenizer.token2idx),
        tgt_vocab_size=len(tgt_tokenizer.token2idx),
        hidden_dim=config['hidden_dim'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout']
    ).to(device)
    print("Model initialized successfully")
    
    # Optimizer and loss function
    print("Setting up optimizer and loss function...")
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=config['pad_idx'])
    
    # Training and Evaluation
    print("\nStarting training loop...")
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Training step
        print("Starting training step...")
        model.train()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validation step
        print("Starting validation step...")
        model.eval()
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Check for early stopping
        print("Checking early stopping conditions...")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("New best validation loss achieved! Saving model...")
            torch.save({'model_state_dict': model.state_dict(), 'src_tokenizer': src_tokenizer, 'tgt_tokenizer': tgt_tokenizer}, 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    print("\nTraining completed!")
    return model, src_tokenizer, tgt_tokenizer

if __name__ == "__main__":
    print("Starting main program...")
    model, src_tokenizer, tgt_tokenizer = train_model()
    
    # Sample translation tests
    print("\nRunning translation tests...")
    test_sentences = [
        "you are good",
        "good",
        "hello how are you",
        "i love programming",
        "thank you very much"
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\nTesting translations:")
    for text in test_sentences:
        print(f"\nTranslating: '{text}'")
        translation = translate_text(model, src_tokenizer, tgt_tokenizer, text, device)
        print(f"Input: {text}")
        print(f"Translation: {translation}")



# def train_model():
#     # Configuration settings
#     config = {
#         'hidden_dim': 256,
#         'n_heads': 8,
#         'n_layers': 3,
#         'ff_dim': 512,
#         'max_len': 64,
#         'dropout': 0.1,
#         'batch_size': 32,
#         'num_epochs': 10,
#         'learning_rate': 0.0001,
#         'pad_idx': 0
#     }
    
#     # Setting the device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # Sample dataset creation
#     src_texts, tgt_texts = create_sample_dataset()
    
#     # Initialize tokenizers and fit on sample data
#     src_tokenizer = SimpleTokenizer(vocab_size=1000)
#     tgt_tokenizer = SimpleTokenizer(vocab_size=1000)
#     src_tokenizer.fit(src_texts)
#     tgt_tokenizer.fit(tgt_texts)
    
#     # Split data into training and validation sets
#     train_size = int(0.8 * len(src_texts))
#     train_src, train_tgt = src_texts[:train_size], tgt_texts[:train_size]
#     val_src, val_tgt = src_texts[train_size:], tgt_texts[train_size:]
    
#     # Create datasets
#     train_dataset = SimpleTranslationDataset(train_src, train_tgt, src_tokenizer, tgt_tokenizer, config['max_len'])
#     val_dataset = SimpleTranslationDataset(val_src, val_tgt, src_tokenizer, tgt_tokenizer, config['max_len'])

#     # DataLoaders for batching
#     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

#     # Model initialization
#     model = NCATransformer(
#         src_vocab_size=len(src_tokenizer.token2idx),
#         tgt_vocab_size=len(tgt_tokenizer.token2idx),
#         hidden_dim=config['hidden_dim'],
#         n_heads=config['n_heads'],
#         n_layers=config['n_layers'],
#         ff_dim=config['ff_dim'],
#         dropout=config['dropout']
#     ).to(device)

#     # Optimizer and loss function
#     optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
#     criterion = nn.CrossEntropyLoss(ignore_index=config['pad_idx'])

#     # Training and Evaluation
#     best_val_loss = float('inf')
#     patience = 3
#     patience_counter = 0

#     for epoch in range(config['num_epochs']):
#         print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

#         # Training step
#         model.train()
#         train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
#         print(f"Training Loss: {train_loss:.4f}")

#         # Validation step
#         model.eval()
#         val_loss = evaluate(model, val_loader, criterion, device)
#         print(f"Validation Loss: {val_loss:.4f}")

#         # Check for early stopping
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save({'model_state_dict': model.state_dict(), 'src_tokenizer': src_tokenizer, 'tgt_tokenizer': tgt_tokenizer}, 'best_model.pt')
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print("Early stopping triggered")
#                 break

#     print("Training completed!")
#     return model, src_tokenizer, tgt_tokenizer

# # Function to generate translations using the trained model
# def translate_text(model: nn.Module, src_tokenizer: SimpleTokenizer, tgt_tokenizer: SimpleTokenizer, text: str, device: torch.device, max_len: int = 64) -> str:
#     model.eval()
#     src_tokens = src_tokenizer.encode(text, max_len)
#     src_tensor = torch.tensor([src_tokens]).to(device)

#     with torch.no_grad():
#         memory = model.encode(src_tensor)
#         output_indices = [tgt_tokenizer.token2idx['<sos>']]

#         for _ in range(max_len):
#             tgt_tensor = torch.tensor([output_indices]).to(device)
#             output = model.decode(tgt_tensor, memory)
#             next_token = output[0, -1].argmax().item()
#             output_indices.append(next_token)
#             if next_token == tgt_tokenizer.token2idx['<eos>']:
#                 break

#     return tgt_tokenizer.decode(output_indices)

# if __name__ == "__main__":
#     model, src_tokenizer, tgt_tokenizer = train_model()
    
#     # Sample translation tests
#     test_sentences = [
#         "you are good",
#         "good",
#     ]
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     print("\nTesting translations:")
#     for text in test_sentences:
#         translation = translate_text(model, src_tokenizer, tgt_tokenizer, text, device)
#         print(f"Input: {text}")
#         print(f"Translation: {translation}")

    
