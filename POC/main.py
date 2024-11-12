import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories for TensorBoard logs and intermediate outputs
log_dir = 'runs/nca_transformer'
intermediate_dir = 'intermediate_outputs'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(intermediate_dir, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=log_dir)

class NCAModule(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(NCAModule, self).__init__()
        self.conv = nn.Conv2d(state_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0))
        self.fc = nn.Linear(hidden_dim, state_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_dim)
        x = x.permute(0, 2, 1).unsqueeze(3)  # (batch_size, hidden_dim, seq_len, 1)
        x = self.conv(x)
        x = F.relu(x)
        x = x.squeeze(3).permute(0, 2, 1)  # (batch_size, seq_len, hidden_dim)
        x = self.fc(x)
        return x

class NCAPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, nca_steps):
        super(NCAPositionalEncoding, self).__init__()
        self.nca_steps = nca_steps
        self.nca = NCAModule(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_dim)
        pos_enc = torch.zeros_like(x)
        for _ in range(self.nca_steps):
            pos_enc = self.nca(pos_enc)
        return x + pos_enc

class NCASelfAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, nca_steps):
        super(NCASelfAttention, self).__init__()
        self.nca_steps = nca_steps
        self.nca = NCAModule(hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads)

    def forward(self, x):
        # x: (seq_len, batch_size, hidden_dim)
        for _ in range(self.nca_steps):
            x = self.nca(x)
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class NCAFeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim, nca_steps, dropout_rate=0.1):
        super(NCAFeedForward, self).__init__()
        self.nca_steps = nca_steps
        self.nca = NCAModule(hidden_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)

    def forward(self, x):
        # x: (seq_len, batch_size, hidden_dim)
        for _ in range(self.nca_steps):
            x = self.nca(x)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class NCAEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, nca_steps, dropout_rate=0.1):
        super(NCAEncoderLayer, self).__init__()
        self.self_attn = NCASelfAttention(hidden_dim, n_heads, nca_steps)
        self.feed_forward = NCAFeedForward(hidden_dim, ff_dim, nca_steps, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (seq_len, batch_size, hidden_dim)
        attn_output = self.self_attn(x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x

class NCAEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, ff_dim, n_layers, nca_steps, dropout_rate=0.1):
        super(NCAEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.positional_encoding = NCAPositionalEncoding(hidden_dim, nca_steps)
        self.layers = nn.ModuleList([
            NCAEncoderLayer(hidden_dim, n_heads, ff_dim, nca_steps, dropout_rate)
            for _ in range(n_layers)
        ])

    def forward(self, src):
        # src: (batch_size, seq_len)
        x = self.embedding(src)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        intermediate_outputs = []
        for layer in self.layers:
            x = layer(x)
            intermediate_outputs.append(x.permute(1, 0, 2))  # Store intermediate outputs
        return x, intermediate_outputs

class NCATransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, ff_dim, n_layers, nca_steps, dropout_rate):
        super(NCATransformer, self).__init__()
        self.encoder = NCAEncoder(input_dim, hidden_dim, n_heads, ff_dim, n_layers, nca_steps, dropout_rate)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, n_heads, ff_dim, dropout=dropout_rate),
            num_layers=n_layers
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, tgt):
        memory, intermediate_outputs = self.encoder(src)
        tgt_emb = self.encoder.embedding(tgt)
        tgt_emb = self.encoder.positional_encoding(tgt_emb)
        tgt_emb = tgt_emb.permute(1, 0, 2)  # (tgt_seq_len, batch_size, hidden_dim)
        output = self.decoder(tgt_emb, memory)
        output = self.fc_out(output)
        return output.permute(1, 0, 2), intermediate_outputs  # (batch_size, tgt_seq_len, output_dim)
    
# Function to save intermediate outputs as images
def save_intermediate_outputs(intermediate_outputs, epoch, batch_idx):
    for layer_idx, output in enumerate(intermediate_outputs):
        output = output.permute(1, 0, 2).cpu().detach().numpy()  # (batch_size, seq_len, hidden_dim)
        sample_output = output[0]  # Select the first sample in the batch
        plt.figure(figsize=(10, 5))
        plt.imshow(sample_output, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f'Layer {layer_idx + 1} Output at Epoch {epoch + 1}, Batch {batch_idx + 1}')
        plt.xlabel('Hidden Dimension')
        plt.ylabel('Sequence Length')
        plt.savefig(os.path.join(intermediate_dir, f'layer{layer_idx + 1}_epoch{epoch + 1}_batch{batch_idx + 1}.png'))
        plt.close()

# Hyperparameters
input_dim = 100
hidden_dim = 64
output_dim = 100
n_heads = 4
ff_dim = 256
n_layers = 2
nca_steps = 5
dropout_rate = 0.1
learning_rate = 0.001
batch_size = 32
num_epochs = 50
patience = 5  # Early stopping patience

# Sample data generation
def generate_sample_data(num_samples, seq_len, vocab_size):
    src_data = torch.randint(0, vocab_size, (num_samples, seq_len))
    tgt_data = torch.randint(0, vocab_size, (num_samples, seq_len))
    return src_data, tgt_data

# Generate dataset
num_samples = 1000
seq_len = 10
src_data, tgt_data = generate_sample_data(num_samples, seq_len, input_dim)

# Create DataLoader
dataset = TensorDataset(src_data, tgt_data)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = NCATransformer(input_dim, hidden_dim, output_dim, n_heads, ff_dim, n_layers, nca_steps, dropout_rate).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Early stopping variables
best_val_loss = float('inf')
epochs_no_improve = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_idx, (src_batch, tgt_batch) in enumerate(train_loader):
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        optimizer.zero_grad()
        output, intermediate_outputs = model(src_batch, tgt_batch)
        loss = criterion(output.reshape(-1, output_dim), tgt_batch.reshape(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * src_batch.size(0)
        
        # Save intermediate outputs
        save_intermediate_outputs(intermediate_outputs, epoch, batch_idx)

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for src_batch, tgt_batch in val_loader:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            output, _ = model(src_batch, tgt_batch)
            loss = criterion(output.reshape(-1, output_dim), tgt_batch.reshape(-1))
            val_loss += loss.item() * src_batch.size(0)

    val_loss /= len(val_loader.dataset)

    # Log losses to TensorBoard
    writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print('Early stopping triggered.')
            break

writer.close()

# Load the best model for evaluation or inference
model.load_state_dict(torch.load('best_model.pth'))
model.eval()



