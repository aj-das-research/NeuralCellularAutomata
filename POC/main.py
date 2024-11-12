import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

class NCAModule(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(NCAModule, self).__init__()
        self.conv = nn.Conv2d(state_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim, state_dim)

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.conv(x)
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1)  # (batch_size, height, width, channels)
        x = self.fc(x)
        x = x.permute(0, 3, 1, 2)  # (batch_size, channels, height, width)
        return x


class NCATransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, n_layers, state_dim, nca_steps):
        super(NCATransformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, hidden_dim))
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers
        )
        self.nca = NCAModule(state_dim, hidden_dim)
        self.nca_steps = nca_steps
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        src_emb = src_emb.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        tgt_emb = tgt_emb.permute(1, 0, 2)

        transformer_output = self.transformer(src_emb, tgt_emb)

        # Reshape for NCA: (batch_size, channels, height, width)
        nca_input = transformer_output.permute(1, 2, 0).unsqueeze(2)  # (batch_size, hidden_dim, seq_len, 1)
        for _ in range(self.nca_steps):
            nca_input = self.nca(nca_input)

        nca_output = nca_input.squeeze(2).permute(2, 0, 1)  # (seq_len, batch_size, hidden_dim)
        output = self.fc_out(nca_output)
        return output.permute(1, 0, 2)  # (batch_size, seq_len, output_dim)


# Sample data
src_seq = torch.randint(0, 100, (32, 10))  # (batch_size, seq_len)
tgt_seq = torch.randint(0, 100, (32, 10))

# Model configuration
input_dim = 100
hidden_dim = 64
output_dim = 100
n_heads = 4
n_layers = 2
state_dim = 64
nca_steps = 2

# Initialize the model
model = NCATransformer(input_dim, hidden_dim, output_dim, n_heads, n_layers, state_dim, nca_steps)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Forward pass
output = model(src_seq, tgt_seq)

# Compute loss
loss = criterion(output.reshape(-1, output_dim), tgt_seq.reshape(-1))
print(loss)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()
