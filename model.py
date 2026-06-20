import torch
import torch.nn as nn

action_dim = 156
d_model = 256
num_heads = 8
max_seq = 128

class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(action_dim, d_model)
    def forward(self, x):
        return self.proj(x)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
    def forward(self, x):
        normed = self.norm1(x)
        T = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        x = x + self.attn(normed, normed, normed, attn_mask=mask, is_causal=True)[0]
        x = x + self.ffn(self.norm2(x))
        return x

class ActionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(d_model, action_dim)
    def forward(self, x):
        return self.proj(x)

class Mars(nn.Module):
    def __init__(self, num_layers=1):
        super().__init__()
        self.encoder = ActionEncoder()
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(num_layers)])
        self.decoder = ActionDecoder()
    def forward(self, x):
        x = self.encoder(x)
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        return self.decoder(x[:, -1, :])
