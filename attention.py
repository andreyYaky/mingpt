import torch
from torch import nn

class CausalSelfAttentionBlock(nn.Module):

    # communication + computation
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.self_attention = nn.MultiheadAttention(n_embd, n_head, dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        residue = x

        x = self.ln1(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(sz=x.shape[1])
        x = self.self_attention(x, x, x, need_weights=False, attn_mask=causal_mask, is_causal=True)[0]
        x += residue

        residue = x

        x = self.ln2(x)
        x = self.feed_forward(x)
        x += residue
        return x