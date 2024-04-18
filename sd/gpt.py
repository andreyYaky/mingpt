import torch
from torch import nn
from torch.nn import functional as F

class CausalSelfAttentionBlock(nn.Module):

    # communication + computation
    def __init__(self, block_size, n_embd, n_head, dropout):
        super().__init__()
        self.block_size = block_size
        
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
        causal_mask = nn.Transformer.generate_square_subsequent_mask(sz=self.block_size)
        x = self.self_attention(x, x, x, need_weights=False, attn_mask=causal_mask, is_causal=True)[0]
        x += residue

        residue = x

        x = self.ln2(x)
        x = self.feed_forward(x)
        x += residue
        return x

class MinGPT(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout, device):
        super().__init__()
        self.block_size = block_size
        self.device = device

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[CausalSelfAttentionBlock(block_size, n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        B, T = x.shape

        tok_emb = self.token_embedding_table(x) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        # idx: (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to last block_size tokens
            # (B, T) where T = min(T, block_size)
            idx_cond = idx[:, -self.block_size:]
            # get predictions
            # (B, block_size, C)
            pred = self(idx_cond)
            # focus on last time step
            # (B, C)
            logits = pred[:, -1,:]
            # softmax over C to get probability distribution
            # (B, C) still
            probs = nn.functional.softmax(logits, dim=-1)
            # sample from distribution
            # (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sample
            # (B, T) + (B, 1) = (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx