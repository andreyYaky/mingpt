from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from attention import CausalSelfAttentionBlock

@dataclass
class ModelArgs:
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    vocab_size: int = 65#-1  # should be defined later by tokenizer
    norm_eps: float = 1e-5

    max_seq_len: int = 256


class MinGPT(nn.Module):

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.block_size = params.max_seq_len

        self.token_embedding_table = nn.Embedding(params.vocab_size, params.dim)
        self.position_embedding_table = nn.Embedding(params.max_seq_len, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(CausalSelfAttentionBlock(params.dim, params.n_heads, dropout=0.2))

        self.ln_f = nn.LayerNorm(params.dim, eps=params.norm_eps)
        self.lm_head = nn.Linear(params.dim, params.vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        tok_emb = self.token_embedding_table(x) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=x.device)) # (T, C)
        x = tok_emb + pos_emb
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        # idx: (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to last block_size tokens
            # (B, T) where T = min(T, block_size)
            idx_cond = idx[:, -self.block_size:]
            # get predictions
            # (B, block_size, C)
            pred, _ = self(idx_cond)
            # focus on last time step
            # (B, C)
            logits = pred[:, -1,:]
            # softmax over C to get probability distribution
            # (B, C) still
            probs = F.softmax(logits, dim=-1)
            # sample from distribution
            # (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sample
            # (B, T) + (B, 1) = (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx