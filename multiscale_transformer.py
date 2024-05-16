from attention import CausalSelfAttentionBlock
import torch
from torch import nn
from torch.nn import functional as F

class DecoderTransformer(nn.Module):

    def __init__(self, vocab_size, block_size, patch_size, n_embd, n_head, n_layer, dropout, device):
        super().__init__()
        self.block_size = block_size
        self.device = device

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[CausalSelfAttentionBlock(block_size, n_embd * patch_size, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd * patch_size)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape

        tok_emb = self.token_embedding_table(x) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = tok_emb + pos_emb

        return x

    def forward(self, x: torch.Tensor, targets=None) -> torch.Tensor:
        # x: already embedded
        x = self.blocks(x)
        x = self.ln_f(x)

        return x

# MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers
# https://arxiv.org/pdf/2305.07185
class MultiscaleDecoder(nn.Module):

    def __init__(self, vocab_size, block_size, patch_size, n_embd, n_head, n_layer, dropout, device):
        super().__init__()
        self.block_size = block_size
        self.patch_size = patch_size
        self.device = device

        self.d_global = n_embd
        self.d_local = n_embd

        # TODO: learned padding embedding
        self.pad = 0

        # TODO: patch embedder
        # maps byte sequence of length T to patch sequence of length T/P

        # global model across patches
        self.global_model = DecoderTransformer(vocab_size=vocab_size,
                                               block_size=block_size,
                                               patch_size=patch_size, # dim is P * D_G
                                               n_embd=self.d_global,
                                               n_head=n_head,
                                               n_layer=n_layer,
                                               dropout=dropout,
                                               device=device)
        # local model within patches
        self.local_model = DecoderTransformer(vocab_size=vocab_size,
                                              block_size=block_size,
                                              patch_size=1, # dim is just D_L
                                              n_embd=self.d_local,
                                              n_head=n_head,
                                              n_layer=n_layer,
                                              dropout=dropout,
                                              device=device)
        # lm head after local model
        self.lm_head = nn.Linear(self.d_local, vocab_size)

    def prepare_input(self, bytes: torch.Tensor):
        #print(f"bytes.shape: {bytes.shape}")
        # pad global bytes with P tokens
        padding_global = bytes.new_empty((bytes.shape[0], self.patch_size)).fill_(self.pad)
        bytes_global = torch.cat((padding_global, bytes[:, : -self.patch_size]), -1)

        # pad local bytes with 1 token per patch
        # rearrange(bytes, "b (t p) -> (b t) p", p=self.patch_size)
        # NOTE: unfold() loses data if T % P != 0
        bytes_input = bytes.unfold(-1, self.patch_size, self.patch_size)
        b, t, p = bytes_input.shape
        bytes_input = bytes_input.view((b * t, p))
        #print(f"bytes_input.shape: {bytes_input.shape}")
        
        padding_local = bytes_input.new_empty((bytes_input.shape[0], 1)).fill_(self.pad)
        bytes_local = torch.cat((padding_local, bytes_input[:, :-1]), -1)

        return bytes_global, bytes_local

    def forward(self, x: torch.Tensor, targets=None):
        bytes_global, bytes_local = self.prepare_input(x)
        # prepare global bytes in T/P patch embeddings
        global_bytes_embedded = self.global_model.embed(bytes_global)
        #print(f"global_bytes_embedded.shape: {global_bytes_embedded.shape}")
        # rearrange(global_bytes_embedded, "b (t p) e -> b t (p e)", p=self.patch_size)
        global_in = global_bytes_embedded.unfold(1, self.patch_size, self.patch_size)
        #print(f"global_bytes_embedded.unfold: {global_in.shape}")
        b, t, e, p = global_in.shape
        global_in = global_in.reshape((b, t, p * e))
        #print(f"global_in.shape: {global_in.shape}")

        # forward through global model and reshape to patches of embeddings
        global_output = self.global_model(global_in)
        #print(f"global_output.shape: {global_output.shape}")
        # rearrange(global_output, "b t (p e) -> (b t) p e", p=self.patch_size)
        global_output_reshaped = global_output.view((b * t, p, e))
        #print(f"global_output_reshaped.shape: {global_output_reshaped.shape}")

        # TODO: w_GL projection matrix from D_G to D_L
        #local_in = wGL(global_output_reshaped) + local_bytes_embedded

        local_bytes_embedded = self.local_model.embed(bytes_local)
        #print(f"local_bytes_embedded.shape: {local_bytes_embedded.shape}")
        # add global model output to local embeddings
        local_in = local_bytes_embedded + global_output_reshaped
        #print(f"local_in.shape: {local_in.shape}")
        local_output = self.local_model(local_in)
        #print(f"local_output.shape: {local_output.shape}")
        # (B * T, P, E) -> (B, T * P, E)
        x = local_output.view((b, t * p, e))

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
            probs = nn.functional.softmax(logits, dim=-1)
            # sample from distribution
            # (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sample
            # (B, T) + (B, 1) = (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx