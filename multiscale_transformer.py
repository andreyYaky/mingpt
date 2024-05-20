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
        self.position_embedding_table = nn.Embedding(block_size * patch_size, n_embd)
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

    def __init__(self,
                 vocab_size,
                 block_size,
                 patch_size,
                 d_global,
                 n_head_global,
                 n_layer_global,
                 d_local,
                 n_head_local,
                 n_layer_local,
                 dropout,
                 device):
        super().__init__()
        self.block_size = block_size
        self.patch_size = patch_size
        self.device = device

        self.d_global = d_global
        self.n_head_global = n_head_global
        self.n_layer_global = n_layer_global

        self.d_local = d_local
        self.n_head_local = n_head_local
        self.n_layer_local = n_layer_local

        # TODO: learned padding embedding
        self.pad = 0

        # global model across patches
        self.global_model = DecoderTransformer(vocab_size=vocab_size,
                                               block_size=block_size // patch_size,
                                               patch_size=patch_size, # dim is P * D_G
                                               n_embd=d_global,
                                               n_head=n_head_global,
                                               n_layer=n_layer_global,
                                               dropout=dropout,
                                               device=device)
        # linear projection from d_global to d_local
        self.gl_projection = nn.Identity()
        if self.d_global != self.d_local:        
            self.gl_projection = nn.Linear(self.d_global, self.d_local)
        # local model within patches
        self.local_model = DecoderTransformer(vocab_size=vocab_size,
                                              block_size=patch_size,
                                              patch_size=1, # dim is just D_L
                                              n_embd=d_local,
                                              n_head=n_head_local,
                                              n_layer=n_layer_local,
                                              dropout=dropout,
                                              device=device)
        # lm head after local model
        self.lm_head = nn.Linear(self.d_local, vocab_size)

    def forward_inference(self, x: torch.Tensor, global_output_reshaped: torch.Tensor):
        #print(f"x.shape: {x.shape}")
        last_completed = ((x.shape[1] - 1) // self.patch_size) * self.patch_size
        #print(f"last_completed: {last_completed}")
        # get last patch to feed into local model (can be incomplete)
        x_current_patch = x[:, last_completed:]

        # ONLY forward through global model if patch completed
        if x_current_patch.shape[1] == 1 or global_output_reshaped is None:
            padding_global = x.new_empty((x.shape[0], self.patch_size)).fill_(self.pad)
            bytes_global = torch.cat((padding_global, x[:, :last_completed]), -1)
            
            # prepare global bytes in T/P patch embeddings
            global_bytes_embedded = self.global_model.embed(bytes_global)
            # (B, T * P, E) -> (B, T, P * E)
            global_in = global_bytes_embedded.unfold(1, self.patch_size, self.patch_size)
            b, t, e, p = global_in.shape
            global_in = global_in.reshape((b, t, p * e))
            #print(f"global_in.shape: {global_in.shape}")

            # forward through global model
            global_output = self.global_model(global_in)
            # reshape to patches of embeddings
            # (B, T, P * E) -> (B * T, P, E)
            global_output_reshaped = global_output.view((b * t, p, e))[[-1],:,:]

        # NOTE: Also need to know which patch we're on within the global model
        # forward through local model until patch generated
        bytes_local = x_current_patch
        #print(f"bytes_local: {bytes_local}")
        local_bytes_embedded = self.local_model.embed(bytes_local)
        b, p, e = local_bytes_embedded.shape
        
        # add projected global model output to local embeddings
        local_in = self.gl_projection(global_output_reshaped[:,:p,:]) + local_bytes_embedded
        local_output = self.local_model(local_in)
        # (B * T, P, E) == (B, T * P, E) since T == 1 (one patch)

        # inference-time mini-optimization:
        # only forward the lm_head on the very last position
        logits = self.lm_head(local_output[:,[-1],:]) # note: using list [-1] to preserve the time dim
        return logits, global_output_reshaped

    def forward(self, x: torch.Tensor, targets=None):        
        # prepare global input
        # TODO: pad global after forwarding?
        padding_global = x.new_empty((x.shape[0], self.patch_size)).fill_(self.pad)
        bytes_global = torch.cat((padding_global, x[:, : -self.patch_size]), -1)
        
        # prepare global bytes in T/P patch embeddings
        global_bytes_embedded = self.global_model.embed(bytes_global)
        # (B, T * P, E) -> (B, T, P * E)
        global_in = global_bytes_embedded.unfold(1, self.patch_size, self.patch_size)
        b, t, e, p = global_in.shape
        global_in = global_in.reshape((b, t, p * e))

        # forward through global model
        global_output = self.global_model(global_in)
        # reshape to patches of embeddings
        # (B, T, P * E) -> (B * T, P, E)
        global_output_reshaped = global_output.view((b * t, p, e))


        # prepare local input
        # rearrange(bytes, "b (t p) -> (b t) p", p=self.patch_size)
        # NOTE: unfold() loses data if T % P != 0
        bytes_local = x.unfold(-1, self.patch_size, self.patch_size)
        b, t, p = bytes_local.shape
        bytes_local = bytes_local.view((b * t, p))
        # Don't pad final model

        local_bytes_embedded = self.local_model.embed(bytes_local)
        # add projected global model output to local embeddings
        e = self.d_local
        local_in = self.gl_projection(global_output_reshaped) + local_bytes_embedded
        local_output = self.local_model(local_in)
        
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
        # crop idx to last block_size tokens
        # (B, T) where T = min(T, block_size)
        idx_cond = idx[:, -self.block_size:]
        global_output = None

        for _ in range(max_new_tokens):
            # get predictions (B, T, C)
            pred, global_output = self.forward_inference(idx_cond, global_output)
            # focus on last time step (B, C)
            logits = pred[:, -1,:]
            # softmax over C to get probability distribution
            probs = nn.functional.softmax(logits, dim=-1)
            # sample from distribution (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sample
            # (B, T) + (B, 1) = (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)
            idx_cond = torch.cat((idx_cond, idx_next), dim=1)

            if idx_cond.shape[1] > self.block_size:
                # drop last patch in idx
                idx_cond = idx_cond[:, self.patch_size:]
                
        return idx