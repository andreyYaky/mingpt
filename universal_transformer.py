from attention import CausalSelfAttentionBlock
import torch
from torch import nn
from torch.nn import functional as F

class UT(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, n_head, dropout, threshold, max_steps, device):
        super().__init__()
        self.block_size = block_size
        self.device = device

        # UT hyper-params
        self.threshold = threshold
        self.max_steps = max_steps

        # UT meta-layers
        self.probs_linear = nn.Sequential(nn.Linear(n_embd, 1), nn.Sigmoid())

        # main pipeline
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.attn_ffwd_block = CausalSelfAttentionBlock(block_size, n_embd, n_head, dropout)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x):
        B, T = x.shape

        tok_emb = self.token_embedding_table(x) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = tok_emb # pos_emb added in recurrent block
        
        # Adaptive Computation Time (ACT) (Graves, 2016) meta-variables:
        should_continue = True
        step = 0
        halting_probability = torch.zeros((B, T), device=self.device)
        remainders = torch.zeros((B, T), device=self.device)
        n_updates = torch.zeros((B, T), device=self.device)

        # extra_output = (ponder_times, remainders)
        # act_loss = act_loss_weight * tf.reduce_mean(ponder_times + remainders)
        # Once the per-symbol recurrent block halts, its state is simply copied to the next step
        while should_continue:
            # probability calculated based on state
            p = self.probs_linear(x).squeeze()
            # mask for inputs which haven't halted yet
            still_running = torch.less(halting_probability, 1.0)
            # mask for inputs which halted at this step
            new_halted = torch.greater(halting_probability + p * still_running, self.threshold) * still_running
            # mask of input which haven't halted including this step
            still_running = torch.less_equal(halting_probability + p * still_running, self.threshold) * still_running

            # add the halting probability for this step to halting prob for unhalted inputs
            halting_probability += p * still_running
            # compute remainders for newly halted inputs
            remainders += new_halted * (1 - halting_probability)
            # add remainders to the newly halted inputs
            halting_probability += new_halted * remainders

            # increment n_updates for all running inputs
            n_updates += still_running + new_halted
            # compute weight to be applied to new state and output:
            # 0 if already halted
            # p when unhalted
            # remainders when halted this step
            update_weights = p * still_running + remainders * new_halted
            # (B, T) -> (B, T, n_embd)
            update_weights = update_weights.unsqueeze(-1).expand(-1,-1,384)

            # apply transformation from blocks
            transformed_x = x + pos_emb
            transformed_x = self.attn_ffwd_block(transformed_x)
            transformed_x = self.ln_f(transformed_x)
            # interpolate transformed and previous states for non-halted inputs
            #x = transformed_x * update_weights + x * (1 - update_weights)
            x = torch.lerp(x, transformed_x, update_weights)
            step += 1
            # break if fully halted or reached max steps
            should_continue = torch.any(
                torch.logical_and(
                    torch.less(halting_probability, self.threshold),
                    torch.less(n_updates, self.max_steps)))

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