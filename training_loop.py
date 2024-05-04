import gpt
import universal_transformer
import data_loader
import torch
from torch import nn
import torchmetrics

DEVICE = "cpu"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()):
    DEVICE = "mps"
print(f"Using device {DEVICE}")

vocab_size = 65
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

'''model = gpt.MinGPT(vocab_size,
                   block_size,
                   n_embd,
                   n_head,
                   n_layer,
                   dropout,
                   DEVICE).to(DEVICE)'''
model = universal_transformer.UT(vocab_size,
                                 block_size,
                                 n_embd,
                                 n_head,
                                 dropout,
                                 threshold=0.99,
                                 max_steps=10,
                                 device=DEVICE).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

@torch.no_grad()
def estimate_loss(eval_iters, batch_size, block_size, device):
    out = {}
    out_p = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        ponder_costs = torch.zeros(eval_iters)
        for k in range(eval_iters):
            inputs, targets = data_loader.get_batch(split, batch_size, block_size, device)
            predictions, loss, ponder_cost = model(inputs, targets)

            losses[k] = loss
            ponder_costs[k] = ponder_cost
        out[split] = losses.mean()
        out_p[split] = ponder_costs.mean()
    model.train()
    return out, out_p

epochs = 5000
batch_size = 64
eval_interval = 500
eval_iters = 100

model.train()

for n in range(epochs):
    # inputs: (B, L)
    # targets: (B, L) inputs shifted left 1
    inputs, targets = data_loader.get_batch('train', batch_size, block_size, DEVICE)
    predictions, loss, _ = model(inputs, targets)

    # backward
    optimizer.zero_grad()
    loss.backward()
    # update
    optimizer.step()
    
    if n % eval_interval == 0 or n == epochs - 1:
        losses, ponder_costs = estimate_loss(eval_iters, batch_size, block_size, DEVICE)
        print(f"Step {n}: train loss {losses['train']:.4f}, ponder cost {ponder_costs['train']:.4f}; val loss {losses['val']:.4f}, ponder cost {ponder_costs['val']:.4f}")

torch.save(model.state_dict(), "./data/state_dict_model.pt")