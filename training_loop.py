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

model = gpt.MinGPT(vocab_size,
                   block_size,
                   n_embd,
                   n_head,
                   n_layer,
                   dropout,
                   DEVICE).to(DEVICE)
'''model = universal_transformer.UT(vocab_size,
                                 block_size,
                                 n_embd,
                                 n_head,
                                 dropout,
                                 threshold=0.95,
                                 max_steps=10,
                                 device=DEVICE).to(DEVICE)'''

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

@torch.no_grad()
def estimate_loss(eval_iters, batch_size, block_size, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            inputs, targets = data_loader.get_batch(split, batch_size, block_size, device)
            predictions, loss = model(inputs, targets)

            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

epochs = 500
batch_size = 64
eval_interval = 50
eval_iters = 10

model.train()

for n in range(epochs):
    # inputs: (B, L)
    # targets: (B, L) inputs shifted left 1
    inputs, targets = data_loader.get_batch('train', batch_size, block_size, DEVICE)
    predictions, loss = model(inputs, targets)

    # backward
    optimizer.zero_grad()
    loss.backward()
    # update
    optimizer.step()
    
    if n % eval_interval == 0 or n == epochs - 1:
        losses = estimate_loss(eval_iters, batch_size, block_size, DEVICE)
        print(f"Step {n}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

torch.save(model.state_dict(), "./data/state_dict_model.pt")