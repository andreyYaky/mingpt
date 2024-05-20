import gpt
import universal_transformer
import multiscale_transformer
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

block_size = 256

'''model = gpt.MinGPT(vocab_size=65,
                   block_size=256,
                   n_embd=384,
                   n_head=6,
                   n_layer=6,
                   dropout=0.2,
                   device=DEVICE).to(DEVICE)'''

'''model = universal_transformer.UT(vocab_size=65,
                                 block_size=256,
                                 n_embd=384,
                                 n_head=6,
                                 dropout=0.2,
                                 threshold=0.99,
                                 max_steps=10,
                                 device=DEVICE).to(DEVICE)'''

model = multiscale_transformer.MultiscaleDecoder(vocab_size=65,
                          block_size=256,
                          patch_size=4,
                          d_global=384,
                          n_head_global=6,
                          n_layer_global=6,
                          d_local=384,
                          n_head_local=6,
                          n_layer_local=6,
                          dropout=0.1,
                          device=DEVICE).to(DEVICE)

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

epochs = 1000#5000
batch_size = 64
eval_interval = 50#0
eval_iters = 10#0

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
        #losses, ponder_costs = estimate_loss(eval_iters, batch_size, block_size, DEVICE)
        print(f"Step {n}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        #print(f"Step {n}: train loss {losses['train']:.4f}, ponder cost {ponder_costs['train']:.4f}; val loss {losses['val']:.4f}, ponder cost {ponder_costs['val']:.4f}")

torch.save(model.state_dict(), "./data/state_dict_model.pt")