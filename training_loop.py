import pickle
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import torch

from dataloader import Dataloader
from model import ModelArgs, Transformer
from tokenizer import CharwiseTokenizer

DEVICE = "cpu"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()):
    DEVICE = "mps"
print(f"Using device {DEVICE}")

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('./data/input.txt', 'r', encoding='utf-8') as f:
    corpus = f.read()

tokenizer = CharwiseTokenizer(corpus)
dataloader = Dataloader(corpus, tokenizer)

params = ModelArgs()
model = Transformer(params).to(DEVICE)
print(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

@torch.no_grad()
def estimate_loss(eval_iters, batch_size, block_size):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            inputs, targets = dataloader.get_batch(split, batch_size, block_size)
            preds, loss = model(inputs.to(DEVICE), targets.to(DEVICE))

            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

epochs = 1000
batch_size = 64
block_size = params.max_seq_len
eval_interval = 50
eval_iters = 10

loss_data = []
model.train()

for n in tqdm(range(epochs)):
    # inputs: (B, L)
    # targets: (B, L) inputs shifted left 1
    inputs, targets = dataloader.get_batch('train', batch_size, block_size)
    preds, loss = model(inputs.to(DEVICE), targets.to(DEVICE))

    # backward
    optimizer.zero_grad()
    loss.backward()
    # update
    optimizer.step()

    if n % eval_interval == 0 or n == epochs - 1:
        losses = estimate_loss(eval_iters, batch_size, block_size)
        tqdm.write(f"Step {n}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        loss_data.append([
            (n + 1) * batch_size * block_size, # number of training tokens
            losses['val'].item() # validation loss
        ])

# save model and tokenizer after training
ckpt_dir = "./ckpt"
torch.save(model.state_dict(), f"{ckpt_dir}/state_dict_model.pt")
with open(f"{ckpt_dir}/tokenizer.pkl", 'wb') as f:
    pickle.dump(tokenizer, f)

# save loss curve as csv and plot
loss_df = pd.DataFrame(loss_data, columns=['Training Tokens', "Validation Loss"])
loss_df.to_csv(f"{ckpt_dir}/loss_curve.csv", index=False)

plt.plot(loss_df['Training Tokens'], loss_df['Validation Loss'])
plt.title('Loss Curve')
plt.xlabel('Training Tokens')
plt.ylabel('Validation Loss')
plt.savefig(f"{ckpt_dir}/loss_curve.png")