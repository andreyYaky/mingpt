from tqdm import tqdm
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import torch

from dataloader import Dataloader
from model import ModelArgs, Transformer
from tokenizer import CharwiseTokenizer

from gpt import GPT

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

model_args = ModelArgs(
    vocab_size=tokenizer.vocab_size
)
model = Transformer(model_args).to(DEVICE)
print(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

@torch.no_grad()
def estimate_loss(
    split: Literal['train', 'val'],
    eval_iters: int,
    batch_size: int,
    block_size: int
):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        inputs, targets = dataloader.get_batch(split, batch_size, block_size)
        _preds, loss = model(inputs.to(DEVICE), targets.to(DEVICE))

        losses[k] = loss
    model.train()
    return losses.mean()

epochs = 1000
batch_size = 64
block_size = model_args.max_seq_len
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
        train_loss = estimate_loss('train', eval_iters, batch_size, block_size)
        val_loss = estimate_loss('val', eval_iters, batch_size, block_size)
        tqdm.write(f"Step {n}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        loss_data.append([
            (n + 1) * batch_size * block_size, # number of training tokens
            val_loss.item() # validation loss
        ])

# save model and tokenizer after training
ckpt_dir = "./ckpt"
GPT(model, tokenizer).save(ckpt_dir)

# save loss curve as csv and plot
loss_df = pd.DataFrame(loss_data, columns=['Training Tokens', "Validation Loss"])
loss_df.to_csv(f"{ckpt_dir}/loss_curve.csv", index=False)

plt.plot(loss_df['Training Tokens'], loss_df['Validation Loss'])
plt.title('Loss Curve')
plt.xlabel('Training Tokens')
plt.ylabel('Validation Loss')
plt.savefig(f"{ckpt_dir}/loss_curve.png")