from typing import Literal

import torch

from tokenizer import CharwiseTokenizer

class Dataloader():

    def __init__(self, corpus: str, tokenizer: CharwiseTokenizer):
        self.data = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)

        n = int(0.9 * len(self.data)) # first 90% will be train, rest val
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def get_batch(
            self,
            split: Literal['train', 'val'],
            batch_size: int,
            block_size: int
    ):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data

        ix = torch.randint(len(data) - block_size, (batch_size,))

        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y