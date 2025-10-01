import torch

from gpt import GPT

DEVICE = "cpu"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif (torch.backends.mps.is_available()):
    DEVICE = "mps"
print(f"Using device {DEVICE}")

ckpt_dir = "./ckpt"
gpt = GPT.build(ckpt_dir)
print(gpt.model)

# generate from the model
gpt.model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
out = gpt.model.generate(context, 1000)[0].tolist()
print(gpt.tokenizer.decode(out))