import torch

from data_loader import decode
from gpt import ModelArgs, MinGPT

DEVICE = "cpu"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif (torch.backends.mps.is_available()):
    DEVICE = "mps"
print(f"Using device {DEVICE}")

params = ModelArgs()
model = MinGPT(params).to(DEVICE)
model.load_state_dict(torch.load("./data/state_dict_model.pt"), strict=True)
print(model)

# generate from the model
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
out = model.generate(context, 1000)[0].tolist()
print(decode(out))