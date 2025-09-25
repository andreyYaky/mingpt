import pickle
import torch

from model import ModelArgs, Transformer

DEVICE = "cpu"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif (torch.backends.mps.is_available()):
    DEVICE = "mps"
print(f"Using device {DEVICE}")

params = ModelArgs()
model = Transformer(params).to(DEVICE)
model.load_state_dict(torch.load("./data/state_dict_model.pt"), strict=True)
print(model)

with open("./ckpt/tokenizer.pkl", 'rb') as file:
    tokenizer = pickle.load(file)

# generate from the model
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
out = model.generate(context, 1000)[0].tolist()
print(tokenizer.decode(out))