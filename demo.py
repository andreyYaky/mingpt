import gpt
import universal_transformer
import multiscale_transformer
from data_loader import decode
import torch
from torch import nn

DEVICE = "cpu"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()):
    DEVICE = "mps"
print(f"Using device {DEVICE}")

'''model = gpt.MinGPT(vocab_size=65,
                   block_size=256,
                   n_embd=384,
                   n_head=6,
                   n_layer=6,
                   dropout=0.2,
                   device=DEVICE).to(DEVICE)'''
#model.load_state_dict(torch.load("./data/state_dict_10p8_model.pt"), strict=True)

'''model = universal_transformer.UT(vocab_size=65,
                                 block_size=256,
                                 n_embd=384,
                                 n_head=6,
                                 dropout=0.2,
                                 threshold=0.99,
                                 max_steps=10,
                                 device=DEVICE).to(DEVICE)'''
#model.load_state_dict(torch.load("./data/state_dict_UTmodel.pt"), strict=True)

model = multiscale_transformer.MultiscaleDecoder(vocab_size=65,
                          block_size=256,
                          patch_size=4,
                          n_embd=384,
                          n_head=2,#6,
                          n_layer=2,#6,
                          dropout=0.2,
                          device=DEVICE).to(DEVICE)
model.load_state_dict(torch.load("./data/state_dict_model.pt"), strict=True)
model.eval()

# generate from the model
#context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
context = torch.zeros((1, 4), dtype=torch.long, device=DEVICE)
out = model.generate(context, 1000)[0].tolist()
print(decode(out))