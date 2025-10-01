import pickle
import torch

from model import ModelArgs, Transformer
from tokenizer import CharwiseTokenizer

class GPT():

    @staticmethod
    def build(ckpt_dir: str) -> "GPT":
        """
        Build a GPT instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.

        Returns:
            GPT: An instance of the GPT class with the loaded model and tokenizer.
        """
        if torch.cuda.is_available():
            DEVICE = "cuda"
        elif (torch.backends.mps.is_available()):
            DEVICE = "mps"
        print(f"Using device {DEVICE}")

        model_args = ModelArgs()

        model = Transformer(model_args).to(DEVICE)
        model.load_state_dict(torch.load(f"{ckpt_dir}/state_dict_model.pt"), strict=True)

        with open(f"{ckpt_dir}/tokenizer.pkl", 'rb') as f:
            tokenizer = pickle.load(f)

        return GPT(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: CharwiseTokenizer):
        self.model = model
        self.tokenizer = tokenizer
