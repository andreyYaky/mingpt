import json
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

        with open(f"{ckpt_dir}/params.json", 'rb') as f:
            params = json.loads(f.read())
        model_args = ModelArgs(**params)

        model = Transformer(model_args).to(DEVICE)
        model.load_state_dict(torch.load(f"{ckpt_dir}/state_dict_model.pt"), strict=True)

        with open(f"{ckpt_dir}/tokenizer.pkl", 'rb') as f:
            tokenizer = pickle.load(f)

        return GPT(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: CharwiseTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def save(self, ckpt_dir: str):
        """
        Save a GPT instance by saving the model checkpoint file and tokenizer.

        Args:
            ckpt_dir (str): Path to the directory to save checkpoint file and tokenizer.
        """
        with open(f"{ckpt_dir}/params.json", 'w') as f:
            f.write(json.dumps(self.model.params.__dict__))
        torch.save(self.model.state_dict(), f"{ckpt_dir}/state_dict_model.pt")

        with open(f"{ckpt_dir}/tokenizer.pkl", 'wb') as file:
            pickle.dump(self.tokenizer, file)