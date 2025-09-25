from typing import List

class CharwiseTokenizer():

    def __init__(
            self,
            corpus: str
        ):
        """
        Initialize a character-level tokenizer based on the training text corpus.

        Args:
            corpus (str): Training text corpus.

        Attributes:
            vocab (List[str]): Sorted list of unique characters in the training corpus.
            vocab_size (int): Number of characters in the vocab.
            stoi (dict[str, int]): Mapping from characters to token IDs
            itos (dict[int, str]): Mapping from token IDs to characters.
        """
        self.vocab = sorted(list(set(corpus)))
        self.vocab_size = len(self.vocab)

        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(self.vocab) }
        self.itos = { i:ch for i,ch in enumerate(self.vocab) }

    def encode(self, s: str) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.

        Returns:
            List[int]: A list of token IDs.
        """
        return [self.stoi[c] for c in s]

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return ''.join([self.itos[i] for i in t])