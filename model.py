import math
from dataclasses import dataclass
from typing import Optional, Tuple

import einops
import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class ModelArgs:
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    vocab_size: int = -1 # defined later by tokenizer
    norm_eps: float = 1e-5

    max_seq_len: int = 256


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    i = torch.arange(0, dim, 2)[: (dim // 2)].float()
    freqs = 1.0 / (theta ** (i / dim))

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    freqs_cis = torch.polar(
        abs=torch.ones_like(freqs),
        angle=freqs
    )
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(
        einops.rearrange(xq, "b n_heads t (h_dim1 h_dim2) -> b n_heads t h_dim1 h_dim2", h_dim2=2)
    )
    xk_ = torch.view_as_complex(
        einops.rearrange(xk, "b n_heads t (h_dim1 h_dim2) -> b n_heads t h_dim1 h_dim2", h_dim2=2)
    )

    # reshape freqs_cis for broadcast
    _bsz, n_heads, seqlen, h_dim1 = xq_.shape
    freqs_cis = einops.repeat(
        freqs_cis, "t h_dim1 -> b n_heads t h_dim1",
        b=_bsz, n_heads=n_heads
    )
    xq_ = xq_ * freqs_cis
    xk_ = xk_ * freqs_cis

    # reshape as real tensors
    xq_out = einops.rearrange(
        torch.view_as_real(xq_ * freqs_cis),
        "b n_heads t h_dim1 h_dim2 -> b n_heads t (h_dim1 h_dim2)"
    )
    xk_out = einops.rearrange(
        torch.view_as_real(xq_ * freqs_cis),
        "b n_heads t h_dim1 h_dim2 -> b n_heads t (h_dim1 h_dim2)"
    )
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):

    def __init__(self, args: ModelArgs):
        """
        Initialize an Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            head_dim (int): Dimension size of each attention head.
            wq (nn.Linear): Linear transformation for queries.
            wk (nn.Linear): Linear transformation for keys.
            wv (nn.Linear): Linear transformation for values.
            wo (nn.Linear): Linear transformation for output.

        """
        super().__init__()
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor]
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = einops.rearrange(xq, "b t (n_heads h_dim) -> b n_heads t h_dim", h_dim=self.head_dim)
        xk = einops.rearrange(xk, "b t (n_heads h_dim) -> b n_heads t h_dim", h_dim=self.head_dim)
        xv = einops.rearrange(xv, "b t (n_heads h_dim) -> b n_heads t h_dim", h_dim=self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        scores = xq @ einops.rearrange(xk, "b n_heads t h_dim -> b n_heads h_dim t") / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores, dim=-1)

        out = scores @ xv
        out = einops.rearrange(out, "b n_heads t h_dim -> b t (n_heads h_dim)")
        return self.wo(out)


class FeedForward(nn.Module):

    def __init__(
            self,
            dim: int,
            hidden_dim: int
        ):
        """
        Initialize a FeedForward module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            w1 (nn.Linear): Linear transformation for first layer.
            w_gate (nn.Linear): Linear transformation for gate layer.
            w2 (nn.Linear): Linear transformation for second layer.

        """
        super().__init__()
        # reduce hidden dimension by a factor of 2/3 to keep parameter count 
        # and computation amount consistent with vanilla transformer
        hidden_dim = int(2 * hidden_dim / 3)

        # SwiGLU (Swish Gated Linear Unit)
        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False
        )
        self.w_gate = nn.Linear(
            dim,
            hidden_dim,
            bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the feedforward module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after feedforward.

        """
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))


class TransformerBlock(nn.Module):

    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            layer_id (int): Identifier for the layer.
            attention_norm (nn.LayerNorm): Layer normalization before attention.
            self_attention (Attention): Attention module.
            ffn_norm (nn.LayerNorm): Layer normalization before feed forward.
            feed_forward (FeedForward): FeedForward module.

        """
        super().__init__()
        self.layer_id = layer_id
        
        self.attention_norm = nn.LayerNorm(args.dim, args.norm_eps)
        self.self_attention = Attention(args)

        self.ffn_norm = nn.LayerNorm(args.dim, args.norm_eps)
        self.feed_forward = FeedForward(args.dim, 4 * args.dim)
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor]
    ):
        """
        Forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        residue = x

        x = self.attention_norm(x)
        x = self.self_attention(x, freqs_cis, mask)
        x += residue

        residue = x

        x = self.ffn_norm(x)
        x = self.feed_forward(x)
        x += residue
        return x


class Transformer(nn.Module):

    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            token_embeddings (torch.nn.Embedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            ln_f (torch.nn.LayerNorm): Layer normalization before final output.
            lm_head (torch.nn.Linear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.block_size = params.max_seq_len

        self.token_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.ln_f = nn.LayerNorm(params.dim, eps=params.norm_eps)
        self.lm_head = nn.Linear(params.dim, params.vocab_size)

        self.freqs_cis = precompute_freqs_cis(
            dim=params.dim // params.n_heads,
            end=params.max_seq_len
        )

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through the Transformer.

        Args:
            tokens (torch.Tensor): Input token indices.
            targets (torch.Tensor, optional): Target token indices for training.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _bsz, seqlen = tokens.shape

        x = self.token_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(x.device)
        freqs_cis = self.freqs_cis[: seqlen]

        subsequent_mask = torch.full(
            [seqlen, seqlen], fill_value=float("-inf"), device=x.device
        )
        subsequent_mask = torch.triu(subsequent_mask, diagonal=1)

        for layer in self.layers:
            x = layer(x, freqs_cis, subsequent_mask)
        x = self.ln_f(x)

        if targets is not None:
            # calculate loss if targets are provided
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int
    ):
        """
        Generate text sequence using the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices for the current context.
            max_new_tokens (int): Number of new tokens to generate.

        Returns:
            torch.Tensor: Output token indices for generated text sequence.

        """
        for _ in range(max_new_tokens):
            # crop tokens to last block_size tokens
            tokens_cond = tokens[:, -self.block_size:]

            # get predicted logits from forward pass
            pred, _ = self(tokens_cond)
            # focus on logits for next token
            logits = pred[:, -1,:]

            # softmax to get probability distribution
            probs = F.softmax(logits, dim=-1)
            # sample from distribution and append next token
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens