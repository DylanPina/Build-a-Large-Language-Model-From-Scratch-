import torch
from torch import nn
from torch import Tensor


class CausalAttention(nn.Module):
    """
    Implements a causal attention mechanism with support for masking to ensure
    that each position in the sequence can only attend to previous positions.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
    ) -> None:
        """
        Initializes the CausalAttention module.

        Args:
            d_in (int): Dimensionality of the input features.
            d_out (int): Dimensionality of the output features.
            context_length (int): Length of the sequence to compute attention over.
            dropout (float): Dropout probability for the attention weights.
            qkv_bias (bool): Whether to include a bias term in query, key, and value projections.
        """
        super().__init__()
        self.d_out = d_out
        # Linear layers for computing query, key, and value projections
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Dropout layer for attention weights
        self.dropout = nn.Dropout(dropout)
        # Causal mask to ensure no information is leaked from future tokens
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the causal attention for a given input sequence.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_tokens, d_out) representing the
            context vectors after applying causal attention.
        """
        _, num_tokens, _ = x.shape
        # Compute keys, queries, and values
        keys = self.W_key(x)  # Shape: (batch_size, num_tokens, d_out)
        queries = self.W_query(x)  # Shape: (batch_size, num_tokens, d_out)
        values = self.W_value(x)  # Shape: (batch_size, num_tokens, d_out)

        # Compute attention scores
        attn_scores = queries @ keys.transpose(
            1, 2
        )  # Shape: (batch_size, num_tokens, num_tokens)
        # Apply causal mask to prevent future tokens from being attended to
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # Compute attention weights using softmax
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)

        # Compute context vectors by applying attention weights to values
        context_vec = attn_weights @ values  # Shape: (batch_size, num_tokens, d_out)
        return context_vec
