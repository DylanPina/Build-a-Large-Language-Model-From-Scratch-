import torch
import torch.nn as nn


class SelfAttention_v2(nn.Module):
    """
    Implements a basic self-attention mechanism. This time we are using nn.Linear for
    optimizied weight initialization and perform matrix mutliplication when the bias units
    are disabled
    """

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False) -> None:
        """
        Initializes the self-attention layer with linear transformations for query, key, and value.

        Args:
            d_in (int): Input feature dimensionality.
            d_out (int): Output feature dimensionality for the query, key, and value.
            qkv_bias (bool): Whether to include bias terms in the linear transformations.
        """
        super().__init__()
        self.W_query: nn.Linear = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key: nn.Linear = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value: nn.Linear = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the self-attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_in),
                        where `batch_size` is the batch size, `seq_length` is the number of input tokens,
                        and `d_in` is the input feature dimensionality.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_out),
                    where `d_out` is the output feature dimensionality.
        """
        # Compute keys, queries, and values
        keys: torch.Tensor = self.W_key(x)  # Shape: (batch_size, seq_length, d_out)
        queries: torch.Tensor = self.W_query(
            x
        )  # Shape: (batch_size, seq_length, d_out)
        values: torch.Tensor = self.W_value(x)  # Shape: (batch_size, seq_length, d_out)

        # Compute attention scores
        attn_scores: torch.Tensor = (
            queries @ keys.T
        )  # Shape: (batch_size, seq_length, seq_length)

        # Apply softmax to get attention weights
        attn_weights: torch.Tensor = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )  # Shape: (batch_size, seq_length, seq_length)

        # Compute context vectors
        context_vec: torch.Tensor = (
            attn_weights @ values
        )  # Shape: (batch_size, seq_length, d_out)

        return context_vec
