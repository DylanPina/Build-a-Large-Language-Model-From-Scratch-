import torch
from torch import Tensor


class SelfAttention_v1(torch.nn.Module):
    """
    A simple self-attention module for processing input sequences.
    """

    def __init__(self, d_in: int, d_out: int) -> None:
        """
        Initializes the self-attention module.

        Args:
            d_in (int): Dimensionality of the input features.
            d_out (int): Dimensionality of the output features.
        """
        super().__init__()
        # Learnable weight matrices for query, key, and value projections.
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the forward pass for self-attention.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, d_in).

        Returns:
            Tensor: Contextualized output tensor of shape (batch_size, seq_length, d_out).
        """
        # Compute the key, query, and value matrices.
        keys = x @ self.W_key  # Shape: (batch_size, seq_length, d_out)
        queries = x @ self.W_query  # Shape: (batch_size, seq_length, d_out)
        values = x @ self.W_value  # Shape: (batch_size, seq_length, d_out)

        # Compute attention scores by taking the dot product of queries and transposed keys.
        attn_scores = queries @ keys.T  # Shape: (batch_size, seq_length, seq_length)

        # Scale the attention scores and normalize them with softmax.
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )  # Shape: (batch_size, seq_length, seq_length)

        # Compute the context vector as a weighted sum of the value vectors.
        context_vec = attn_weights @ values  # Shape: (batch_size, seq_length, d_out)

        return context_vec
