import torch
from torch import nn
from casual_attention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        """
        Initializes the MultiHeadAttentionWrapper.

        Args:
            d_in (int): Input feature dimension.
            d_out (int): Output feature dimension for each head.
            context_length (int): Context length for causal attention.
            dropout (float): Dropout rate.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether to include bias in QKV projections.
        """
        super().__init__()

        # Create a list of `num_heads` instances of the CausalAttention module.
        self.heads: nn.ModuleList = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MultiHeadAttentionWrapper.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_in).

        Returns:
            torch.Tensor: Concatenated output from all attention heads,
            with shape (batch_size, seq_length, num_heads * d_out).
        """
        # Apply each attention head to the input `x` independently and concatenate their outputs
        return torch.cat([head(x) for head in self.heads], dim=-1)
