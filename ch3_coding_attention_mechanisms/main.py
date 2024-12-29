import torch
from self_attention_v2 import (
    SelfAttention_v2,
)
from self_attention_v1 import (
    SelfAttention_v1,
)
from casual_attention import (
    CausalAttention,
)
from multi_head_attention import (
    MultiHeadAttentionWrapper,
)

if __name__ == "__main__":
    torch.manual_seed(123)

    # Define input tensor representing token embeddings
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^0)
            [0.55, 0.87, 0.66],  # journey  (x^1)
            [0.57, 0.85, 0.64],  # starts   (x^2)
            [0.22, 0.58, 0.33],  # with     (x^3)
            [0.77, 0.25, 0.10],  # one      (x^4)
            [0.05, 0.80, 0.55],  # step     (x^5)
        ]
    )

    x_1 = inputs[1]  # Select the second token embedding (x^1)
    d_in = inputs.shape[1]  # Dimensionality of input embeddings
    d_out = 2  # Dimensionality of output vectors

    # Initialize weight matrices for query, key, and value computations
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    # Compute query, key, and value vectors for the second token embedding
    query_1 = x_1 @ W_query
    key_1 = x_1 @ W_key
    value_1 = x_1 @ W_value
    print("query_1:", query_1)

    # Compute keys and values for all token embeddings
    keys = inputs @ W_key
    values = inputs @ W_value
    print("keys.shape:", keys.shape)  # Shape of key matrix
    print("values.shape", values.shape)  # Shape of value matrix

    # Compute attention score between token x^1 and itself
    keys_1 = keys[1]  # Key vector for token x^1
    attn_score_11 = query_1.dot(keys_1)  # Dot product between query and key
    print("attention_score_11:", attn_score_11)

    # Compute attention scores between token x^1 and all other tokens
    attn_scores_1 = query_1 @ keys.T  # Dot product of query with all keys
    print("attn_scores_1:", attn_scores_1)

    # Compute attention weights using softmax normalization
    d_k = keys.shape[-1]  # Dimensionality of key vectors
    attn_weights_1 = torch.softmax(
        attn_scores_1 / d_k**0.5, dim=-1
    )  # Scaled dot-product attention
    print("attn_weights_1:", attn_weights_1)

    # Compute context vector as a weighted sum of values
    context_vec_1 = attn_weights_1 @ values
    print("context_vec_1:", context_vec_1)

    # Apply Self-Attention version 1
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print("SelfAttentionV1:\n", sa_v1(inputs))

    # Apply Self-Attention version 2
    torch.manual_seed(789)  # Reset seed for consistency
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print("SelfAttentionV2:\n", sa_v2(inputs))

    # Compute queries, keys, and attention scores for Self-Attention v2
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    print("SelfAttentionV2 attention weights:\n", attn_weights)

    # Generate a lower triangular mask for causal attention
    context_length = attn_scores.shape[0]  # Number of tokens
    mask_simple = torch.tril(
        torch.ones(context_length, context_length)
    )  # Lower triangular matrix
    print("mask_simple:\n", mask_simple)

    # Apply the mask to attention scores
    masked_simple = attn_scores * mask_simple
    print("masked_simple:\n", masked_simple)

    # Normalize the masked attention scores to sum to 1
    row_sums = masked_simple.sum(dim=-1, keepdim=True)  # Row-wise sum for normalization
    masked_simple_norm = masked_simple / row_sums  # Renormalize scores
    print("masked_simple_norm:\n", masked_simple_norm)

    # Create a mask that sets upper triangular elements to negative infinity
    mask = torch.triu(
        torch.ones(context_length, context_length), diagonal=1
    )  # Upper triangular mask
    masked = attn_scores.masked_fill(
        mask.bool(), -float("inf")
    )  # Replace upper triangular elements
    print("mask:\n", mask)
    print("masked:\n", masked)

    # Compute final attention weights after masking
    attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
    print("attn_weights:\n", attn_weights)

    # Applying dropout
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5)
    print("attn_weights with dropout:\n", dropout(attn_weights))

    # Causal attention
    batch = torch.stack((inputs, inputs), dim=0)  # Shape: (2, num_tokens, d_in)
    context_length = batch.shape[1]  # Number of tokens
    ca = CausalAttention(d_in, d_out, context_length, 0.5)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape)

    # Multihead attention
    context_length = batch.shape[1]  # Number of tokens
    d_in, d_out = 3, 2
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0, num_heads=2)
    context_vecs = mha(batch)

    print(context_vecs)
    print("MHA context_vecs.shape:", context_vecs.shape)
