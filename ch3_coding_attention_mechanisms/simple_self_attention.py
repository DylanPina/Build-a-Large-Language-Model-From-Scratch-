import torch


if __name__ == "__main__":
    """
    Simple self-attention mechanism without trainable weights
    """
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^0)
            [0.55, 0.87, 0.66],  # journey  (x^1)
            [0.57, 0.85, 0.64],  # starts   (x^2)
            [0.22, 0.58, 0.33],  # with     (x^3)
            [0.77, 0.25, 0.10],  # one      (x^4)
            [0.05, 0.80, 0.55],
        ]  # step     (x^5)
    )

    # Compute attention scores based on query
    query = inputs[1]
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(query, x_i)
    print("x1 - Attention scores:\n", attn_scores_2)

    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print("x1 - Attention weights\n", attn_weights_2)

    context_vec_2 = torch.zeros(query.shape)
    for attention_weight, x_i in zip(attn_weights_2, inputs):
        context_vec_2 += attention_weight * x_i
    print("x1 - Context vector\n", context_vec_2)

    attention_scores = inputs @ inputs.T
    print("\nAttention scores:\n", attention_scores)

    attention_weights = torch.softmax(attention_scores, dim=-1)
    print("Attention weights:\n", attention_weights)

    all_context_vectors = attention_weights @ inputs
    print("Context vectors:\n", all_context_vectors)
