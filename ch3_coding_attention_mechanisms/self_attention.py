import torch

if __name__ == "__main__":
    torch.manual_seed(123)

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

    x_1 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2

    # Initialize attention weight matrices
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    # Compute query, key, and value vectors for x_1
    query_1 = x_1 @ W_query
    key_1 = x_1 @ W_key
    value_1 = x_1 @ W_value
    print("query_1:", query_1)

    # Compute keys and values
    keys = inputs @ W_key
    values = inputs @ W_value
    print("keys.shape:", keys.shape)
    print("values.shape", values.shape)

    # Compute the attention score w_11
    keys_1 = keys[1]
    attn_score_11 = query_1.dot(keys_1)
    print("attention_score_11:", attn_score_11)

    attn_scores_1 = query_1 @ keys.T
    print("attn_scores_1:", attn_scores_1)

    d_k = keys.shape[-1]  # Dimension of the keys
    attn_weights_1 = torch.softmax(attn_scores_1 / d_k**0.5, dim=-1)
    print("attn_weights_1:", attn_weights_1)

    context_vec_1 = attn_weights_1 @ values
    print("context_vec_1:", context_vec_1)
