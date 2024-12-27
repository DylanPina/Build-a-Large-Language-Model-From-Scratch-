import torch
from dataset.gpt_dataset_v1 import create_dataloader_v1


if __name__ == "__main__":
    torch.manual_seed(123)

    file_path = "data/the-verdict.txt"
    with open(file_path, "r") as f:
        raw_text = f.read()

    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    token_embeddings = token_embedding_layer(inputs)
    print("\nToken embeddings shape:\n", token_embeddings.shape)

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))

    input_embeddings = token_embeddings + pos_embeddings
    print("\nInput embeddings shape:\n", input_embeddings.shape)
