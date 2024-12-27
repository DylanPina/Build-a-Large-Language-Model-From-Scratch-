import tiktoken
from dataset.gpt_dataset_v1 import create_dataloader_v1


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    file_path = "data/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)
