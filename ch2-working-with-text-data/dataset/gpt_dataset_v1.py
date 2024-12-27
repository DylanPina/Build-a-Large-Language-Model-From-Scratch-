import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from typing import Any


class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer: Any, max_length: int, stride: int):
        """
        Initializes the dataset.

        Args:
        - txt (str): The input text data.
        - tokenizer: A tokenizer instance to convert text into token IDs.
        - max_length (int): The maximum length of each input sequence.
        - stride (int): The stride (step size) used to slide the window over the tokens.

        Attributes:
        - input_ids (list): A list of input sequences as tensors.
        - target_ids (list): A list of target sequences (shifted input sequences) as tensors.
        """
        self.input_ids = []  # To store input sequences
        self.target_ids = []  # To store target sequences

        # Tokenize the entire input text
        token_ids = tokenizer.encode(txt)

        # Create input and target sequences using a sliding window approach
        for i in range(0, len(token_ids) - max_length, stride):
            # Extract a chunk of tokens for the input
            input_chunk = token_ids[i : i + max_length]
            # Create the corresponding target chunk (input shifted by 1)
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            # Convert chunks to tensors and store them
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
        - int: Number of input sequences in the dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single data sample.

        Args:
        - idx (int): The index of the desired sample.

        Returns:
        - tuple: A tuple containing the input tensor and the target tensor for the specified index.
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    """
    Creates a PyTorch DataLoader for tokenized text data, designed for training language models.

    Args:
    - txt (str): The input text data to tokenize and split into sequences.
    - batch_size (int, optional): The number of samples per batch. Default is 4.
    - max_length (int, optional): The maximum length of each input sequence. Default is 256.
    - stride (int, optional): The stride (step size) for sliding the window over the tokenized data. Default is 128.
    - shuffle (bool, optional): Whether to shuffle the data at the start of each epoch. Default is True.
    - drop_last (bool, optional): Whether to drop the last incomplete batch if the dataset size isn't divisible by the batch size. Default is True.
    - num_workers (int, optional): The number of worker processes for data loading. Default is 0.

    Returns:
    - DataLoader: A PyTorch DataLoader instance that yields batches of input and target tensors.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader
