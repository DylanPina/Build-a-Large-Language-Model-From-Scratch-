import re


class SimpleTokenizerV1:
    """
    A simple tokenizer for text processing.

    This class provides methods to encode text into a sequence of integers
    based on a predefined vocabulary and to decode sequences of integers back
    into text.
    """

    def __init__(self, vocab: dict[str, int]):
        """
        Initializes the tokenizer with a given vocabulary.

        Args:
            vocab (dict): A dictionary where keys are strings (tokens) and
                          values are their corresponding integer IDs.
        """
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """
        Encodes the input text into a list of integer IDs.

        This method splits the text into tokens using regular expressions, strips
        any whitespace, and maps each token to its corresponding integer ID from
        the vocabulary.

        Args:
            text (str): The input text to encode.

        Returns:
            list: A list of integer IDs representing the tokens in the text.
        """
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of integer IDs back into text.

        This method maps each integer ID back to its corresponding token and
        reconstructs the original text, ensuring proper formatting for punctuation.

        Args:
            ids (list): A list of integer IDs to decode.

        Returns:
            str: The decoded text.
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text
