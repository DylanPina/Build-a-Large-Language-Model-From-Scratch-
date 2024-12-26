import re


class SimpleTokenizerV2:
    """
    A tokenizer class for encoding text into integer IDs and decoding them back into text.

    This version introduces support for unknown tokens, represented as "<|unk|>",
    ensuring graceful handling of out-of-vocabulary tokens during encoding.
    """

    def __init__(self, vocab: dict[str, int]):
        """
        Initializes the tokenizer with a given vocabulary.

        Args:
            vocab (dict): A dictionary where keys are strings (tokens) and
                          values are their corresponding integer IDs.
        """
        self.str_to_int = vocab  # Maps strings (tokens) to integer IDs.
        self.int_to_str = {
            i: s for s, i in vocab.items()
        }  # Maps integer IDs back to strings (tokens).

    def encode(self, text: str):
        """
        Encodes the input text into a list of integer IDs.

        This method splits the input text into tokens using regular expressions,
        ensures proper handling of punctuation and whitespace, and replaces tokens
        not found in the vocabulary with the unknown token "<|unk|>".

        Args:
            text (str): The input text to encode.

        Returns:
            list: A list of integer IDs representing the tokens in the text.
        """
        # Step 1: Split the text into tokens using regex.
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]  # Remove empty or whitespace-only tokens.

        # Step 2: Replace out-of-vocabulary tokens with the unknown token "<|unk|>".
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed  # Handle unknown tokens.
        ]

        # Step 3: Map each token to its corresponding integer ID.
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]):
        """
        Decodes a list of integer IDs back into text.

        This method reconstructs the text from integer IDs, ensuring proper
        formatting for punctuation and handling of unknown tokens.

        Args:
            ids (list): A list of integer IDs to decode.

        Returns:
            str: The decoded text.
        """
        # Step 1: Convert each ID back to its corresponding token.
        text = " ".join([self.int_to_str[i] for i in ids])

        # Step 2: Remove unnecessary whitespace before punctuation marks.
        text = re.sub(
            r'\s+([,.:;?!"()\'])', r"\1", text
        )  # Handle punctuation formatting.
        return text
