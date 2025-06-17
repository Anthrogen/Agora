from src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
import torch


class SequenceTokenizer():
    """
    A Tokenizer and Padder for amino acid sequences.

    This class does NOT handle BOS, EOS, or MASK tokens.
    """

    def __init__(self, word_length):

        # Get sequence tokens (amino acids)
        sequence = {name: member.value for name, member in SEQUENCE_TOKENS.__members__.items()}

        # Get the highest sequence token value to avoid conflicts
        max_seq_value = max(sequence.values()) if sequence else -1

        # Add special tokens with non-conflicting values
        special = {name: member.value + max_seq_value + 1 
                  for name, member in SPECIAL_TOKENS.__members__.items()}

        self.mapping = {**sequence, **special}
        # self.inverse = {v: k for k, v in self.mapping.items()}

        self.mapping['X'] = self.mapping["UNK"]
        self.word_length = word_length

    def tokenize(self, observation):

        ret = torch.ones(self.word_length, dtype=torch.long) * self.mapping['PAD']
        for col, char in enumerate(observation):
            ret[col] = self.mapping[char]

        assert torch.max(ret) < len(self.mapping), "Sequence Tokenization failed!"
        assert torch.min(ret) >= 0, "Sequence Tokenization failed!"

        return ret