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

class StructureTokenizer():
    """
    A Tokenizer and Padder for structure sequences.

    This class does NOT handle BOS, EOS, or MASK tokens.
    """
    def __init__(self, word_length, fsq_encoder, fsq_output_max=(7*5*5*5*5)-1):

        # TODO: Calculate fsq_output_max automatically from the fsq_encoder object
        self.fsq_encoder = fsq_encoder
        self.fsq_output_max = fsq_output_max

        self.word_length = word_length
        self.special = {name: member.value + self.fsq_output_max + 1 for name, member in SPECIAL_TOKENS.__members__.items()}

    def tokenize(self, observation):

        coords = observation
        M = coords.shape[0]
        H = coords.shape[1]
        """
        Observation will appear as (seq, coords, len)

        coords is (M, 4, 3) OR (M, 3, 3), depending upon CB.

        M will be at most word_length.

        From this class, we return both coords (with zero-padding) AND tokens (with <PAD> padding)
        """

        # Zero-pad coords to word_length (which is max_len - 2 = 2046)
        num_pad = self.word_length - M
        padded_coords = torch.cat([coords, torch.zeros(num_pad, H, 3)], dim=0)
        
        # The FSQ encoder was trained with sequences that have BOS/EOS tokens
        coords_with_special = torch.cat([
            torch.zeros((1, H, 3)),  # BOS padding
            padded_coords,
            torch.zeros((1, H, 3))   # EOS padding
        ], dim=0)  # Now shape is [2048, H, 3]

        # Get structure embeddings from frozen FSQ encoder
        with torch.no_grad():
            # FSQ encoder expects [B, L, 3, 3] - only use N, CA, C atoms
            coords_for_fsq = coords_with_special[:, :3, :] if coords_with_special.shape[1] == 4 else coords_with_special
            # Add batch dimension and move to encoder's device
            device = next(self.fsq_encoder.parameters()).device
            coords_for_fsq = coords_for_fsq.unsqueeze(0).to(device)  # [1, 2048, 3, 3]
            struct_tokens = self.fsq_encoder.encode_to_tokens(coords_for_fsq).squeeze(0).squeeze(-1)  # [2048]
            # Move back to original device
            struct_tokens = struct_tokens.cpu().long()

        # Extract tokens corresponding to actual content (skip BOS/EOS positions)
        struct_tokens_content = struct_tokens[1:-1]  # Remove BOS/EOS tokens, back to [2046]
        
        # Only use the non-padded portion for the final tokens
        struct_tokens_valid = struct_tokens_content[:M]
        padded_struct_tokens = torch.cat([struct_tokens_valid, torch.ones(num_pad, dtype=torch.long) * self.special['PAD']])

        assert torch.max(padded_struct_tokens) < self.fsq_output_max + len(self.special), "Structure Tokenization failed!"
        assert torch.min(padded_struct_tokens) >= 0, "Structure Tokenization failed!"

        return padded_coords, padded_struct_tokens