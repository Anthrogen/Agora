from src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
import torch


#TODO: get all tokenizers to operate over a whole batch at a time.
# right now batch sizes are small (e.g. 4) so it doesn't matter, but for scale-up it might.

class SequenceTokenizer():
    """
    A Tokenizer and Padder for amino acid sequences.

    This class does NOT handle MASK tokens.
    """

    def __init__(self, full_length):

        # Get sequence tokens (amino acids)
        sequence = {name: member.value for name, member in SEQUENCE_TOKENS.__members__.items()}

        # Get the highest sequence token value to avoid conflicts
        max_seq_value = max(sequence.values()) if sequence else -1

        # Add special tokens with non-conflicting values
        special = {name: member.value + max_seq_value + 1 
                  for name, member in SPECIAL_TOKENS.__members__.items()}

        self.mapping = {**sequence, **special}

        self.mapping['X'] = self.mapping["UNK"]
        self.full_length = full_length

    def tokenize(self, observation):

        # Convert characters in the observation to token indices
        seq_tokens = [self.mapping[char] for char in observation]
        seq_tokens = seq_tokens[:self.full_length-2] # -2 for BOS and EOS

        # Add BOS and EOS tokens
        content = torch.tensor([self.mapping["BOS"], *seq_tokens, self.mapping["EOS"]], dtype=torch.long,)
        content_len = content.numel()

        # Add Padding
        ret = torch.full((self.full_length,), self.mapping["PAD"], dtype=torch.long)
        ret[:content_len] = content[:content_len]

        # Create beospad mask: 1s for BOS/EOS/PAD, 0s for real content
        beospad = torch.ones(self.full_length, dtype=torch.bool)
        if len(seq_tokens) > 0:  # Only if there's actual sequence content
            beospad[1:1+len(seq_tokens)] = 0  # Mark real content as 0, keep BOS/EOS/PAD as 1

        # Sanity checks – ensure indices are within vocabulary bounds.
        assert torch.max(ret) < len(self.mapping), "Sequence Tokenization failed!"
        assert torch.min(ret) >= 0, "Sequence Tokenization failed!"
        assert ret.numel() == self.full_length, "Sequence Tokenization failed!"

        return ret, beospad
    

class CoordinatesTokenizer():
    def __init__(self, full_length):
        self.full_length = full_length

    def tokenize(self, coords):
        device = coords.device
        dtype = coords.dtype
        M, H = coords.shape[0], coords.shape[1]

        coords = coords[:self.full_length-2]       # reserve space for BOS & EOS

        # Add BOS and EOS tokens
        bos_coord = torch.zeros((1, H, 3), dtype=dtype, device=device)  # BOS token
        eos_coord = torch.zeros((1, H, 3), dtype=dtype, device=device)  # EOS token
        content_coords = torch.cat([bos_coord, coords, eos_coord], dim=0)  # [C, H, 3]
        content_len = content_coords.shape[0]

        # Add Padding
        pad_coords = torch.zeros((self.full_length - content_len, H, 3), dtype=dtype, device=device)
        padded_coords = torch.cat([content_coords, pad_coords], dim=0)  # [self.full_length, H, 3]

        # Create beospad mask: 1s for BOS/EOS/PAD, 0s for real content
        beospad = torch.ones(self.full_length, dtype=torch.bool)
        actual_content_len = coords.shape[0]  # Length before BOS/EOS were added
        if actual_content_len > 0:  # Only if there's actual coordinate content
            beospad[1:1+actual_content_len] = 0  # Mark real content as 0, keep BOS/EOS/PAD as 1

        # Sanity Checks
        assert padded_coords.shape == (self.full_length, H, 3), "Structure padding length mismatch!"

        return padded_coords, beospad
        
class StructureTokenizer():
    """
    A Tokenizer and Padder for structure sequences.

    This class does NOT handle BOS, EOS, or MASK tokens.
    """
    def __init__(self, full_length, fsq_encoder, fsq_output_max=(7*5*5*5*5)-1, fsq_num_atoms=3, reapply_bos_eos_pad=True):

        # TODO: Calculate fsq_output_max automatically from the fsq_encoder object
        self.fsq_encoder = fsq_encoder
        self.fsq_output_max = fsq_output_max
        self.fsq_num_atoms = fsq_num_atoms

        self.full_length = full_length
        self.mapping = {name: member.value + self.fsq_output_max + 1 for name, member in SPECIAL_TOKENS.__members__.items()}

        self.coordinates_tokenizer = CoordinatesTokenizer(full_length)
        self.reapply_bos_eos_pad = reapply_bos_eos_pad

    def tokenize(self, coords):
        assert self.fsq_encoder is not None, "FSQ encoder is not set!"
        assert coords.shape[0] == self.full_length


        # Coordinates tensor: [M, H, 3]
        # M is at most full_length.  H is number of atoms per residue.

        # Pass through coordinates tokenizer to get padded coords and beospad.
        padded_coords, coords_beospad = self.coordinates_tokenizer.tokenize(coords)
        bos_position = 0
        eos_position = coords_beospad.numel() - coords_beospad.sum() + 1 

        with torch.no_grad():
            # FSQ encoder expects [B, L, 3, 3]; if coords contain 4 atoms use first 3
            coords_for_fsq = padded_coords[:, :self.fsq_num_atoms, :]
            coords_for_fsq = coords_for_fsq.unsqueeze(0).to(next(self.fsq_encoder.parameters()).device)
            struct_tokens_full = self.fsq_encoder.encode_to_tokens(coords_for_fsq)  # [1, L]
            struct_tokens_full = struct_tokens_full.squeeze(0).cpu().long()         # [L]

        # ------------------------------------------------------------------ #
        # Postprocess FSQ output -- replace BOS, EOS, PAD positions with special tokens
        # ------------------------------------------------------------------ #
        padded_struct_tokens = struct_tokens_full.clone()
        assert padded_struct_tokens.shape[0] == self.full_length, "Structure padding length mismatch!"


        if self.reapply_bos_eos_pad:        
            # Apply BOS, EOS, and PAD tokens to structure tokens, copying from corresponding special tokens of the coords.
            padded_struct_tokens[coords_beospad] = self.mapping['PAD']
            padded_struct_tokens[bos_position] = self.mapping['BOS']
            padded_struct_tokens[eos_position] = self.mapping['EOS']
            struct_boespad = coords_beospad.clone()
        else:
            struct_boespad = torch.zeros(self.full_length, dtype=torch.bool)

        # ------------------------------------------------------------------ #
        # Sanity checks                                                    #
        # ------------------------------------------------------------------ #
        assert torch.max(padded_struct_tokens) < self.fsq_output_max + len(self.mapping), "Structure Tokenization failed!"
        assert torch.min(padded_struct_tokens) >= 0, "Structure Tokenization failed!"
        assert padded_coords.shape[0] == self.full_length, "Structure padding length mismatch!"

        return padded_coords, coords_beospad, padded_struct_tokens, struct_boespad
