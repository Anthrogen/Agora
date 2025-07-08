from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS, SS8_TOKENS, SASA_TOKENS
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

        # Create beospank mask: 1s for BOS/EOS/PAD/UNK, 0s for real content
        beospank = torch.ones(self.full_length, dtype=torch.bool)
        if len(seq_tokens) > 0:  # Only if there's actual sequence content
            beospank[1:1+len(seq_tokens)] = 0  # Mark real content as 0, keep BOS/EOS/PAD as 1
            
        # Also mark UNK tokens as 1s
        beospank[ret == self.mapping["UNK"]] = 1

        # Sanity checks – ensure indices are within vocabulary bounds.
        assert torch.max(ret) < len(self.mapping), "Sequence Tokenization failed!"
        assert torch.min(ret) >= 0, "Sequence Tokenization failed!"
        assert ret.numel() == self.full_length, "Sequence Tokenization failed!"

        return ret, beospank.bool()
    

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

        # Create beospank mask: 1s for BOS/EOS/PAD, 0s for real content
        beospank = torch.ones(self.full_length, dtype=torch.bool)
        actual_content_len = coords.shape[0]  # Length before BOS/EOS were added
        if actual_content_len > 0:  # Only if there's actual coordinate content
            beospank[1:1+actual_content_len] = 0  # Mark real content as 0, keep BOS/EOS/PAD as 1
            

        # Sanity Checks
        assert padded_coords.shape == (self.full_length, H, 3), "Structure padding length mismatch!"

        return padded_coords, beospank.bool()


class StructureTokenizer():
    """
    A Tokenizer and Padder for structure sequences.

    This class does NOT handle BOS, EOS, or MASK tokens.
    """
    def __init__(self, full_length, fsq_encoder, fsq_output_max=(7*5*5*5*5)-1, fsq_num_atoms=3, reapply_bos_eos_pad=True):

        # TODO: Calculate fsq_output_max automatically from the fsq_encoder object
        self.fsq_encoder = fsq_encoder
        # TODO: get this automatically from FsqEncoder object model configuration, which could be a feature of the model object.
        self.fsq_output_max = fsq_output_max
        self.fsq_num_atoms = fsq_num_atoms

        self.full_length = full_length
        self.mapping = {name: member.value + self.fsq_output_max + 1 for name, member in SPECIAL_TOKENS.__members__.items()}
        #self.mapping = {name: member.value + self.fsq_output_max for name, member in SPECIAL_TOKENS.__members__.items()} #off by one?

        self.coordinates_tokenizer = CoordinatesTokenizer(full_length)
        self.reapply_bos_eos_pad = reapply_bos_eos_pad

    def tokenize(self, coords): # coords should be UNTOKENIZED.
        assert self.fsq_encoder is not None, "FSQ encoder is not set!"
        
        # Coordinates tensor: [M, H, 3]
        # M is at most full_length.  H is number of atoms per residue.

        # Pass through coordinates tokenizer to get padded coords and beospank.
        padded_coords, coords_beospank = self.coordinates_tokenizer.tokenize(coords)
        bos_position = 0
        eos_position = coords_beospank.numel() - coords_beospank.sum() + 1 

        with torch.no_grad():
            # FSQ encoder expects [B, L, 3, 3]; if coords contain 4 atoms use first 3
            coords_for_fsq = padded_coords[:, :self.fsq_num_atoms, :]
            coords_for_fsq = coords_for_fsq.unsqueeze(0).to(next(self.fsq_encoder.parameters()).device)
            struct_tokens_full = self.fsq_encoder.encode_to_tokens(coords_for_fsq)  # [1, L]
            struct_tokens_full = struct_tokens_full.squeeze(0).cpu().long().squeeze(-1)         # [L]

        # ------------------------------------------------------------------ #
        # Postprocess FSQ output -- replace BOS, EOS, PAD positions with special tokens
        # ------------------------------------------------------------------ #
        padded_struct_tokens = struct_tokens_full.clone()
        assert padded_struct_tokens.shape[0] == self.full_length, "Structure padding length mismatch!"

        if self.reapply_bos_eos_pad:        
            # Apply BOS, EOS, and PAD tokens to structure tokens, copying from corresponding special tokens of the coords.
            padded_struct_tokens[coords_beospank] = self.mapping['PAD']
            padded_struct_tokens[bos_position] = self.mapping['BOS']
            padded_struct_tokens[eos_position] = self.mapping['EOS']
            struct_beospank = coords_beospank.clone()
        else:
            struct_beospank = torch.zeros(self.full_length, dtype=torch.bool)

        # ------------------------------------------------------------------ #
        # Sanity checks                                                    #
        # ------------------------------------------------------------------ #
        assert torch.max(padded_struct_tokens) < self.fsq_output_max + 1 + len(self.mapping), f"Structure Tokenization failed! Max token = {torch.max(padded_struct_tokens)}"
        assert torch.min(padded_struct_tokens) >= 0, f"Structure Tokenization failed! Min token = {torch.min(padded_struct_tokens)}"
        assert padded_coords.shape[0] == self.full_length, "Structure padding length mismatch!"

        return padded_coords, coords_beospank.bool(), padded_struct_tokens, struct_beospank.bool()


class SS8Tokenizer():
    """
    A Tokenizer and Padder for secondary structure 8-class tokens.

    This class does NOT handle MASK tokens.
    """

    def __init__(self, full_length):
        # Get SS8 tokens (secondary structure classes)
        ss8 = {name: member.value for name, member in SS8_TOKENS.__members__.items()}

        # Get the highest SS8 token value to avoid conflicts
        max_ss8_value = max(ss8.values()) if ss8 else -1

        # Add special tokens with non-conflicting values
        special = {name: member.value + max_ss8_value + 1 
                  for name, member in SPECIAL_TOKENS.__members__.items()}

        self.mapping = {**ss8, **special}
        self.full_length = full_length

    def tokenize(self, observation):
        """
        Args:
            observation: List of SS8 labels (or None values), e.g., ['H', 'E', None, 'C']
        Returns:
            padded_ss8: Tensor of SS8 tokens
            ss8_beospank: Boolean mask for BOS/EOS/PAD/None positions
        """
        # Convert SS8 labels to token indices, mapping None to PAD
        ss8_tokens = []
        for label in observation:
            if label is None: ss8_tokens.append(self.mapping["PAD"])
            else: ss8_tokens.append(self.mapping[label])
        
        # Truncate if too long (reserve space for BOS/EOS)
        ss8_tokens = ss8_tokens[:self.full_length-2]

        # Add BOS and EOS tokens
        content = torch.tensor([self.mapping["BOS"], *ss8_tokens, self.mapping["EOS"]], dtype=torch.long)
        content_len = content.numel()

        # Add Padding
        padded_ss8 = torch.full((self.full_length,), self.mapping["PAD"], dtype=torch.long)
        padded_ss8[:content_len] = content[:content_len]

        # Create beospank mask: 1s for BOS/EOS/PAD/None, 0s for real content
        ss8_beospank = torch.ones(self.full_length, dtype=torch.bool)
        # Mark real content (non-None) as 0
        for i, token in enumerate(ss8_tokens):
            if token != self.mapping["PAD"]:  # Real SS8 content
                ss8_beospank[1 + i] = 0  # +1 to account for BOS token

        # Sanity checks
        assert torch.max(padded_ss8) < len(self.mapping), "SS8 Tokenization failed!"
        assert torch.min(padded_ss8) >= 0, "SS8 Tokenization failed!"
        assert padded_ss8.numel() == self.full_length, "SS8 Tokenization failed!"

        return padded_ss8, ss8_beospank.bool()


class SASATokenizer():
    """
    A Tokenizer and Padder for SASA tokens.

    This class does NOT handle MASK tokens.
    """

    def __init__(self, full_length):
        # Get SASA tokens (binned values)
        sasa = {name: member.value for name, member in SASA_TOKENS.__members__.items()}

        # Get the highest SASA token value to avoid conflicts
        max_sasa_value = max(sasa.values()) if sasa else -1

        # Add special tokens with non-conflicting values
        special = {name: member.value + max_sasa_value + 1 
                  for name, member in SPECIAL_TOKENS.__members__.items()}

        self.mapping = {**sasa, **special}
        self.full_length = full_length

    def _bin_sasa_value(self, value):
        """Convert continuous SASA value to bin token using optimal quantile-based boundaries."""
        if value is None: return self.mapping["PAD"]
        
        # Optimal bin boundaries based on quantile analysis of 22,185 real SASA values
        # Each bin contains approximately equal numbers of residues (~1,386 each)
        if value < 0.14: return self.mapping["BIN_0"]
        elif value < 2.09: return self.mapping["BIN_1"]
        elif value < 6.49: return self.mapping["BIN_2"]
        elif value < 12.69: return self.mapping["BIN_3"]
        elif value < 20.23: return self.mapping["BIN_4"]
        elif value < 29.03: return self.mapping["BIN_5"]
        elif value < 38.17: return self.mapping["BIN_6"]
        elif value < 47.43: return self.mapping["BIN_7"]
        elif value < 56.62: return self.mapping["BIN_8"]
        elif value < 65.50: return self.mapping["BIN_9"]
        elif value < 76.08: return self.mapping["BIN_10"]
        elif value < 86.71: return self.mapping["BIN_11"]
        elif value < 99.48: return self.mapping["BIN_12"]
        elif value < 115.35: return self.mapping["BIN_13"]
        elif value < 137.27: return self.mapping["BIN_14"]
        else: return self.mapping["BIN_15"]
        
    def tokenize(self, observation):
        """
        Args:
            observation: List of SASA values (or None values), e.g., [45.2, None, 12.8, 156.7]
        Returns:
            padded_sasa: Tensor of SASA tokens
            sasa_beospank: Boolean mask for BOS/EOS/PAD/None positions
        """
        # Convert SASA values to binned token indices
        sasa_tokens = [self._bin_sasa_value(value) for value in observation]
        
        # Truncate if too long (reserve space for BOS/EOS)
        sasa_tokens = sasa_tokens[:self.full_length-2]

        # Add BOS and EOS tokens
        content = torch.tensor([self.mapping["BOS"], *sasa_tokens, self.mapping["EOS"]], dtype=torch.long)
        content_len = content.numel()

        # Add Padding
        padded_sasa = torch.full((self.full_length,), self.mapping["PAD"], dtype=torch.long)
        padded_sasa[:content_len] = content[:content_len]

        # Create beospank mask: 1s for BOS/EOS/PAD/None, 0s for real content
        sasa_beospank = torch.ones(self.full_length, dtype=torch.bool)
        # Mark real content (non-None) as 0
        for i, token in enumerate(sasa_tokens):
            if token != self.mapping["PAD"]:  # Real SASA content
                sasa_beospank[1 + i] = 0  # +1 to account for BOS token

        # Sanity checks
        assert torch.max(padded_sasa) < len(self.mapping), "SASA Tokenization failed!"
        assert torch.min(padded_sasa) >= 0, "SASA Tokenization failed!"
        assert padded_sasa.numel() == self.full_length, "SASA Tokenization failed!"

        return padded_sasa, sasa_beospank.bool()
        