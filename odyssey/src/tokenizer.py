from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS, SS8_TOKENS, SASA_TOKENS, PLDDT_TOKENS, GLOBAL_ANNOTATION_TOKENS, PER_RESIDUE_ANNOTATION_TOKENS
import torch
from bisect import bisect_right
from tabulate import tabulate
from enum import IntEnum

#TODO: get all tokenizers to operate over a whole batch at a time.
# right now batch sizes are small (e.g. 4) so it doesn't matter, but for scale-up it might.

class CorruptionMode(IntEnum): MASK = 0; UNIFORM = 1

class Tokenizer(): pass

def unmask(actual_mask, beospank, min_unmasked, generator):
    """TODO: make this a method of the base class Tokenizer."""
    
    # Count positions that are NOT masked AND NOT beospank
    real_residues = (~actual_mask & ~beospank).sum()
    if real_residues < min_unmasked:
        num_to_unmask = min_unmasked - real_residues
        
        # Find positions that are currently masked but NOT beospank (candidates for unmasking)
        candidate_positions = (actual_mask & ~beospank).nonzero(as_tuple=False)
        if candidate_positions.numel() < num_to_unmask: raise ValueError(f"Need {min_unmasked} unmasked residues, but only {real_residues} residues in protein.")
        
        # Randomly select positions to unmask
        #TODO: use random number generator of the dataloader object.
        if generator is None:
            print("Warning: No generator provided to MaskedBatch. Using default generator.")
            perm = torch.randperm(candidate_positions.numel(), device=actual_mask.device)
        else: # Use generator without device parameter to avoid device mismatch
            perm = torch.randperm(candidate_positions.numel(), generator=generator).to(actual_mask.device)
            
        # Unmask these positions
        positions_to_unmask = candidate_positions[perm[:num_to_unmask]]
        actual_mask[positions_to_unmask] = False

    return actual_mask


class SequenceTokenizer(Tokenizer):
    """
    A Tokenizer and Padder for amino acid sequences.
    """
    def __init__(self, full_length, min_unmasked=0, generator=None, corruption_mode=CorruptionMode.MASK):
        # Get sequence tokens (amino acids)
        sequence = {name: member.value for name, member in SEQUENCE_TOKENS.__members__.items()}

        # Get the highest sequence token value to avoid conflicts
        self.max_seq_value = max(sequence.values()) if sequence else -1

        # Add special tokens with non-conflicting values
        special = {name: member.value + self.max_seq_value + 1 
                  for name, member in SPECIAL_TOKENS.__members__.items()}

        self.mapping = {**sequence, **special}
        # DO NOT include "X" (or any redundant symbols)in the reverse mapping.
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

        # Keep this line AFTER creation of self.mapping so that self.mapping can remain bijective.
        self.mapping['X'] = self.mapping["UNK"]
        self.full_length = full_length
        self.generator = generator
        self.min_unmasked = min_unmasked
        self.corruption_mode = corruption_mode

    def tokenize(self, observation, mask):
        """
        Args:
            observation: List of characters (amino acids), e.g., ['A', 'C', 'G', 'T']
            mask: Boolean mask for MASK positions
        Returns:
            padded_seq_tokens: Tensor of sequence tokens, shape [self.full_length]
            masked_seq_tokens: Tensor of sequence tokens, shape [self.full_length]  
            seq_beospank: Boolean mask for BOS/EOS/PAD/UNK positions
            seq_masks: Boolean mask for MASK positions
        """
        device = mask.device

        # Convert characters in the observation to token indices
        seq_tokens = [self.mapping[char] for char in observation]
        seq_tokens = seq_tokens[:self.full_length-2] # -2 for BOS and EOS

        # Add BOS and EOS tokens
        content = torch.tensor([self.mapping["BOS"], *seq_tokens, self.mapping["EOS"]], dtype=torch.long, device=device)
        content_len = content.numel()

        # Add Padding
        padding_seq_tokens = torch.full((self.full_length,), self.mapping["PAD"], dtype=torch.long, device=device)
        padding_seq_tokens[:content_len] = content[:content_len]

        # Create sequence beospank mask: 1s for BOS/EOS/PAD/UNK, 0s for real content
        seq_beospank = torch.ones(self.full_length, dtype=torch.bool, device=device)
        if len(seq_tokens) > 0:  # Only if there's actual sequence content
            seq_beospank[1:1+len(seq_tokens)] = 0  # Mark real content as 0, keep BOS/EOS/PAD as 1
            
        # Also mark UNK tokens as 1s
        seq_beospank[padding_seq_tokens == self.mapping["UNK"]] = 1
        # seq_pad_pos = (padding_seq_tokens == self.mapping["PAD"])

        seq_masks = mask & ~seq_beospank
        seq_masks = unmask(seq_masks, seq_beospank, self.min_unmasked, self.generator)
        masked_seq_tokens = self.corrupt(padding_seq_tokens, seq_masks)
        
        # Sanity checks – ensure indices are within vocabulary bounds.
        assert torch.max(padding_seq_tokens) < len(self.mapping), "Sequence Tokenization failed!"
        assert torch.min(padding_seq_tokens) >= 0, "Sequence Tokenization failed!"
        assert padding_seq_tokens.numel() == self.full_length, "Sequence Tokenization failed!"

        return padding_seq_tokens, masked_seq_tokens, seq_beospank.bool(), seq_masks.bool() # seq_pad_pos.bool()

    def corrupt(self, unmasked_data, masks):
        if self.corruption_mode == CorruptionMode.MASK:
            tensor_of_masks = torch.full((self.full_length,), self.mapping["MASK"], dtype=torch.long, device=unmasked_data.device)
            return torch.where(masks, tensor_of_masks, unmasked_data)

        elif self.corruption_mode == CorruptionMode.UNIFORM:
            uniform_content = torch.randint(0, self.max_seq_value + 1, (unmasked_data.shape[0],), dtype=torch.long, generator=self.generator).to(unmasked_data.device)
            return torch.where(masks, uniform_content, unmasked_data)
        
        raise ValueError(f"Unknown corruption mode: {self.corruption_mode}")

    def print_token(self, tok):
        return self.reverse_mapping[tok]


class CoordinatesTokenizer(Tokenizer):
    """
    A Tokenizer and Padder for structure coordinates.
    """
    def __init__(self, full_length, min_unmasked=0, generator=None):
        self.full_length = full_length
        self.generator = generator
        self.min_unmasked = min_unmasked

    def tokenize(self, coords, mask):
        """
        Args:
            coords: Tensor of coordinates, shape [M, H, 3]
            mask: Boolean mask for MASK positions
        Returns:
            padded_coords: Tensor of coordinates, shape [self.full_length, H, 3]
            masked_coords: Tensor of coordinates, shape [self.full_length, H, 3]
            coords_beospank: Boolean mask for BOS/EOS/PAD positions
            coords_masks: Boolean mask for MASK positions
        """
        device = mask.device
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

        # Create coordinates beospank mask: 1s for BOS/EOS/PAD, 0s for real content
        coords_beospank = torch.ones(self.full_length, dtype=torch.bool, device=device)
        actual_content_len = coords.shape[0]  # Length before BOS/EOS were added
        if actual_content_len > 0:  # Only if there's actual coordinate content
            coords_beospank[1:1+actual_content_len] = 0  # Mark real content as 0, keep BOS/EOS/PAD as 1
        # coords_pad_pos = torch.zeros_like(coords_beospank, dtype=torch.bool)
        # coords_pad_pos[actual_content_len+2:] = True

        # No UNK for coords.
        coords_masks = mask & ~coords_beospank
        coords_masks = unmask(coords_masks, coords_beospank, self.min_unmasked, self.generator)
        masked_coords = padded_coords.clone() * (~coords_masks).long().unsqueeze(1).unsqueeze(2).expand_as(padded_coords) 

        # Sanity Checks
        assert padded_coords.shape == (self.full_length, H, 3), "Structure padding length mismatch!"

        return padded_coords, masked_coords, coords_beospank.bool(), coords_masks.bool() # coords_pad_pos.bool()


class StructureTokenizer(Tokenizer):
    """
    A Tokenizer and Padder for structure sequences.
    """
    def __init__(self, full_length, autoencoder, min_unmasked=0, generator=None, corruption_mode=CorruptionMode.MASK):
        self.autoencoder = autoencoder
        self.generator = generator
        self.corruption_mode = corruption_mode

        if self.autoencoder is not None:
            self.fsq_output_max = autoencoder.codebook_size - 1
            assert self.fsq_output_max == 4375-1

            self.full_length = full_length
            self.mapping = {name: member.value + self.fsq_output_max + 1 for name, member in SPECIAL_TOKENS.__members__.items()}
            self.reverse_mapping = {v: f"{k}" for k, v in self.mapping.items()}
            self.reverse_mapping.update({i: i for i in range(self.fsq_output_max)})

            self.coordinates_tokenizer = CoordinatesTokenizer(full_length, min_unmasked=min_unmasked, generator=generator)

    def tokenize(self, coords, mask): # coords should be UNTOKENIZED.
        """
        Args:
            coords: Tensor of coordinates, shape [M, H, 3]
            mask: Boolean mask for MASK positions
        Returns:
            padded_coords: Tensor of coordinates, shape [self.full_length, H, 3]
            masked_coords: Tensor of coordinates, shape [self.full_length, H, 3]    
            coords_beospank: Boolean mask for BOS/EOS/PAD positions
            coords_masks: Boolean mask for MASK positions
            padded_struct_tokens: Tensor of structure tokens, shape [self.full_length]
            masked_struct_tokens: Tensor of structure tokens, shape [self.full_length]
            struct_beospank: Boolean mask for BOS/EOS/PAD positions
            struct_masks: Boolean mask for MASK positions
        """
        device = mask.device
        assert self.autoencoder is not None, "Autoencoder is not set!"
        
        # Pass through coordinates tokenizer to get padded coords and beospank.
        padded_coords, masked_coords, coords_beospank, coords_masks = self.coordinates_tokenizer.tokenize(coords, mask)
        bos_position = 0
        eos_position = coords_beospank.numel() - coords_beospank.sum() + 1 

        with torch.no_grad():
            # FSQ encoder expects [B, L, 3, 3]; if coords contain 4 atoms use first 3
            three_atom = masked_coords[:, :3, :]
            four_atom = masked_coords[:, :4, :]
            three_atom = three_atom.unsqueeze(0).to(next(self.autoencoder.parameters()).device)
            four_atom = four_atom.unsqueeze(0).to(next(self.autoencoder.parameters()).device)
            content_elements = (~coords_masks & ~coords_beospank).unsqueeze(0).to(next(self.autoencoder.parameters()).device)
            nonbeospank = ~coords_beospank.unsqueeze(0).to(next(self.autoencoder.parameters()).device)
            
            # For GA/RA cases, we need to pass the coordinates and mask
            if self.autoencoder.cfg.first_block_cfg.initials() in ("GA", "RA"):
                struct_tokens_full = self.autoencoder.encode_to_tokens(three_atom, coords=four_atom, content_elements=content_elements, nonbeospank=nonbeospank)
            else:
                struct_tokens_full = self.autoencoder.encode_to_tokens(three_atom, nonbeospank=nonbeospank)
            struct_tokens_full = struct_tokens_full.squeeze(0).long().squeeze(-1)         # [L] - must be long for embedding

        # ------------------------------------------------------------------ #
        # Postprocess FSQ output -- replace BOS, EOS, PAD positions with special tokens
        # ------------------------------------------------------------------ #
        padded_struct_tokens = struct_tokens_full.clone()
        assert padded_struct_tokens.shape[0] == self.full_length, "Structure padding length mismatch!"

        # Apply BOS, EOS, and PAD tokens to structure tokens, copying from corresponding special tokens of the coords.
        padded_struct_tokens[coords_beospank] = self.mapping['PAD']
        padded_struct_tokens[bos_position] = self.mapping['BOS']
        padded_struct_tokens[eos_position] = self.mapping['EOS']
        
        # Create beospank mask: 1s for BOS/EOS/PAD, 0s for real content
        struct_beospank = coords_beospank.clone()
        # struct_pad_pos = coords_pad_pos.clone()
        
        struct_masks = coords_masks.clone()
        masked_struct_tokens = self.corrupt(padded_struct_tokens, coords_masks)

        # ------------------------------------------------------------------ #
        # Sanity checks                                                    #
        # ------------------------------------------------------------------ #
        assert torch.max(padded_struct_tokens) < self.fsq_output_max + 1 + len(self.mapping), f"Structure Tokenization failed! Max token = {torch.max(padded_struct_tokens)}"
        assert torch.min(padded_struct_tokens) >= 0, f"Structure Tokenization failed! Min token = {torch.min(padded_struct_tokens)}"
        assert padded_coords.shape[0] == self.full_length, "Structure padding length mismatch!"

        return padded_coords, masked_coords, coords_beospank.bool(), coords_masks.bool(), padded_struct_tokens, masked_struct_tokens, struct_beospank.bool(), struct_masks.bool() # coords_pad_pos.bool(), struct_pad_pos.bool()

    def corrupt(self, unmasked_data, masks):
        if self.corruption_mode == CorruptionMode.MASK:
            tensor_of_masks = torch.full((self.full_length,), self.mapping["MASK"], dtype=torch.long, device=unmasked_data.device)
            return torch.where(masks, tensor_of_masks, unmasked_data)

        elif self.corruption_mode == CorruptionMode.UNIFORM:
            uniform_content = torch.randint(0, self.fsq_output_max + 1, (unmasked_data.shape[0],), dtype=torch.long, generator=self.generator).to(unmasked_data.device)
            return torch.where(masks, uniform_content, unmasked_data)
        
        raise ValueError(f"Unknown corruption mode: {self.corruption_mode}")

    def print_token(self, tok):
        return self.reverse_mapping[tok]


class SS8Tokenizer(Tokenizer):
    """
    A Tokenizer and Padder for secondary structure 8-class tokens.
    """
    def __init__(self, full_length, generator=None, corruption_mode=CorruptionMode.MASK):
        # Get SS8 tokens (secondary structure classes)
        ss8 = {name: member.value for name, member in SS8_TOKENS.__members__.items()}

        # Get the highest SS8 token value to avoid conflicts
        self.max_ss8_value = max(ss8.values()) if ss8 else -1

        # Add special tokens with non-conflicting values
        special = {name: member.value + self.max_ss8_value + 1 
                  for name, member in SPECIAL_TOKENS.__members__.items()}

        self.mapping = {**ss8, **special}
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        self.full_length = full_length
        self.generator = generator
        self.corruption_mode = corruption_mode

    def tokenize(self, observation, mask):
        """
        Args:
            observation: List of SS8 labels (or None values), e.g., ['H', 'E', None, 'C']
            mask: Boolean mask for MASK positions
        Returns:
            padded_ss8: Tensor of SS8 tokens
            masked_ss8: Tensor of SS8 tokens
            ss8_beospank: Boolean mask for BOS/EOS/PAD/None positions
            ss8_masks: Boolean mask for MASK positions
        """
        device = mask.device
        
        # Convert SS8 labels to token indices, mapping None to UNK
        ss8_tokens = []
        for label in observation:
            if label is None: ss8_tokens.append(self.mapping["UNK"])
            else: ss8_tokens.append(self.mapping[label])
        
        # Truncate if too long (reserve space for BOS/EOS)
        ss8_tokens = ss8_tokens[:self.full_length-2]

        # Add BOS and EOS tokens
        content = torch.tensor([self.mapping["BOS"], *ss8_tokens, self.mapping["EOS"]], dtype=torch.long, device=device)
        content_len = content.numel()

        # Add Padding
        padded_ss8 = torch.full((self.full_length,), self.mapping["PAD"], dtype=torch.long, device=device)
        padded_ss8[:content_len] = content[:content_len]

        # Create beospank mask: 1s for BOS/EOS/PAD/None, 0s for real content
        ss8_beospank = torch.zeros(self.full_length, dtype=torch.bool, device=device)
        ss8_beospank[padded_ss8 == self.mapping["BOS"]] = 1
        ss8_beospank[padded_ss8 == self.mapping["EOS"]] = 1
        ss8_beospank[padded_ss8 == self.mapping["PAD"]] = 1
        ss8_beospank[padded_ss8 == self.mapping["UNK"]] = 1
        # ss8_pad_pos = (padded_ss8 == self.mapping["PAD"])

        ss8_masks = mask & ~ss8_beospank
        masked_ss8 = self.corrupt(padded_ss8, ss8_masks)

        # Sanity checks
        assert torch.max(padded_ss8) < len(self.mapping), "SS8 Tokenization failed!"
        assert torch.min(padded_ss8) >= 0, "SS8 Tokenization failed!"
        assert padded_ss8.numel() == self.full_length, "SS8 Tokenization failed!"

        return padded_ss8, masked_ss8, ss8_beospank.bool(), ss8_masks.bool() # ss8_pad_pos.bool()

    def corrupt(self, unmasked_data, masks):
        if self.corruption_mode == CorruptionMode.MASK:
            tensor_of_masks = torch.full((self.full_length,), self.mapping["MASK"], dtype=torch.long, device=unmasked_data.device)
            return torch.where(masks, tensor_of_masks, unmasked_data)

        elif self.corruption_mode == CorruptionMode.UNIFORM:
            uniform_content = torch.randint(0, self.max_ss8_value + 1, (unmasked_data.shape[0],), dtype=torch.long, generator=self.generator).to(unmasked_data.device)
            return torch.where(masks, uniform_content, unmasked_data)
        
        raise ValueError(f"Unknown corruption mode: {self.corruption_mode}")
    
    def print_token(self, tok):
        return self.reverse_mapping[tok]


class SASATokenizer(Tokenizer):
    """
    A Tokenizer and Padder for SASA tokens.
    """
    def __init__(self, full_length, generator=None, corruption_mode=CorruptionMode.MASK):
        # Get SASA tokens (binned values)
        sasa = {name: member.value for name, member in SASA_TOKENS.__members__.items()}

        # Get the highest SASA token value to avoid conflicts
        self.max_sasa_value = max(sasa.values()) if sasa else -1

        # Add special tokens with non-conflicting values
        special = {name: member.value + self.max_sasa_value + 1 
                  for name, member in SPECIAL_TOKENS.__members__.items()}
        
        self.thresholds = (0.14, 2.09, 6.49, 12.69, 20.23, 29.03, 38.17, 47.43, 56.62, 65.50, 76.08, 86.71, 99.48, 115.35, 137.27)
        self.mapping = {**sasa, **special}
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        self.full_length = full_length
        self.generator = generator
        self.corruption_mode = corruption_mode

    def _bin_sasa_value(self, value):
        """Convert continuous SASA value to bin token using optimal quantile-based boundaries."""
        # Optimal bin boundaries based on quantile analysis of 22,185 real SASA values
        # Each bin contains approximately equal numbers of residues (~1,386 each)
        if value is None: return self.mapping["UNK"]
        idx = bisect_right(self.thresholds, value)
        return self.mapping[f"BIN_{idx}"]

    def tokenize(self, observation, mask):
        """
        Args:
            observation: List of SASA values (or None values), e.g., [45.2, None, 12.8, 156.7]
            mask: Boolean mask for MASK positions
        Returns:
            padded_sasa: Tensor of SASA tokens
            sasa_beospank: Boolean mask for BOS/EOS/PAD/None positions
        """
        device = mask.device
        
        # Convert SASA values to binned token indices
        sasa_tokens = [self._bin_sasa_value(value) for value in observation]
        
        # Truncate if too long (reserve space for BOS/EOS)
        sasa_tokens = sasa_tokens[:self.full_length-2]

        # Add BOS and EOS tokens
        content = torch.tensor([self.mapping["BOS"], *sasa_tokens, self.mapping["EOS"]], dtype=torch.long, device=device)
        content_len = content.numel()

        # Add Padding
        padded_sasa = torch.full((self.full_length,), self.mapping["PAD"], dtype=torch.long, device=device)
        padded_sasa[:content_len] = content[:content_len]

        # Create beospank mask: 1s for BOS/EOS/PAD/None, 0s for real content
        sasa_beospank = torch.zeros(self.full_length, dtype=torch.bool, device=device)
        sasa_beospank[padded_sasa == self.mapping["BOS"]] = 1
        sasa_beospank[padded_sasa == self.mapping["EOS"]] = 1
        sasa_beospank[padded_sasa == self.mapping["PAD"]] = 1
        sasa_beospank[padded_sasa == self.mapping["UNK"]] = 1
        # sasa_pad_pos = (padded_sasa == self.mapping["PAD"])

        sasa_masks = mask & ~sasa_beospank
        masked_sasa = self.corrupt(padded_sasa, sasa_masks)

        # Sanity checks
        assert torch.max(padded_sasa) < len(self.mapping), "SASA Tokenization failed!"
        assert torch.min(padded_sasa) >= 0, "SASA Tokenization failed!"
        assert padded_sasa.numel() == self.full_length, "SASA Tokenization failed!"

        return padded_sasa, masked_sasa, sasa_beospank.bool(), sasa_masks.bool() # sasa_pad_pos.bool()

    def corrupt(self, unmasked_data, masks):
        if self.corruption_mode == CorruptionMode.MASK:
            tensor_of_masks = torch.full((self.full_length,), self.mapping["MASK"], dtype=torch.long, device=unmasked_data.device)
            return torch.where(masks, tensor_of_masks, unmasked_data)

        elif self.corruption_mode == CorruptionMode.UNIFORM:
            uniform_content = torch.randint(0, self.max_sasa_value + 1, (unmasked_data.shape[0],), dtype=torch.long, generator=self.generator).to(unmasked_data.device)
            return torch.where(masks, uniform_content, unmasked_data)
        
        raise ValueError(f"Unknown corruption mode: {self.corruption_mode}")
    
    def print_token(self, tok):
        return self.reverse_mapping[tok]


class PLDDTTokenizer(Tokenizer):
    """
    A Tokenizer and Padder for pLDDT tokens.
    """
    def __init__(self, full_length, generator=None, corruption_mode=CorruptionMode.MASK):
        # Get pLDDT tokens (binned values)
        plddt = {name: member.value for name, member in PLDDT_TOKENS.__members__.items()}

        # Get the highest pLDDT token value to avoid conflicts
        self.max_plddt_value = max(plddt.values()) if plddt else -1

        # Add special tokens with non-conflicting values
        special = {name: member.value + self.max_plddt_value + 1 
                  for name, member in SPECIAL_TOKENS.__members__.items()}

        self.mapping = {**plddt, **special}
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        self.full_length = full_length
        self.generator = generator
        self.corruption_mode = corruption_mode
        self.thresholds = tuple((i+1)/50 for i in range(49))

    def _bin_plddt_value(self, value):
        """Convert continuous pLDDT value to bin token using uniform intervals."""
        if value is None: return self.mapping["UNK"]
        assert value >= 0
        idx = bisect_right(self.thresholds, value)
        return self.mapping[f"BIN_{idx}"]
        
    def tokenize(self, observation, mask):
        """
        Args:
            observation: List of pLDDT values (or None values), e.g., [45.2, None, 12.8, 156.7]
            mask: Boolean mask for MASK positions
        Returns:
            padded_plddt: Tensor of pLDDT tokens
            plddt_beospank: Boolean mask for BOS/EOS/PAD/None positions
        """
        device = mask.device
        
        # Convert pLDDT values to binned token indices
        plddt_tokens = [self._bin_plddt_value(value) for value in observation]
        
        # Truncate if too long (reserve space for BOS/EOS) 
        plddt_tokens = plddt_tokens[:self.full_length-2]

        # Add BOS and EOS tokens
        content = torch.tensor([self.mapping["BOS"], *plddt_tokens, self.mapping["EOS"]], dtype=torch.long, device=device)
        content_len = content.numel()

        # Add Padding
        padded_plddt = torch.full((self.full_length,), self.mapping["PAD"], dtype=torch.long, device=device)
        padded_plddt[:content_len] = content[:content_len]

        # Create beospank mask: 1s for BOS/EOS/PAD/None, 0s for real content
        plddt_beospank = torch.zeros(self.full_length, dtype=torch.bool, device=device)
        plddt_beospank[padded_plddt == self.mapping["BOS"]] = 1
        plddt_beospank[padded_plddt == self.mapping["EOS"]] = 1
        plddt_beospank[padded_plddt == self.mapping["PAD"]] = 1
        plddt_beospank[padded_plddt == self.mapping["UNK"]] = 1
        # plddt_pad_pos = (padded_plddt == self.mapping["PAD"])

        plddt_masks = mask & ~plddt_beospank
        masked_plddt = self.corrupt(padded_plddt, plddt_masks)

        # Sanity checks
        assert torch.max(padded_plddt) < len(self.mapping), "pLDDT Tokenization failed!"
        assert torch.min(padded_plddt) >= 0, "pLDDT Tokenization failed!"
        assert padded_plddt.numel() == self.full_length, "pLDDT Tokenization failed!"

        return padded_plddt, masked_plddt, plddt_beospank.bool(), plddt_masks.bool() # plddt_pad_pos.bool()

    def corrupt(self, unmasked_data, masks):
        if self.corruption_mode == CorruptionMode.MASK:
            tensor_of_masks = torch.full((self.full_length,), self.mapping["MASK"], dtype=torch.long, device=unmasked_data.device)
            return torch.where(masks, tensor_of_masks, unmasked_data)

        elif self.corruption_mode == CorruptionMode.UNIFORM:
            uniform_content = torch.randint(0, self.max_plddt_value + 1, (unmasked_data.shape[0],), dtype=torch.long, generator=self.generator).to(unmasked_data.device)
            return torch.where(masks, uniform_content, unmasked_data)
        
        raise ValueError(f"Unknown corruption mode: {self.corruption_mode}")

    def print_token(self, tok):
        return self.reverse_mapping[tok]

class GlobalAnnotationTokenizer(Tokenizer):
    """
    A Tokenizer and Padder for global annotation tokens.
    This class does NOT handle MASK tokens.
    """
    def __init__(self, full_length):
        # Get global annotation tokens (dynamically loaded vocabulary)
        global_annotation = {name: value for name, value in GLOBAL_ANNOTATION_TOKENS._members.items()}

        # Get the highest global annotation token value to avoid conflicts
        max_global_annotation_value = max(global_annotation.values()) if global_annotation else -1

        # Add special tokens with non-conflicting values
        special = {name: member.value + max_global_annotation_value + 1 
                  for name, member in SPECIAL_TOKENS.__members__.items()}

        self.mapping = {**global_annotation, **special}
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        self.full_length = full_length

    def tokenize(self, observation):
        """
        Args:
            observation: List of global annotation labels (or None values), e.g., ['dimer interface', 'DNA binding site']
        Returns:
            padded_global_annotation: Tensor of global annotation tokens
            global_annotation_beospank: Boolean mask for BOS/EOS/PAD/None positions
        """
        # Convert global annotation labels to token indices
        global_annotation_tokens = []
        for label in observation:
            if label in self.mapping: global_annotation_tokens.append(self.mapping[label])
            else: global_annotation_tokens.append(self.mapping["UNK"])
        
        # Truncate if too long (reserve space for BOS/EOS)
        global_annotation_tokens = global_annotation_tokens[:self.full_length-2]

        # Add BOS and EOS tokens
        content = torch.tensor([self.mapping["BOS"], *global_annotation_tokens, self.mapping["EOS"]], dtype=torch.long)
        content_len = content.numel()

        # Add Padding
        padded_global_annotation = torch.full((self.full_length,), self.mapping["PAD"], dtype=torch.long)
        padded_global_annotation[:content_len] = content[:content_len]

        # Create beospank mask: 1s for BOS/EOS/PAD/None, 0s for real content
        global_annotation_beospank = torch.zeros(self.full_length, dtype=torch.bool)
        global_annotation_beospank[padded_global_annotation == self.mapping["BOS"]] = 1
        global_annotation_beospank[padded_global_annotation == self.mapping["EOS"]] = 1
        global_annotation_beospank[padded_global_annotation == self.mapping["PAD"]] = 1
        global_annotation_beospank[padded_global_annotation == self.mapping["UNK"]] = 1

        # Sanity checks
        assert torch.max(padded_global_annotation) < len(self.mapping), "Global Annotation Tokenization failed!"
        assert torch.min(padded_global_annotation) >= 0, "Global Annotation Tokenization failed!"
        assert padded_global_annotation.numel() == self.full_length, "Global Annotation Tokenization failed!"

        return padded_global_annotation, global_annotation_beospank.bool()

    def corrupt(self, unmasked_data, masks):
        if self.corruption_mode == CorruptionMode.MASK:
            tensor_of_masks = torch.full((self.full_length,), self.mapping["MASK"], dtype=torch.long, device=unmasked_data.device)
            return torch.where(masks, tensor_of_masks, unmasked_data)

        elif self.corruption_mode == CorruptionMode.UNIFORM:
            uniform_content = torch.randint(0, self.max_plddt_value + 1, (unmasked_data.shape[0],), dtype=torch.long, generator=self.generator).to(unmasked_data.device)
            return torch.where(masks, uniform_content, unmasked_data)
        
        raise ValueError(f"Unknown corruption mode: {self.corruption_mode}")

    def print_token(self, tok):
        return self.reverse_mapping[tok]


class PerResidueAnnotationTokenizer(Tokenizer):
    """
    A Tokenizer and Padder for per-residue annotation tokens.
    Handles multiple annotations per residue by creating a fixed-size tensor
    of shape [full_length, max_annotations_per_residue].
    """
    def __init__(self, full_length, max_annotations_per_residue, generator=None, corruption_mode=CorruptionMode.MASK):
        # Get per-residue annotation tokens (dynamically loaded vocabulary)
        per_residue_annotation = {name: value for name, value in PER_RESIDUE_ANNOTATION_TOKENS._members.items()}

        # Get the highest per-residue annotation token value to avoid conflicts
        self.max_annotation_value = max(per_residue_annotation.values()) if per_residue_annotation else -1

        # Add special tokens with non-conflicting values
        special = {name: member.value + self.max_annotation_value + 1 
                  for name, member in SPECIAL_TOKENS.__members__.items()}

        self.mapping = {**per_residue_annotation, **special}
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        self.full_length = full_length
        self.max_annotations_per_residue = max_annotations_per_residue
        self.generator = generator
        self.corruption_mode = corruption_mode

    def _process_residue_annotations(self, annotations):
        """Convert residue annotations to list of token indices, pad/truncate to max_annotations_per_residue."""        
        # Convert to token IDs
        tokens = []
        for term in annotations[:self.max_annotations_per_residue]:  # Truncate to max
            if term is None:
                tokens.append(self.mapping["UNK"])
            else:
                term = term.strip()
                if term in self.mapping: tokens.append(self.mapping[term])
                else: tokens.append(self.mapping["UNK"])
        
        # Pad to max_annotations_per_residue
        while len(tokens) < self.max_annotations_per_residue:
            tokens.append(self.mapping["PAD"])
        
        return tokens
        
    def tokenize(self, observation, mask):
        """
        Args:
            observation: List of annotation lists per residue, e.g., 
                        [["dimer interface","DNA binding site"], None, ["active site"], ...]
                        Each element is either None or a list of annotation term strings.
            mask: Boolean mask for MASK positions
        Returns:
            padded_annotations: Tensor of annotation tokens [full_length, max_annotations_per_residue]
            annotation_beospank: Boolean mask for BOS/EOS/PAD/UNK annotation tokens [full_length, max_annotations_per_residue]
        """
        device = mask.device
        
        # Convert each residue's annotations to token indices
        residue_annotation_tokens = []
        for annot in observation:
            tokens = self._process_residue_annotations(annot)
            residue_annotation_tokens.append(tokens)
        
        # Truncate if too long (reserve space for BOS/EOS)
        residue_annotation_tokens = residue_annotation_tokens[:self.full_length-2]

        # Add BOS and EOS tokens
        bos_tokens = [self.mapping["BOS"]] * self.max_annotations_per_residue
        eos_tokens = [self.mapping["EOS"]] * self.max_annotations_per_residue
        content = torch.tensor([bos_tokens, *residue_annotation_tokens, eos_tokens], dtype=torch.long, device=device)
        content_len = len(content)

        # Add Padding
        padded_annotations = torch.full((self.full_length, self.max_annotations_per_residue), self.mapping["PAD"], dtype=torch.long, device=device)
        padded_annotations[:content_len] = content[:content_len] # size: [full_length, max_annotations_per_residue]

        # Create beospank mask: 1s for BOS/EOS/PAD/UNK annotation tokens, 0s for real content - size: [full_length, max_annotations_per_residue]
        annotation_beospank = (padded_annotations == self.mapping["BOS"]) | \
                              (padded_annotations == self.mapping["EOS"]) | \
                              (padded_annotations == self.mapping["PAD"]) | \
                              (padded_annotations == self.mapping["UNK"])
        
        # Expand mask to match annotation dimensions [L] -> [L, K]
        mask_expanded = mask.unsqueeze(-1).expand(-1, self.max_annotations_per_residue)
        annotation_masks = mask_expanded & ~annotation_beospank
        masked_annotations = self.corrupt(padded_annotations, annotation_masks)

        # Sanity checks
        assert torch.max(padded_annotations) < len(self.mapping), "Per-residue annotation Tokenization failed!"
        assert torch.min(padded_annotations) >= 0, "Per-residue annotation Tokenization failed!"
        assert padded_annotations.shape == (self.full_length, self.max_annotations_per_residue), "Per-residue annotation padding shape mismatch!"

        return padded_annotations, masked_annotations, annotation_beospank.bool(), annotation_masks.bool()

    def corrupt(self, unmasked_data, masks):
        if self.corruption_mode == CorruptionMode.MASK:
            tensor_of_masks = torch.full_like(unmasked_data, self.mapping["MASK"], dtype=torch.long, device=unmasked_data.device)
            return torch.where(masks, tensor_of_masks, unmasked_data)

        elif self.corruption_mode == CorruptionMode.UNIFORM:
            uniform_content = torch.randint(0, self.max_annotation_value + 1, unmasked_data.shape, dtype=torch.long, generator=self.generator).to(unmasked_data.device)
            return torch.where(masks, uniform_content, unmasked_data)
        
        raise ValueError(f"Unknown corruption mode: {self.corruption_mode}")

    def print_token(self, tok):
        return self.reverse_mapping[tok]


"""
To use the following test function, run the following code:

from odyssey.src.tokenizer import *
from odyssey.src.dataset import *

pd = ProteinDataset("/workspace/demo/Odyssey/sample_data/3k.csv")
mask = torch.bernoulli(torch.full((2048,), 0.1)).bool()
idx = 0

seq = SequenceTokenizer(2048)
print_tokenized_sequence(seq.print_token, *seq.tokenize(pd.__getitem__(idx)[0], mask))

coord = CoordinatesTokenizer(2048)
print_tokenized_sequence(coord.print_token, *coord.tokenize(pd.__getitem__(idx)[1], mask))
"""
def print_tokenized_sequence(print_token, unmasked, masked, beospank, mask, limit=500):

    def hot(b):
        return "1" if b else ""


    data = {'Unmasked': [print_token(tok) for tok in unmasked[:limit]], 
            'Masked': [print_token(tok) for tok in masked[:limit]],
            'Beospank?': [hot(b) for b in beospank[:limit]],
            'Mask?': [hot(m) for m in mask[:limit]],
            }

    table = tabulate(data, headers='keys', tablefmt='rst')

    print(table)
