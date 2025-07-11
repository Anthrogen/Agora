from enum import Enum
import os

class SPECIAL_TOKENS(Enum):
    # Note= you cannot do something like <PAD> in an Enum environment.
    PAD  = 0     # Padding token
    MASK = 1    # Mask token for MLM
    UNK  = 2     # Unknown amino acid
    BOS  = 3     # Beginning of sequence
    EOS  = 4     # End of sequence

class SEQUENCE_TOKENS(Enum):
    # 20 canonical amino acids
    A= 0   # Alanine
    R= 1   # Arginine  
    N= 2   # Asparagine
    D= 3   # Aspartic acid
    C= 4   # Cysteine
    Q= 5   # Glutamine
    E= 6   # Glutamic acid
    G= 7   # Glycine
    H= 8   # Histidine
    I= 9   # Isoleucine
    L= 10  # Leucine
    K= 11  # Lysine
    M= 12  # Methionine
    F= 13  # Phenylalanine
    P= 14  # Proline
    S= 15  # Serine
    T= 16  # Threonine
    W= 17  # Tryptophan
    Y= 18  # Tyrosine
    V= 19  # Valine
    
    # 4 non-standard amino acids as used in Odyssey
    B= 20  # Asparagine or Aspartic acid (ambiguous)
    U= 21  # Selenocysteine  
    Z= 22  # Glutamic acid or Glutamine (ambiguous)
    O= 23  # Ornithine
    J= 24  # Leucine or Isoleucine (ambiguous)

    # X= SPECIAL_TOKENS[__UNK__] # Unknown amino acid sometimes denoted by X in JSON files.

class SS8_TOKENS(Enum):
    # 9-class secondary structure classification (DSSP)
    C = 0   # Coil/Loop
    E = 1   # Extended strand, participates in beta ladder
    B = 2   # Residue in isolated beta-bridge
    T = 3   # Turn
    S = 4   # Bend
    H = 5   # Alpha helix
    G = 6   # 3_10 helix
    I = 7   # Pi helix
    P = 8   # Polyproline II helix

class SASA_TOKENS(Enum):
    # 16-bin SASA (Solvent Accessible Surface Area) values
    # Optimal quantile-based bins for equal population distribution
    # Based on analysis of 22,185 real SASA values from protein dataset
    BIN_0  = 0   # 0.00 - 0.14 Ų (deeply buried)
    BIN_1  = 1   # 0.14 - 2.09 Ų (buried)
    BIN_2  = 2   # 2.09 - 6.49 Ų (mostly buried)
    BIN_3  = 3   # 6.49 - 12.69 Ų (partially buried)
    BIN_4  = 4   # 12.69 - 20.23 Ų (partially buried)
    BIN_5  = 5   # 20.23 - 29.03 Ų (partially exposed)
    BIN_6  = 6   # 29.03 - 38.17 Ų (partially exposed)
    BIN_7  = 7   # 38.17 - 47.43 Ų (moderately exposed)
    BIN_8  = 8   # 47.43 - 56.62 Ų (moderately exposed)
    BIN_9  = 9   # 56.62 - 65.50 Ų (exposed)
    BIN_10 = 10  # 65.50 - 76.08 Ų (exposed)
    BIN_11 = 11  # 76.08 - 86.71 Ų (highly exposed)
    BIN_12 = 12  # 86.71 - 99.48 Ų (highly exposed)
    BIN_13 = 13  # 99.48 - 115.35 Ų (very highly exposed)
    BIN_14 = 14  # 115.35 - 137.27 Ų (extremely exposed)
    BIN_15 = 15  # 137.27+ Ų (maximally exposed)

class PLDDT_TOKENS(Enum):
    # 50-bin pLDDT (predicted Local Distance Difference Test) confidence scores
    # Bins represent uniform intervals from 0.0 to 1.0 confidence    
    # BIN_0 = 0.00-0.02, BIN_1 = 0.02-0.04, ..., BIN_49 = 0.98-1.00
    BIN_0 = 0; BIN_1 = 1; BIN_2 = 2; BIN_3 = 3; BIN_4 = 4; BIN_5 = 5; BIN_6 = 6; BIN_7 = 7; BIN_8 = 8; BIN_9 = 9
    BIN_10 = 10; BIN_11 = 11; BIN_12 = 12; BIN_13 = 13; BIN_14 = 14; BIN_15 = 15; BIN_16 = 16; BIN_17 = 17; BIN_18 = 18; BIN_19 = 19
    BIN_20 = 20; BIN_21 = 21; BIN_22 = 22; BIN_23 = 23; BIN_24 = 24; BIN_25 = 25; BIN_26 = 26; BIN_27 = 27; BIN_28 = 28; BIN_29 = 29
    BIN_30 = 30; BIN_31 = 31; BIN_32 = 32; BIN_33 = 33; BIN_34 = 34; BIN_35 = 35; BIN_36 = 36; BIN_37 = 37; BIN_38 = 38; BIN_39 = 39
    BIN_40 = 40; BIN_41 = 41; BIN_42 = 42; BIN_43 = 43; BIN_44 = 44; BIN_45 = 45; BIN_46 = 46; BIN_47 = 47; BIN_48 = 48; BIN_49 = 49

class PER_RESIDUE_ANNOTATION_TOKENS:
    """
    Dynamic vocabulary for per-residue annotation tokens.
    This class is populated by load_annotation_tokens() based on frequency thresholds.
    """
    # Class variables to store the vocabulary
    _members = {}  # {term_name: value}
    _value_to_term = {}  # {value: term_name}
    
    @classmethod
    def __len__(cls): return len(cls._members)
    
    @classmethod
    def __iter__(cls): return iter(cls._members.items())
    
    @classmethod
    def __members__(cls):
        """Return members dict for compatibility with Enum interface."""
        return cls._members


class GLOBAL_ANNOTATION_TOKENS:
    """
    Dynamic vocabulary for global annotation tokens.
    This class is populated by load_annotation_tokens() based on frequency thresholds.
    """
    # Class variables to store the vocabulary
    _members = {}  # {term_name: value}
    _value_to_term = {}  # {value: term_name}
    
    @classmethod
    def __len__(cls): return len(cls._members)
    
    @classmethod
    def __iter__(cls): return iter(cls._members.items())
    
    @classmethod
    def __members__(cls):
        """Return members dict for compatibility with Enum interface."""
        return cls._members


def load_annotation_tokens(vocab_file_path, token_class):
    """
    Load annotation tokens from a vocabulary text file and populate the given token class.
    
    Args:
        vocab_file_path (str): Path to the vocabulary text file (one term per line)
        token_class: Either PER_RESIDUE_ANNOTATION_TOKENS or GLOBAL_ANNOTATION_TOKENS
    
    Returns:
        int: Number of annotation tokens loaded (not including special tokens)
    """
    print(f"Loading annotation tokens from {vocab_file_path}")
    # Load the vocabulary file
    with open(vocab_file_path, 'r') as f: terms = [line.strip() for line in f if line.strip()]
    
    # Clear existing members
    token_class._members = {}
    token_class._value_to_term = {}
    
    # Populate the class with terms (already sorted by frequency in the file)
    for i, term in enumerate(terms):
        token_class._members[term] = i
        token_class._value_to_term[i] = term
    
    # Return number of annotation tokens
    return len(terms)
