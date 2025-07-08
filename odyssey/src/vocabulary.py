from enum import Enum

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
    Q= 5  # Glutamine
    E= 6  # Glutamic acid
    G= 7  # Glycine
    H= 8 # Histidine
    I= 9  # Isoleucine
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
