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
