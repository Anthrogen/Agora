import torch
from dataclasses import dataclass
from typing import List, Optional, Any
from src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from dataclasses import dataclass, field

@dataclass
class TransformerConfig:
    """Model architecture configuration."""
    d_model:                        int # 768  # Model dimensions
    n_heads:                        int  # 12
    n_layers:                       int  # 12
    seq_vocab:                      int   # Sequence tokens + special tokens
    struct_vocab:                   int   # FSQ tokens + special tokens
    max_len:                        int 
    dropout:                        float   # Other architecture params
    ff_mult:                        int 
                                     
    
    # Consensus-specific parameters
    consensus_num_iterations:       int = None # Number of consensus gradient iterations
    consensus_connectivity_type:    str = None # "local_window" or "top_w"
    consensus_w:                    int = None  # Window size for local_window, or w value for top_w
    consensus_r:                    int = None  # Rank of Lambda_ij matrices
    consensus_edge_hidden_dim:      int = None # Hidden dim for edge networks

    def __post_init__(self):
        self.ff_hidden_dim: int = self.d_model * self.ff_mult
        assert self.d_model > 0
        assert self.n_heads > 0
        assert self.n_layers > 0
        assert self.seq_vocab > 0
        assert self.struct_vocab > 0
        assert self.max_len > 0


@dataclass
class FsqConfig:
    """Model architecture configuration."""
    # FSQ parameters
    fsq_dim:                        int 

    # Transformer parameters
    model_type:                     Optional[str] # Placeholder for model type
    d_model:                        int  # 768  # Model dimensions
    latent_dim:                     int 
    n_heads:                        int  # 12
    n_layers:                       int  # 12
    seq_vocab:                      int   # Sequence tokens + special tokens
    struct_vocab:                   int   # FSQ tokens + special tokens
    max_len:                        int 
    dropout:                        float   # Other architecture params
    ff_mult:                        int
                                    
    # Consensus-specific parameters
    consensus_num_iterations:       int   # Number of consensus gradient iterations
    consensus_connectivity_type:    str   # "local_window" or "top_w"
    consensus_w:                    int   # Window size for local_window, or w value for top_w
    consensus_r:                    int   # Rank of Lambda_ij matrices
    consensus_edge_hidden_dim:      int   # Hidden dim for edge networks

    fsq_levels:                     list[int] = field(default_factory=lambda: [7, 5, 5, 5, 5])
    # TOOD: needs to be in training_configuration
    stage:                          str = "stage_1" # "stage_1" or "stage_2"
    fsq_encoder:                    Optional[Any] = None

    def __post_init__(self):
        self.ff_hidden_dim = self.d_model * self.ff_mult


@dataclass
class TrainingConfig:
    """Training process configuration."""
    # Model types should be in models configuration...
    model_types:                  List[str]  = None# Models to train - can be any subset of ["SA", "GA", "RA", "SC"]
    batch_size:                   int = None # Training hyperparameters
    max_epochs:                   int = None
    learning_rate:                float = None
    num_iter:                     int  = None# Number of iterations to repeat training
    
    masking_strategy:             str = None
    # for masking_strategy = "simple":



    # for masking_strategy = "complex"



    data_dir:                     str = None  # Data paths
    checkpoint_dir:               str = None   # Checkpointing
    reference_model_seed:         int=1234  # Reference model seed for consistent parameter initialization across architectures

    #########################################################


    # Optional parameters:
    mask_prob_seq:                float  = None # Masking probability for sequence tokens
    mask_prob_coords:             float  = None # Masking probability for structure tokens

    # TODO: should be incorporated into FSQEncoder training where possible.
    # Or should there be an entirely new object for training the FSQ?
    seq_loss_weight:              float = None  # sequence loss weight - simple: 1.0
    struct_loss_weight:           float = None  # structure loss weight - simple: 1.0

    # Cross-entropy loss function: which elements should contribute to the loss?
    # "masked": only masked positions
    # "non_beospank": all non-BOS/EOS/PAD/UNK positions, including masks
    ce_loss_function_elements:    str = None

    # TODO: remove entirely and incorporate into model_librarian.py
    # TODO: use os.path.join instead of /.
    # Model paths (models in /scripts/checkpoints)
    simple_checkpoint_pattern:    str = None
    complex_checkpoint_pattern:   str = None
    discrete_diffusion_checkpoint_pattern: str = None
    
    # FSQ encoder paths (in /checkpoints, not /scripts/checkpoints)
    fsq_encoder_pattern:          str = None

    # Here only because of validation_trunk.py.  We NEED to get these out of here ASAP.
    time_indices:                 list[int] = None # TODO: move into DiffusionConfig or elsewhere.  Perhaps into MaskingConfig?
    # Time indices to evaluate (directly specified)
    training_methods:             list[Any] = None


    def __post_init__(self):
        # Bascially assert that nothing is None, except for optional arugments
            
        # if self.masking_strategy == 'simple':
        #     assert self.mask_prob_seq is not None and self.mask_prob_coords is not None

        # assert self.masking_strategy in ('simple', 'complex', 'discrete_diffusion')
        # if self.ce_loss_function_elements is not None:
        #     assert self.ce_loss_function_elements in ('masked', 'non_beospank')

        # for mt in self.model_types:
        #     assert mt in ('SA', 'GA', 'RA', 'C')

        # assert self.batch_size > 0
        # assert self.max_epochs > 0
        # assert self.learning_rate > 0
        # assert self.num_iter > 0
        pass
        
@dataclass
class DiffusionConfig:
    """Discrete diffusion configuration."""
    # Noise schedule parameters
    noise_schedule:        str  # Type of noise schedule ("linear", "inverted_u", or "uniform")
    sigma_min:             float  # Minimum noise level
    sigma_max:             float  # Maximum noise level
    num_timesteps:         int  # Number of discrete timesteps for training
    
    # Absorbing state tokens (using MASK token index)
    seq_absorb_token:      int
    struct_absorb_token:   int

    def __post_init__(self):
        assert self.noise_schedule in ('linear', 'inverted_u', 'uniform')
        assert self.sigma_min > 0
        assert self.sigma_max > 0
        assert self.num_timesteps > 0
        assert self.seq_absorb_token is not None
        assert self.struct_absorb_token is not None



@dataclass
class AttentionConfig:
    """Attention configuration."""
    pass

class SelfConsensusConfig(AttentionConfig):
    """SelfConsensus configuration."""
    # Consensus-specific parameters
    consensus_num_iterations:       int   # Number of consensus gradient iterations
    consensus_connectivity_type:    str   # "local_window" or "top_w"
    consensus_w:                    int   # Window size for local_window, or w value for top_w
    consensus_r:                    int   # Rank of Lambda_ij matrices
    consensus_edge_hidden_dim:      int   # Hidden dim for edge networks

class ReflexiveAttentionConfig(AttentionConfig):
    """Reflexive attention configuration."""
    pass

class SelfAttentionConfig(AttentionConfig):
    pass

class GeometricAttentionConfig(AttentionConfig):
    pass

class ConfigurationError(Exception):
    def __init__(self, message):
        self.message = message
        
    def __str__(self):
        return self.message
