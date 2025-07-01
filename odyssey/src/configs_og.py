import torch
from dataclasses import dataclass
from typing import List, Optional, Any
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from dataclasses import dataclass, field
import os

@dataclass
class Config:
    pass

@dataclass
class TransformerConfig(Config):
    """Model architecture configuration."""

    style:                          str = None

    d_model:                        int # 768  # Model dimensions
    n_heads:                        int  # 12
    n_layers:                       int  # 12

    max_len:                        int 
    dropout:                        float   # Other architecture params
    ff_mult:                        int 
                                     
    # # Consensus-specific parameters
    # consensus_num_iterations:       int = None # Number of consensus gradient iterations
    # consensus_connectivity_type:    str = None # "local_window" or "top_w"
    # consensus_w:                    int = None  # Window size for local_window, or w value for top_w
    # consensus_r:                    int = None  # Rank of Lambda_ij matrices
    # consensus_edge_hidden_dim:      int = None # Hidden dim for edge networks
    first_block_config:               BlockConfig = None

    # TODO: These need to go.  seq_vocab should come from voacbulary.py and struuct_vocab should come from the FSQEncoder object.
    seq_vocab:                      int   # Sequence tokens + special tokens
    struct_vocab:                   int   # FSQ tokens + special tokens

    def __post_init__(self):
        assert self.style in ('stage_1', 'stage_2', 'mlm', 'discrete_diffusion')
        self.ff_hidden_dim: int = self.d_model * self.ff_mult

        assert isinstance(self.d_model, int) and self.d_model > 0
        assert isinstance(self.n_heads, int) and self.n_heads > 0
        assert isinstance(self.n_layers, int) and self.n_layers > 0

        assert isinstance(self.max_len, int) and self.max_len > 0
        assert isinstance(self.first_block_config, BlockConfig)

        # TODO: get rid of
        assert self.seq_vocab > 0
        assert self.struct_vocab > 0


@dataclass
class TrunkConfig(TransformerConfig):
    """Trunk model configuration."""
    fsq_encoder_path:                  str = None

    def __post_init__(self):
        assert self.style in ('mlm', 'discrete_diffusion')
        assert self.fsq_encoder_path is not None and os.path.exists(self.fsq_encoder_path)

@dataclass
class FSQConfig(TransformerConfig):
    """Model architecture configuration."""

    # Transformer parameters
    latent_dim:                     int = None # pre-quantized CONTINUOUS latent dimension.

    fsq_levels:                     list[int] = None # codebook
    fsq_encoder_path:               str = None

    def __post_init__(self):

        self.fsq_dim = len(self.fsq_levels)

        assert self.fsq_dim > 0
        assert self.latent_dim > 0
        assert len(self.fsq_levels) > 0
        for l in self.fsq_levels:
            assert l > 0

        # Should this just be in training config?
        assert self.style in ('stage_1', 'stage_2')
        
        if self.style == 'stage_2':
            assert self.fsq_encoder_path is not None and os.path.exists(self.fsq_encoder_path)
            # TODO: check that the codebook of this FSQencoder object matches the codebook provided above.

@dataclass
class TrainingConfig(Config):
    """Training process configuration."""
    # Model types should be in models configuration...
    batch_size:                   int = None # Training hyperparameters
    max_epochs:                   int = None
    learning_rate:                float = None
    
    mask_config:                  MaskConfig = None
    loss_config:                  LossConfig = None

    data_dir:                     str = None  # Data paths
    checkpoint_dir:               str = None   # Checkpointing


    #########################################################

    # num_iter
    # get rid fo num_iter and absorb into the training wrapper (possibly shell script.)
    # # TODO: remove entirely and incorporate into model_librarian.py
    # # TODO: use os.path.join instead of /.
    # # Model paths (models in /scripts/checkpoints)
    # simple_checkpoint_pattern:    str = None
    # complex_checkpoint_pattern:   str = None
    # discrete_diffusion_checkpoint_pattern: str = None
    
    # # FSQ encoder paths (in /checkpoints, not /scripts/checkpoints)
    # # this hsould just be in the ModelConfig object.
    # # fsq_encoder_pattern:          str = None

    # # we'll worry about validation later.
    # # maybe we can have a "pseudo-diffusion-masker" config object that is derived from MaskConfig
    # # Here only because of validation_trunk.py.  We NEED to get these out of here ASAP.
    # time_indices:                 list[int] = None # TODO: move into DiffusionConfig or elsewhere.  Perhaps into MaskConfig?
    # # Time indices to evaluate (directly specified)
    # training_methods:             list[Any] = None


    def __post_init__(self):
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.max_epochs, int) and self.max_epochs > 0
        assert isinstance(self.learning_rate, float) and self.learning_rate > 0

        assert isinstance(self.mask_config, MaskConfig)
        assert isinstance(self.loss_config, LossConfig)

        assert self.data_dir is not None and os.path.exists(self.data_dir), f"Data directory {self.data_dir} does not exist."
        assert self.checkpoint_dir is not None and os.path.exists(self.checkpoint_dir), f"Checkpoint directory {self.checkpoint_dir} does not exist."
        
################################################################################
# Training Loss Function Configurations
################################################################################
@dataclass
class LossConfig:
    """Model configuration."""
    pass

@dataclass
class CrossEntropyLossConfig(LossConfig):
    """Cross-entropy loss configuration."""
    seq_loss_weight: float = None
    struct_loss_weight: float = None

    # Cross-entropy loss function: which elements should contribute to the loss?
    # "masked": only masked positions
    # "non_beospank": all non-BOS/EOS/PAD/UNK positions, including masks
    # "non_special": all non-special tokens.  Special includes BOS, EOS, PAD, UNK, MASK.
    loss_elements:    str = None

    def __post_init__(self):
        assert self.seq_loss_weight is not None
        assert self.struct_loss_weight is not None
        assert self.loss_elements in ('masked', 'non_beospank', 'non_special')

@dataclass
class KabschRMSDLossConfig(LossConfig):
    """Kabsch RMSD loss configuration."""
    # Cross-entropy loss function: which elements should contribute to the loss?
    # "masked": only masked positions
    # "non_beospank": all non-BOS/EOS/PAD/UNK positions, including masks
    # "non_special": all non-special tokens.  Special includes BOS, EOS, PAD, UNK, MASK.
    # 'non_masked': all positions excluding mask.
    rmsd_elements:    str = None

    def __post_init__(self):
        assert self.rmsd_elements in ('masked', 'non_beospank', 'non_special', 'non_masked')
    


# Note: DiffusionConfig is a subclass of LossConfig, included in the "MaskConfig" section.


################################################################################
# Masking Configurations
################################################################################
@dataclass
class MaskConfig:
    """Noise configuration."""
    pass

@dataclass
class SimpleMaskConfig(MaskConfig):
    """Simple noise configuration."""
    mask_prob_seq:        float = None
    mask_prob_struct:     float = None

@dataclass
class ComplexMaskConfig(MaskConfig):
    """Complex noise configuration."""
    pass

@dataclass
class NoMaskConfig(MaskConfig):
    pass

@dataclass
class DiffusionConfig(MaskConfig, LossConfig):
    """Discrete diffusion configuration."""
    # Noise schedule parameters
    noise_schedule:        str  # Type of noise schedule ("linear", "inverted_u", or "uniform")
    sigma_min:             float  # Minimum noise level
    sigma_max:             float  # Maximum noise level
    num_timesteps:         int  # Number of discrete timesteps for training

    def __post_init__(self):
        assert self.noise_schedule in ('linear', 'inverted_u', 'uniform')
        assert self.sigma_min > 0
        assert self.sigma_max > 0
        assert self.num_timesteps > 0


################################################################################
# Transformer Block Configurations
################################################################################
@dataclass
class BlockConfig():
    """Transformer block configuration."""
    pass

@dataclass
class SelfConsensusConfig(BlockConfig):
    """SelfConsensus configuration."""
    # Consensus-specific parameters
    consensus_num_iterations:       int   # Number of consensus gradient iterations
    consensus_connectivity_type:    str   # "local_window" or "top_w"
    consensus_w:                    int   # Window size for local_window, or w value for top_w
    consensus_r:                    int   # Rank of Lambda_ij matrices
    consensus_edge_hidden_dim:      int   # Hidden dim for edge networks

    def __post_init__(self):
        assert self.consensus_num_iterations > 0
        assert self.consensus_connectivity_type in ('local_window', 'top_w')
        assert self.consensus_w > 0
        assert self.consensus_r > 0
        assert self.consensus_edge_hidden_dim > 0

@dataclass
class ReflexiveAttentionConfig(BlockConfig):
    """Reflexive attention configuration."""
    pass

@dataclass
class SelfAttentionConfig(BlockConfig):
    pass

@dataclass
class GeometricAttentionConfig(BlockConfig):
    pass

class ConfigurationError(Exception):
    def __init__(self, message):
        self.message = message
        
    def __str__(self):
        return self.message