from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Type
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS, SS8_TOKENS, SASA_TOKENS, PLDDT_TOKENS
from odyssey.src.vocabulary import PER_RESIDUE_ANNOTATION_TOKENS, GLOBAL_ANNOTATION_TOKENS, load_annotation_tokens
import os
from copy import deepcopy

# Global registry to map config types to classes
CONFIG_REGISTRY: Dict[str, Type['Config']] = {}

def register_config(config_type: str):
    """Decorator to register configuration classes."""
    def decorator(cls):
        CONFIG_REGISTRY[config_type] = cls
        return cls
    return decorator

@dataclass
class Config:
    def to_dict(self) -> dict:
        """Convert configuration to dictionary, handling nested dataclasses."""
        # List of computed fields to exclude
        # Can we do this better with inheritance?
        computed_fields = {'ff_hidden_dim', 'fsq_dim'}
        
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith('_') or key in computed_fields:
                continue
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = list(value)  # Create a copy of the list
            elif isinstance(value, dict):
                result[key] = dict(value)  # Create a copy of the dict
            else:
                result[key] = value
        return result

    def make_copy(self):
        return deepcopy(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Recursively build configuration from dictionary with type/params structure."""
        processed_dict = {}
        config_type = next(iter(config_dict.keys()))
        constructor = CONFIG_REGISTRY[config_type]

        if config_dict[config_type] is None:
            return constructor()

        else:
            for key, value in config_dict[config_type].items():
                if isinstance(value, dict) and any(k in CONFIG_REGISTRY for k in value.keys()):
                    # This is a nested config - call from_dict on the Config class
                    processed_dict[key] = Config.from_dict(value)
                else:
                    processed_dict[key] = value
            return constructor(**processed_dict)

    def get_config_dict(self) -> dict:
        """Get the stored configuration dictionary."""
        if hasattr(self, '_config_dict'):
            return dict(self._config_dict)  # Return a copy
        else:
            return self.to_dict()

################################################################################
# Transformer Block Configurations
################################################################################
@dataclass
class BlockConfig():
    """Transformer block configuration."""
    def __str__(self):
        raise NotImplementedError("Subclasses must implement __str__")

    def initials(self):
        raise NotImplementedError("Subclasses must implement initials")
    
    def to_dict(self) -> dict:
        """Convert to dictionary with type information."""
        result = super().to_dict() if hasattr(super(), 'to_dict') else {}
        result.update({k: v for k, v in self.__dict__.items() if not k.startswith('_')})
        result['_block_type'] = self.__class__.__name__
        return result

@register_config("self_consensus_cfg")
@dataclass
class SelfConsensusConfig(BlockConfig):
    """SelfConsensus configuration."""
    # Consensus-specific parameters
    consensus_num_iterations:       int   # Number of consensus gradient iterations
    consensus_connectivity_type:    str   # "local_window" or "scored_window"
    consensus_w:                    int   # Window size for local_window, or w value for scored_window
    consensus_r:                    int   # Rank of Lambda_ij matrices
    consensus_edge_hidden_dim:      int   # Hidden dim for edge networks

    def __post_init__(self):
        assert self.consensus_num_iterations > 0
        assert self.consensus_connectivity_type in ('local_window', 'scored_window')
        assert self.consensus_w > 0
        assert self.consensus_r > 0
        assert self.consensus_edge_hidden_dim > 0

    def __str__(self): return "Self Consensus"
    def initials(self): return "SC"

@register_config("reflexive_attention_cfg")
@dataclass
class ReflexiveAttentionConfig(BlockConfig):
    """Reflexive attention configuration."""
    def __str__(self): return "Reflexive Attention"
    def initials(self): return "RA"

@register_config("self_attention_cfg")
@dataclass
class SelfAttentionConfig(BlockConfig):
    def __str__(self): return "Self Attention"
    def initials(self): return "SA"

@register_config("geometric_attention_cfg")
@dataclass
class GeometricAttentionConfig(BlockConfig):
    def __str__(self): return "Geometric Attention"
    def initials(self): return "GA"

@register_config("cross_consensus_cfg")
@dataclass
class CrossConsensusConfig(BlockConfig):
    """CrossConsensus configuration."""
    # Consensus-specific parameters (same as SelfConsensus)
    consensus_num_iterations:       int   # Number of consensus gradient iterations
    consensus_connectivity_type:    str   # "local_window" or "scored_window"
    consensus_w:                    int   # Window size for local_window, or w value for scored_window
    consensus_r:                    int   # Rank of Lambda_ij matrices
    consensus_edge_hidden_dim:      int   # Hidden dim for edge networks

    def __post_init__(self):
        assert self.consensus_num_iterations > 0
        assert self.consensus_connectivity_type in ('local_window', 'scored_window')
        assert self.consensus_w > 0
        assert self.consensus_r > 0
        assert self.consensus_edge_hidden_dim > 0

    def __str__(self): return "Cross Consensus"
    def initials(self): return "CC"

@register_config("cross_attention_cfg")
@dataclass
class CrossAttentionConfig(BlockConfig):
    """CrossAttention configuration."""
    def __str__(self): return "Cross Attention"
    def initials(self): return "CA"

################################################################################
# Training Loss Function Configurations
################################################################################
@dataclass
class LossConfig:
    """Model configuration."""
    def to_dict(self) -> dict:
        """Convert to dictionary with type information."""
        result = super().to_dict() if hasattr(super(), 'to_dict') else {}
        result.update({k: v for k, v in self.__dict__.items() if not k.startswith('_')})
        result['_loss_type'] = self.__class__.__name__
        return result

@register_config("cross_entropy_loss_cfg")
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

@register_config("score_entropy_loss_cfg")
@dataclass
class ScoreEntropyLossConfig(LossConfig):
    """Score entropy loss configuration for discrete diffusion."""
    seq_loss_weight: float = None
    struct_loss_weight: float = None
    
    def __post_init__(self):
        assert self.seq_loss_weight is not None
        assert self.struct_loss_weight is not None

@register_config("kabsch_rmsd_loss_cfg")
@dataclass
class KabschRMSDLossConfig(LossConfig):
    pass
    # """Kabsch RMSD loss configuration."""
    # # Kabsch RMSD loss function: which elements should contribute to the loss?
    # # "masked": only masked positions
    # # "non_beospank": all non-BOS/EOS/PAD/UNK positions, including masks
    # # "non_special": all non-special tokens.  Special includes BOS, EOS, PAD, UNK, MASK.
    # # 'non_masked': all positions excluding mask.
    # rmsd_elements:    str = None
    # def __post_init__(self):
    #     assert self.rmsd_elements in ('masked', 'non_beospank', 'non_special', 'non_masked')
    
################################################################################
# Masking Configurations
################################################################################
@dataclass
class MaskConfig:
    """Noise configuration."""
    def __str__(self): raise NotImplementedError("Subclasses must implement __str__")
    
    def to_dict(self) -> dict:
        """Convert to dictionary with type information."""
        result = super().to_dict() if hasattr(super(), 'to_dict') else {}
        result.update({k: v for k, v in self.__dict__.items() if not k.startswith('_')})
        result['_mask_type'] = self.__class__.__name__
        return result

@register_config("simple_mask_cfg")
@dataclass
class SimpleMaskConfig(MaskConfig):
    """Simple noise configuration."""
    mask_prob_seq:        float = None
    mask_prob_struct:     float = None
    def __str__(self): return "simple"

@register_config("complex_mask_cfg")
@dataclass
class ComplexMaskConfig(MaskConfig):
    """Complex noise configuration."""
    def __str__(self): return "complex"

@register_config("no_mask_cfg")
@dataclass
class NoMaskConfig(MaskConfig):
    def __str__(self): return "no_mask"

@register_config("diffusion_mask_cfg")  
@dataclass
class DiffusionMaskConfig(MaskConfig):
    """Discrete diffusion configuration."""
    # Noise schedule parameters
    noise_schedule:        str  # Type of noise schedule ("linear", "inverted_u", or "uniform")
    sigma_min:             float  # Minimum noise level
    sigma_max:             float  # Maximum noise level
    num_timesteps:         int  # Number of discrete timesteps for training
    corruption_mode:       str  # Type of corruption ("absorb", "uniform")

    def __post_init__(self):
        assert self.noise_schedule in ('linear', 'inverted_u', 'uniform')
        assert self.sigma_min > 0
        assert self.sigma_max > 0
        assert self.num_timesteps > 0
        assert self.corruption_mode in ('absorb', 'uniform')

    def __str__(self): return "discrete_diffusion"

@dataclass
class TransformerConfig(Config):
    """Model architecture configuration."""
    style:                          str = None
    d_model:                        int = None # 768  # Model dimensions
    n_heads:                        int = None  # 12
    n_layers:                       int = None  # 12
    max_len:                        int = None
    max_annotations_per_residue:    int = None
    max_len_global:                 int = None
    dropout:                        float = None   # Other architecture params
    ff_mult:                        int = None
    first_block_cfg:                BlockConfig = None  # SelfConsensusConfig, GeometricAttentionConfig, ReflexiveAttentionConfig, or SelfAttentionConfig
    context_cfg:                    BlockConfig = None  # CrossConsensusConfig or CrossAttentionConfig for SS8/SASA injection
    reference_model_seed:           int = None
    fsq_encoder_path:               str = None
    vocab_per_residue_path:         str = None
    vocab_global_path:              str = None

    # TODO: These need to go.  seq_vocab should come from voacbulary.py and struuct_vocab should come from the FSQEncoder object.
    seq_vocab:                      int = len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS)  # Sequence tokens + special tokens
    struct_vocab:                   int = 4375 + len(SPECIAL_TOKENS)  # FSQ tokens + special tokens
    ss8_vocab:                      int = len(SS8_TOKENS) + len(SPECIAL_TOKENS)  # SS8 tokens + special tokens
    sasa_vocab:                     int = len(SASA_TOKENS) + len(SPECIAL_TOKENS)  # SASA tokens + special tokens
    plddt_vocab:                    int = len(PLDDT_TOKENS) + len(SPECIAL_TOKENS)  # pLDDT tokens + special tokens
    per_residue_annotation_vocab:   int = None  # per-residue annotation tokens + special tokens (set in post init)
    global_annotation_vocab:        int = None  # global annotation tokens + special tokens (set in post init)

    def __post_init__(self):
        assert self.style in ('stage_1', 'stage_2', 'mlm', 'discrete_diffusion')
        self.ff_hidden_dim: int = self.d_model * self.ff_mult
        assert isinstance(self.d_model, int) and self.d_model > 0
        assert isinstance(self.n_heads, int) and self.n_heads > 0
        assert isinstance(self.n_layers, int) and self.n_layers > 0
        assert isinstance(self.max_len, int) and self.max_len > 0
        assert isinstance(self.max_annotations_per_residue, int) and self.max_annotations_per_residue > 0
        assert isinstance(self.max_len_global, int) and self.max_len_global > 0
        assert isinstance(self.first_block_cfg, BlockConfig)

        # Load annotation vocabularies and set sizes
        self.per_residue_annotation_vocab = load_annotation_tokens(self.vocab_per_residue_path, PER_RESIDUE_ANNOTATION_TOKENS) + len(SPECIAL_TOKENS)
        self.global_annotation_vocab = load_annotation_tokens(self.vocab_global_path, GLOBAL_ANNOTATION_TOKENS) + len(SPECIAL_TOKENS)

        # TODO: get rid of
        assert self.seq_vocab > 0
        assert self.struct_vocab > 0
        assert self.ss8_vocab > 0
        assert self.sasa_vocab > 0
        assert self.plddt_vocab > 0
        assert self.per_residue_annotation_vocab > 0
        assert self.global_annotation_vocab > 0
        
        # Store configuration as dictionary for safety
        self._config_dict = self.to_dict()


@register_config("trunk_cfg")
@dataclass
class TrunkConfig(TransformerConfig):
    """Trunk model configuration."""
    seq_absorb_token:                  int = SPECIAL_TOKENS.MASK.value + len(SEQUENCE_TOKENS) # Absorbing sequence state tokens (using MASK token index)
    struct_absorb_token:               int = SPECIAL_TOKENS.MASK.value + 4375 # Absorbing structure state tokens (using MASK token index)

    def __post_init__(self):
        # Call parent's __post_init__ to set ff_hidden_dim and other attributes
        super().__post_init__()
        
        assert self.style in ('mlm', 'discrete_diffusion')
        assert self.fsq_encoder_path is not None and os.path.exists(self.fsq_encoder_path)
        
        # Update stored dictionary with trunk-specific fields
        self._config_dict = self.to_dict()

@register_config("fsq_cfg")
@dataclass
class FSQConfig(TransformerConfig):
    """Model architecture configuration."""
    # Transformer parameters
    latent_dim:                     int = None # pre-quantized CONTINUOUS latent dimension.
    fsq_levels:                     str = None # codebook

    def __post_init__(self):
        # Call parent's __post_init__ to set ff_hidden_dim and other attributes
        super().__post_init__()

        self.fsq_levels = self.fsq_levels.split('x')
        # print(f"DEBUG: fsq_levels: {self.fsq_levels}")
        self.fsq_levels = [int(str(l)) for l in self.fsq_levels]

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
        
        # Update stored dictionary with FSQ-specific fields
        self._config_dict = self.to_dict()

    def make_copy(self):
        d = self.to_dict()
        codebook = d['fsq_levels']
        new_d = {key: deepcopy(getattr(self, key)) for key in d.keys()}
        new_d['fsq_levels'] = "".join(str(c) + "x" for c in codebook)[:-1]
        return FSQConfig(**new_d)

@register_config("training_cfg")
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
        assert isinstance(self.learning_rate, (int, float)) and self.learning_rate > 0
        assert isinstance(self.mask_config, MaskConfig)
        assert isinstance(self.loss_config, LossConfig)

        assert self.data_dir is not None and os.path.exists(self.data_dir), f"Data directory {self.data_dir} does not exist."
        assert not isinstance(self.checkpoint_dir, list) and ',' not in str(self.checkpoint_dir) and ';' not in str(self.checkpoint_dir), f"Multiple checkpoint directories not allowed: {self.checkpoint_dir}"
        assert self.checkpoint_dir is not None and os.path.exists(self.checkpoint_dir), f"Checkpoint directory {self.checkpoint_dir} does not exist."
        
        # Store configuration as dictionary for safety
        self._config_dict = self.to_dict()
        
class ConfigurationError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
