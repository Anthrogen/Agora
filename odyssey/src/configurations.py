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

    def is_equal(self, other: Config):
        raise NotImplementedError("Not today.")
        

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

    # def get_config_dict(self) -> dict:
    #     """Get the stored configuration dictionary."""
    #     if hasattr(self, '_config_dict'):
    #         return dict(self._config_dict)  # Return a copy
    #     else:
    #         return self.to_dict()

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
@register_config("transformer_cfg")
class TransformerConfig(Config):
    """Model architecture configuration."""
    d_model:                        int = None # 768  # Model dimensions
    n_heads:                        int = None  # 12
    n_layers:                       int = None  # 12
    max_len:                        int = None
    dropout:                        float = None   # Other architecture params
    ff_mult:                        int = None
    context_cfg:                    BlockConfig = None  # CrossConsensusConfig or CrossAttentionConfig for context injection
    first_block_cfg:                BlockConfig = None  # SelfConsensusConfig, GeometricAttentionConfig, ReflexiveAttentionConfig, or SelfAttentionConfig

    def __post_init__(self):
        self.ff_hidden_dim: int = self.d_model * self.ff_mult
        assert isinstance(self.d_model, int) and self.d_model > 0
        assert isinstance(self.n_heads, int) and self.n_heads > 0
        assert isinstance(self.n_layers, int) and self.n_layers > 0
        assert isinstance(self.max_len, int) and self.max_len > 0

        assert isinstance(self.first_block_cfg, BlockConfig)

@dataclass
class ModelConfig(Config):
    style:                             str = None
    autoencoder_path:                  str = None
    reference_model_seed:              int = None

    vocab_per_residue_path:            str = None
    vocab_global_path:                 str = None
    max_annotations_per_residue:       int = None
    max_len_global:                    int = None

    seq_absorb_token:                  int = SPECIAL_TOKENS.MASK.value + len(SEQUENCE_TOKENS) # Absorbing sequence state tokens (using MASK token index)
    struct_absorb_token:               int = SPECIAL_TOKENS.MASK.value + 4375 # Absorbing structure state tokens (using MASK token index)

    seq_vocab:                         int = len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS)  # Sequence tokens + special tokens
    struct_vocab:                      int = 4375 + len(SPECIAL_TOKENS)  # FSQ tokens + special tokens
    ss8_vocab:                         int = len(SS8_TOKENS) + len(SPECIAL_TOKENS)  # SS8 tokens + special tokens
    sasa_vocab:                        int = len(SASA_TOKENS) + len(SPECIAL_TOKENS)  # SASA tokens + special tokens
    plddt_vocab:                       int = len(PLDDT_TOKENS) + len(SPECIAL_TOKENS)  # pLDDT tokens + special tokens
    per_residue_annotation_vocab:      int = None  # per-residue annotation tokens + special tokens (set in post init)
    global_annotation_vocab:           int = None  # global annotation tokens + special tokens (set in post init)

    def initials(self):
        raise NotImplementedError("Subclasses must implement initials")

    # Getter methods for commonly accessed properties
    @property
    def max_len(self):
        """Get max_len from the appropriate transformer configuration."""
        raise NotImplementedError("Subclasses must implement max_len property")
    
    @property
    def first_block_cfg(self):
        """Get first_block_cfg from the appropriate transformer configuration."""
        raise NotImplementedError("Subclasses must implement first_block_cfg property")

    def __post_init__(self):
        # Load annotation vocabularies and set sizes
        self.per_residue_annotation_vocab = load_annotation_tokens(self.vocab_per_residue_path, PER_RESIDUE_ANNOTATION_TOKENS) + len(SPECIAL_TOKENS)
        self.global_annotation_vocab = load_annotation_tokens(self.vocab_global_path, GLOBAL_ANNOTATION_TOKENS) + len(SPECIAL_TOKENS)
        assert isinstance(self.max_annotations_per_residue, int) and self.max_annotations_per_residue > 0
        assert isinstance(self.max_len_global, int) and self.max_len_global > 0

        # TODO: get rid of
        assert self.seq_vocab > 0
        assert self.struct_vocab > 0
        assert self.ss8_vocab > 0
        assert self.sasa_vocab > 0
        assert self.plddt_vocab > 0
        assert self.per_residue_annotation_vocab > 0
        assert self.global_annotation_vocab > 0

        assert self.seq_absorb_token is not None
        assert self.struct_absorb_token is not None
        assert isinstance(self.reference_model_seed, int) 

@register_config("trunk_cfg")
@dataclass
class TrunkConfig(ModelConfig):
    """Trunk model configuration."""
    transformer_cfg:                   TransformerConfig = None

    def initials(self):
        return self.transformer_cfg.first_block_cfg.initials()

    @property
    def max_len(self):
        """Get max_len from transformer configuration."""
        return self.transformer_cfg.max_len
    
    @property
    def first_block_cfg(self):
        """Get first_block_cfg from transformer configuration."""
        return self.transformer_cfg.first_block_cfg

    def __post_init__(self):
        # Call parent's __post_init__ to set ff_hidden_dim and other attributes
        super().__post_init__()

        assert isinstance(self.transformer_cfg, TransformerConfig)
        
        assert self.style in ('mlm', 'discrete_diffusion')
        assert self.autoencoder_path is not None and os.path.exists(self.autoencoder_path)

@register_config("autoencoder_cfg")
@dataclass
class AutoencoderConfig(ModelConfig):
    """Autoencoder configuration."""
    encoder_cfg:                       FSQEncoderConfig = None
    decoder_cfg:                       FSQDecoderConfig = None

    def initials(self):
        # NOTE: this will print ONLY THE INITIALS of the encoder.  It's up to YOU to make sure you keep track of your decoders.
        return self.encoder_cfg.first_block_cfg.initials()

    @property
    def max_len(self):
        """Get max_len from encoder configuration."""
        return self.encoder_cfg.max_len
    
    @property
    def first_block_cfg(self):
        """Get first_block_cfg from encoder configuration."""
        return self.encoder_cfg.first_block_cfg

    @property
    def fsq_levels(self):
        """Get fsq_levels from encoder configuration."""
        return self.encoder_cfg.fsq_levels

    def __post_init__(self):
        super().__post_init__()
     
        assert isinstance(self.encoder_cfg, FSQEncoderConfig), f"Actual is {self.encoder_cfg}"
        assert isinstance(self.decoder_cfg, FSQDecoderConfig), f"Actual is {self.decoder_cfg}"

        assert self.style in ('stage_1', 'stage_2')
        if self.style == 'stage_2':
            assert self.autoencoder_path is not None and os.path.exists(self.autoencoder_path)

        must_match = ('max_len', 'latent_dim', 'fsq_levels')
        for mm in must_match:
            assert getattr(self.encoder_cfg, mm) == getattr(self.decoder_cfg, mm)

@register_config("encoder_cfg")
@dataclass 
class FSQEncoderConfig(TransformerConfig):
    """FSQ Encoder configuration."""
    latent_dim:                     int = None
    fsq_levels:                     str = None

    def __post_init__(self):
        super().__post_init__()
        self.fsq_levels = self.fsq_levels.split('x')
        self.fsq_levels = [int(str(l)) for l in self.fsq_levels]
        self.fsq_dim = len(self.fsq_levels)

    def make_copy(self):
        d = self.to_dict()
        codebook = d['fsq_levels']
        new_d = {key: deepcopy(getattr(self, key)) for key in d.keys()}
        new_d['fsq_levels'] = "".join(str(c) + "x" for c in codebook)[:-1]
        return FSQEncoderConfig(**new_d)

@register_config("decoder_cfg")
@dataclass
class FSQDecoderConfig(TransformerConfig):
    """FSQ Decoder configuration."""
    latent_dim:                     int = None
    fsq_levels:                     str = None

    def __post_init__(self):
        super().__post_init__()
        self.fsq_levels = self.fsq_levels.split('x')
        self.fsq_levels = [int(str(l)) for l in self.fsq_levels]
        self.fsq_dim = len(self.fsq_levels)

    def make_copy(self):
        d = self.to_dict()
        codebook = d['fsq_levels']
        new_d = {key: deepcopy(getattr(self, key)) for key in d.keys()}
        new_d['fsq_levels'] = "".join(str(c) + "x" for c in codebook)[:-1]
        return FSQDecoderConfig(**new_d)

@dataclass
class SchedulerConfig(Config):
    """Scheduler configuration."""
    pass

@register_config("flat_scheduler_cfg")
@dataclass
class FlatSchedulerConfig(SchedulerConfig):
    """Flat scheduler configuration."""
    learning_rate:              float = None

    def __post_init__(self):
        assert self.learning_rate is not None and self.learning_rate > 0

@register_config("warmup_decay_scheduler_cfg")
@dataclass
class WarmupDecaySchedulerConfig(SchedulerConfig):
    """Warmup decay scheduler configuration."""
    base_learning_rate:              float = None
    min_learning_rate:               float = None
    num_epochs_decay:                int = None
    num_epochs_warmup:               int = None

    def __post_init__(self):
        assert self.base_learning_rate is not None and isinstance(self.base_learning_rate, (int, float)) and self.base_learning_rate > 0
        assert self.min_learning_rate is not None and isinstance(self.min_learning_rate, (int, float)) and self.min_learning_rate > 0
        assert self.num_epochs_decay is not None and isinstance(self.num_epochs_decay, int) and self.num_epochs_decay > 0
        assert self.num_epochs_warmup is not None and isinstance(self.num_epochs_warmup, int) and self.num_epochs_warmup > 0
        assert self.base_learning_rate >= self.min_learning_rate

@register_config("decay_scheduler_cfg")
@dataclass
class DecaySchedulerConfig(SchedulerConfig):
    """Decay scheduler configuration."""
    base_learning_rate:              float = None
    min_learning_rate:               float = None
    num_epochs_decay:                int = None

    def __post_init__(self):
        assert self.base_learning_rate is not None and isinstance(self.base_learning_rate, (int, float)) and self.base_learning_rate > 0
        assert self.min_learning_rate is not None and isinstance(self.min_learning_rate, (int, float)) and self.min_learning_rate > 0
        assert self.num_epochs_decay is not None and isinstance(self.num_epochs_decay, int) and self.num_epochs_decay > 0
        assert self.base_learning_rate >= self.min_learning_rate

@register_config("training_cfg")
@dataclass
class TrainingConfig(Config):
    """Training process configuration."""
    # Model types should be in models configuration...
    batch_size:                   int = None # Training hyperparameters
    max_epochs:                   int = None
    checkpoint_freq:              Optional[int] = None  # Save checkpoints every N steps (if None, once per epoch)
    max_steps_val:                Optional[int] = None  # Max validation steps (if None, all validation batches)
    optim_schedule_config:        SchedulerConfig = None
    mask_config:                  MaskConfig = None
    loss_config:                  LossConfig = None
    data_dir:                     str = None  # Data paths

    # Optional, the path to the .pt checkpoint to jump start the training from.
    #  If none, the training will cold-start.
    jump_start:                   str = None 
    checkpoint_dir:               str = None   # Checkpointing

    def __post_init__(self):
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.max_epochs, int) and self.max_epochs > 0
        assert self.checkpoint_freq is None or (isinstance(self.checkpoint_freq, int) and self.checkpoint_freq > 0)
        assert self.max_steps_val is None or (isinstance(self.max_steps_val, int) and self.max_steps_val > 0)
        assert isinstance(self.optim_schedule_config, SchedulerConfig)
        assert isinstance(self.mask_config, MaskConfig)
        assert isinstance(self.loss_config, LossConfig)

        assert self.data_dir is not None and os.path.exists(self.data_dir), f"Data directory {self.data_dir} does not exist."
        assert not isinstance(self.checkpoint_dir, list) and ',' not in str(self.checkpoint_dir) and ';' not in str(self.checkpoint_dir), f"Multiple checkpoint directories not allowed: {self.checkpoint_dir}"
        assert self.checkpoint_dir is not None and os.path.exists(self.checkpoint_dir), f"Checkpoint directory {self.checkpoint_dir} does not exist."

        assert self.jump_start is None or os.path.exists(self.jump_start)
        
class ConfigurationError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
