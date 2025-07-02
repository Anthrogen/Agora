from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Type
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
import os
import json

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
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Recursively build configuration from dictionary with type/params structure."""
        if 'type' in config_dict and 'params' in config_dict:
            config_type = config_dict['type']
            params = config_dict['params']
            
            # Get the appropriate class from registry
            if config_type not in CONFIG_REGISTRY:
                raise ValueError(f"Unknown config type: {config_type}. Available types: {list(CONFIG_REGISTRY.keys())}")
            
            config_class = CONFIG_REGISTRY[config_type]
            
            # Recursively process nested configs
            processed_params = {}
            for key, value in params.items():
                if isinstance(value, dict) and 'type' in value and 'params' in value:
                    # This is a nested config
                    processed_params[key] = Config.from_dict(value)
                else:
                    processed_params[key] = value
            
            return config_class(**processed_params)
        else:
            # Direct instantiation without type/params structure
            return cls(**config_dict)
    
    def get_config_dict(self) -> dict:
        """Get the stored configuration dictionary."""
        if hasattr(self, '_config_dict'):
            return dict(self._config_dict)  # Return a copy
        else:
            return self.to_dict()
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create configuration from dictionary. Override in subclasses for nested configs."""
        return cls(**config_dict)
    
    def save_to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        config_dict = self.get_config_dict()
        config_dict['_config_class'] = self.__class__.__name__
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @staticmethod
    def load_from_json(filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Remove the class name marker
        class_name = config_dict.pop('_config_class', 'Config')
        
        # Import all config classes to ensure they're available
        import odyssey.src.configurations as configs
        
        # Get the appropriate class
        config_class = getattr(configs, class_name, Config)
        
        # Handle nested configs if needed
        if 'first_block_cfg' in config_dict and isinstance(config_dict['first_block_cfg'], dict):
            block_dict = config_dict['first_block_cfg'].copy()
            block_class_name = block_dict.pop('_block_type', 'BlockConfig')
            block_class = getattr(configs, block_class_name, BlockConfig)
            config_dict['first_block_cfg'] = block_class(**block_dict)
        
        if 'mask_config' in config_dict and isinstance(config_dict['mask_config'], dict):
            mask_dict = config_dict['mask_config'].copy()
            mask_class_name = mask_dict.pop('_mask_type', 'MaskConfig')
            mask_class = getattr(configs, mask_class_name, MaskConfig)
            config_dict['mask_config'] = mask_class(**mask_dict)
        
        if 'loss_config' in config_dict and isinstance(config_dict['loss_config'], dict):
            loss_dict = config_dict['loss_config'].copy()
            loss_class_name = loss_dict.pop('_loss_type', 'LossConfig')
            loss_class = getattr(configs, loss_class_name, LossConfig)
            config_dict['loss_config'] = loss_class(**loss_dict)
        
        return config_class(**config_dict)

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

    def __post_init__(self):
        assert self.noise_schedule in ('linear', 'inverted_u', 'uniform')
        assert self.sigma_min > 0
        assert self.sigma_max > 0
        assert self.num_timesteps > 0

    def __str__(self): return "discrete_diffusion"

@dataclass
class TransformerConfig(Config):
    """Model architecture configuration."""
    style:                          str = None
    d_model:                        int = None # 768  # Model dimensions
    n_heads:                        int = None  # 12
    n_layers:                       int = None  # 12
    max_len:                        int = None
    dropout:                        float = None   # Other architecture params
    ff_mult:                        int = None
    first_block_cfg:                BlockConfig = None
    reference_model_seed:           int = None

    # TODO: These need to go.  seq_vocab should come from voacbulary.py and struuct_vocab should come from the FSQEncoder object.
    seq_vocab:                      int = None  # Sequence tokens + special tokens
    struct_vocab:                   int = None  # FSQ tokens + special tokens

    def __post_init__(self):
        assert self.style in ('stage_1', 'stage_2', 'mlm', 'discrete_diffusion')
        self.ff_hidden_dim: int = self.d_model * self.ff_mult
        assert isinstance(self.d_model, int) and self.d_model > 0
        assert isinstance(self.n_heads, int) and self.n_heads > 0
        assert isinstance(self.n_layers, int) and self.n_layers > 0
        assert isinstance(self.max_len, int) and self.max_len > 0
        assert isinstance(self.first_block_cfg, BlockConfig)

        # TODO: get rid of
        assert self.seq_vocab > 0
        assert self.struct_vocab > 0
        
        # Store configuration as dictionary for safety
        self._config_dict = self.to_dict()


@register_config("trunk_cfg")
@dataclass
class TrunkConfig(TransformerConfig):
    """Trunk model configuration."""
    fsq_encoder_path:                  str = None
    seq_absorb_token:                  int = None # Absorbing sequence state tokens (using MASK token index)
    struct_absorb_token:               int = None # Absorbing structure state tokens (using MASK token index)

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
    fsq_levels:                     list[int] = None # codebook
    fsq_encoder_path:               str = None

    def __post_init__(self):
        # Call parent's __post_init__ to set ff_hidden_dim and other attributes
        super().__post_init__()

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
        assert self.checkpoint_dir is not None and os.path.exists(self.checkpoint_dir), f"Checkpoint directory {self.checkpoint_dir} does not exist."
        
        # Store configuration as dictionary for safety
        self._config_dict = self.to_dict()
        
class ConfigurationError(Exception):
    def __init__(self, message):
        self.message = message
        
    def __str__(self):
        return self.message