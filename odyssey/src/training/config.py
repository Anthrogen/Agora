"""
Unified configuration classes for training.

This module consolidates all configuration dataclasses used across different
training scripts to ensure consistency and reduce code duplication.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS


@dataclass
class BaseModelConfig:
    """Base model architecture configuration shared across all models."""
    # Core transformer parameters
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    seq_vocab: int = len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS)
    struct_vocab: int = 4375 + len(SPECIAL_TOKENS)
    max_len: int = 2048
    dropout: float = 0.1
    ff_mult: int = 4
    
    # Model type for identification
    model_type: Optional[str] = None
    
    @property
    def ff_hidden_dim(self) -> int:
        """Calculate feedforward hidden dimension."""
        return self.d_model * self.ff_mult
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.d_model > 0, "d_model must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


@dataclass
class ConsensusConfig:
    """Configuration for Self-Consensus attention mechanism."""
    consensus_num_iterations: int = 1
    consensus_connectivity_type: str = "local_window"  # "local_window" or "top_w"
    consensus_w: int = 2  # Window size or top-w value
    consensus_r: int = 24  # Rank of Lambda_ij matrices
    consensus_edge_hidden_dim: int = 12  # Hidden dim for edge networks
    
    def __post_init__(self):
        """Validate consensus configuration."""
        assert self.consensus_connectivity_type in ["local_window", "top_w"]
        assert self.consensus_num_iterations > 0
        assert self.consensus_w > 0
        assert self.consensus_r > 0
        assert self.consensus_edge_hidden_dim > 0


@dataclass
class FSQModelConfig(BaseModelConfig, ConsensusConfig):
    """Configuration for FSQ (Finite Scalar Quantization) models."""
    # FSQ-specific parameters
    fsq_dim: int = 5
    fsq_levels: List[int] = field(default_factory=lambda: [7, 5, 5, 5, 5])
    latent_dim: int = 32
    stage: str = "stage_1"  # "stage_1" or "stage_2"
    fsq_encoder: Optional[Any] = None  # Will hold the actual encoder
    
    def __post_init__(self):
        """Validate FSQ configuration."""
        super().__post_init__()
        assert self.fsq_dim == len(self.fsq_levels), "fsq_dim must match length of fsq_levels"
        assert self.stage in ["stage_1", "stage_2"], "stage must be 'stage_1' or 'stage_2'"
        assert self.latent_dim > 0, "latent_dim must be positive"


@dataclass
class TransformerModelConfig(BaseModelConfig, ConsensusConfig):
    """Configuration for standard Transformer models."""
    pass


@dataclass
class BaseTrainingConfig:
    """Base training configuration shared across all training scripts."""
    # Training hyperparameters
    batch_size: int = 32
    max_epochs: int = 100
    learning_rate: float = 1e-4
    num_iter: int = 1  # Number of training iterations
    
    # Data paths
    data_dir: str = "sample_data/1k"
    csv_file: str = "sample_data/1k.csv"
    checkpoint_dir: str = "checkpoints"
    
    # Loss weights
    seq_loss_weight: float = 1.0
    struct_loss_weight: float = 1.0
    
    # Model and masking strategy
    model_types: List[str] = field(default_factory=lambda: ["SA"])
    masking_strategy: str = "simple"  # "simple", "complex", or "discrete_diffusion"
    
    # CE loss configuration
    ce_loss_function_elements: str = "masked"  # "masked" or "non_beospank"
    
    # Random seed for reproducibility
    reference_model_seed: int = 1234
    
    # Validation split
    val_split: float = 0.1
    
    def __post_init__(self):
        """Validate training configuration."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.max_epochs > 0, "max_epochs must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 < self.val_split < 1, "val_split must be between 0 and 1"
        assert self.masking_strategy in ["simple", "complex", "discrete_diffusion"]
        assert self.ce_loss_function_elements in ["masked", "non_beospank"]


@dataclass
class SimpleMaskingConfig:
    """Configuration for simple masking strategy."""
    mask_prob_seq: float = 0.2
    mask_prob_coords: float = 0.2
    
    def __post_init__(self):
        """Validate masking probabilities."""
        assert 0 <= self.mask_prob_seq <= 1, "mask_prob_seq must be between 0 and 1"
        assert 0 <= self.mask_prob_coords <= 1, "mask_prob_coords must be between 0 and 1"


@dataclass
class ComplexMaskingConfig:
    """Configuration for complex masking strategy."""
    mask_prob: float = 0.2
    noise_schedule: str = "linear"  # "linear" or "inverted_u"
    lambda_val: float = 1.0
    
    def __post_init__(self):
        """Validate complex masking configuration."""
        assert 0 <= self.mask_prob <= 1, "mask_prob must be between 0 and 1"
        assert self.noise_schedule in ["linear", "inverted_u"]
        assert self.lambda_val > 0, "lambda_val must be positive"


@dataclass
class DiffusionConfig:
    """Configuration for discrete diffusion training."""
    # Noise schedule parameters
    noise_schedule: str = "linear"  # "linear", "inverted_u", or "uniform"
    sigma_min: float = 0.31
    sigma_max: float = 5.68
    num_timesteps: int = 100
    
    # Absorbing state tokens
    seq_absorb_token: int = SPECIAL_TOKENS.MASK.value + len(SEQUENCE_TOKENS)
    struct_absorb_token: int = SPECIAL_TOKENS.MASK.value + 4375
    
    # Time indices for evaluation
    time_indices: Optional[List[int]] = None
    
    def __post_init__(self):
        """Validate diffusion configuration."""
        assert self.noise_schedule in ["linear", "inverted_u", "uniform"]
        assert 0 < self.sigma_min < self.sigma_max
        assert self.num_timesteps > 0


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""
    # Checkpoint patterns
    simple_checkpoint_pattern: str = "{model_type}_stage_{stage}_iter{iter}_simple.pt"
    complex_checkpoint_pattern: str = "{model_type}_stage_{stage}_iter{iter}_complex.pt"
    discrete_diffusion_checkpoint_pattern: str = "{model_type}_stage_{stage}_iter{iter}_discrete_diffusion.pt"
    fsq_encoder_pattern: str = "{model_type}_stage_{stage}_iter{iter}_{masking_strategy}.pt"
    
    # Save frequency
    save_every: int = 10
    save_best: bool = True
    
    # Resume training
    resume: bool = False
    resume_path: Optional[str] = None


@dataclass
class FSQTrainingConfig(BaseTrainingConfig):
    """Training configuration specific to FSQ models."""
    # FSQ-specific training parameters
    stage: str = "stage_1"
    
    # Stage-specific epochs
    stage_1_epochs: int = 70
    stage_2_epochs: int = 30
    
    @property
    def max_epochs(self) -> int:
        """Get max epochs based on stage."""
        if self.stage == "stage_1":
            return self.stage_1_epochs
        else:
            return self.stage_2_epochs


@dataclass
class TransformerTrainingConfig(BaseTrainingConfig):
    """Training configuration specific to Transformer models."""
    # Additional transformer training parameters
    warmup_steps: int = 1000
    weight_decay: float = 0.01


# Factory functions for creating configurations
def create_fsq_config(
    model_type: str = "SC",
    stage: str = "stage_1",
    masking_strategy: str = "simple",
    **kwargs
) -> FSQModelConfig:
    """Create FSQ model configuration with sensible defaults."""
    # Adjust layers based on stage
    n_layers = 3 if stage == "stage_1" else 10
    
    config = FSQModelConfig(
        model_type=model_type,
        stage=stage,
        n_layers=n_layers,
        d_model=kwargs.get("d_model", 128),
        n_heads=kwargs.get("n_heads", 1),
        **{k: v for k, v in kwargs.items() if k not in ["d_model", "n_heads"]}
    )
    return config


def create_transformer_config(
    model_type: str = "SA",
    **kwargs
) -> TransformerModelConfig:
    """Create Transformer model configuration with sensible defaults."""
    config = TransformerModelConfig(
        model_type=model_type,
        **kwargs
    )
    return config


def create_training_config(
    model_types: List[str],
    masking_strategy: str = "simple",
    is_fsq: bool = False,
    **kwargs
) -> BaseTrainingConfig:
    """Create appropriate training configuration based on model type."""
    if is_fsq:
        config = FSQTrainingConfig(
            model_types=model_types,
            masking_strategy=masking_strategy,
            **kwargs
        )
    else:
        config = TransformerTrainingConfig(
            model_types=model_types,
            masking_strategy=masking_strategy,
            **kwargs
        )
    return config