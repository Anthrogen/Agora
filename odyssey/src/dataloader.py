import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from typing import Tuple
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from odyssey.src.tokenizer import SequenceTokenizer, StructureTokenizer, CoordinatesTokenizer
import math
from abc import abstractmethod

from odyssey.src.configurations import DiffusionMaskConfig, SimpleMaskConfig, ComplexMaskConfig, NoMaskConfig

def worker_init_fn(worker_id):
    """Initialize each worker with a deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --------------------------------------------------------------------------- #
#  Noise Schedules for complex masking                                        #
# --------------------------------------------------------------------------- #
_DEFAULT_MIN_UNMASKED = {'seq': 0, 'struct': 0, 'coords': 0}
#TODO: make min_unmasked positional in base class constructor MaskingDataLoader.
def _get_noise_levels(s_min, s_max, T, schedule_type="linear"):
    """Generate instantaneous and cumulative noise levels for discrete diffusion.
    
    Args:
        s_min: Minimum noise level (sigma_min)
        s_max: Maximum noise level (sigma_max)
        T: Number of timesteps
        schedule_type: "linear", "inverted_u", or "uniform"
        
    Returns:
        inst_noise_levels: Tensor of shape (T,) with instantaneous noise at each timestep
        cumulative_noise_levels: Tensor of shape (T,) with cumulative noise up to each timestep
    """
    t = torch.arange(T, dtype=torch.float32)
    normalized_t = t / (T - 1) if T > 1 else torch.zeros_like(t)
    
    if schedule_type == "linear":
        # Linear schedule: σ(t) = σ_min + (σ_max - σ_min) * t/T
        inst_noise_levels = s_min + (s_max - s_min) * normalized_t
        
        # Cumulative noise: ∫_0^t σ(s) ds 
        # For linear schedule σ(s) = σ_min + (σ_max - σ_min) * s
        # ∫_0^t σ(s) ds = σ_min * t + 0.5 * (σ_max - σ_min) * t^2
        cumulative_noise_levels = s_min * normalized_t + 0.5 * (s_max - s_min) * normalized_t**2
        
    elif schedule_type == "inverted_u":
        # Inverted-U schedule: concentrated training time distribution
        mask_probs = torch.zeros(T)
        
        for i in range(T):
            t_norm = i / (T - 1)  # 0 to 1
            # Transform uniform t_norm to create concentrated density
            # Use inverse sine to concentrate values in the middle
            if t_norm <= 0.5:
                # First half: map [0, 0.5] to [0.05, 0.5] with more density in middle
                local_t = t_norm * 2  # Scale to [0, 1]
                # Use sqrt to concentrate more values toward the end (middle of overall range)
                transformed = math.sqrt(local_t)
                mask_probs[i] = 0.05 + 0.45 * transformed
            else:
                # Second half: map [0.5, 1] to [0.5, 0.95] with more density in middle  
                local_t = (t_norm - 0.5) * 2  # Scale to [0, 1]
                # Use (1 - sqrt(1 - t)) to concentrate more values toward the beginning (middle of overall range)
                transformed = 1 - math.sqrt(1 - local_t)
                mask_probs[i] = 0.5 + 0.45 * transformed
        
        # Convert to cumulative noise levels
        cumulative_noise_levels = -torch.log(1 - mask_probs + 1e-8)
        
        # Compute instantaneous noise levels as derivatives
        inst_noise_levels = torch.zeros_like(cumulative_noise_levels)
        inst_noise_levels[0] = cumulative_noise_levels[0]
        
        for i in range(1, T):
            dt = 1.0 / (T - 1)
            inst_noise_levels[i] = (cumulative_noise_levels[i] - cumulative_noise_levels[i-1]) / dt
        
        # Clamp instantaneous noise for return
        inst_noise_levels = torch.clamp(inst_noise_levels, s_min, s_max)
        

    elif schedule_type == "uniform":
        # Uniform schedule: equal time spent at all mask percentages
        # Linear progression from 5% to 95% mask probability
        mask_probs = 0.05 + 0.9 * normalized_t  # Maps [0,1] to [0.05, 0.95]
        
        # Convert to cumulative noise levels
        cumulative_noise_levels = -torch.log(1 - mask_probs + 1e-8)
        
        # Compute instantaneous noise as derivative of cumulative noise
        # d/dt[-log(1 - mask_probs)] = d/dt[-log(1 - (0.05 + 0.9*t))] = 0.9 / (1 - (0.05 + 0.9*t))
        inst_noise_levels = 0.9 / (1 - mask_probs + 1e-8)
        
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}. Must be 'linear', 'inverted_u', or 'uniform'")
    
    return inst_noise_levels, cumulative_noise_levels

def _sample_betalinear30(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample valid rates from betalinear30 distribution. 80% Beta(3,9), 20% Uniform(0,1), avg ~30%"""
    mask_rates = torch.zeros(batch_size, device=device)
    use_beta = torch.rand(batch_size) < 0.8  # Choose distribution for each batch element
    
    beta_samples = torch.distributions.Beta(3.0, 9.0).sample((batch_size,)).to(device)  # Beta(3, 9) samples
    uniform_samples = torch.rand(batch_size, device=device)  # Uniform(0, 1) samples
    mask_rates = torch.where(use_beta.to(device), beta_samples, uniform_samples)  # Combine based on use_beta
    
    return torch.clamp(mask_rates, min=0.05, max=0.95)  # Clamp to avoid extreme values

def _sample_cosine(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample valid rates from cosine distribution using sin(u * π/2) for higher density near 1."""
    u = torch.rand(batch_size, device=device)  # Sample uniform values
    mask_rates = torch.sin(u * np.pi / 2)  # Apply sine transformation
    return torch.clamp(mask_rates, min=0.05, max=0.95)  # Clamp to avoid extreme values

def _get_training_dataloader(dataset, model_cfg, train_cfg, tracks, device, min_unmasked=_DEFAULT_MIN_UNMASKED, **kwargs):
    # Determine dataloader type based on mask_config type
    if isinstance(train_cfg.mask_config, SimpleMaskConfig):
        return SimpleDataLoader(dataset, model_cfg, train_cfg, tracks, device, min_unmasked=min_unmasked, **kwargs)
    elif isinstance(train_cfg.mask_config, ComplexMaskConfig):
        return ComplexDataLoader(dataset, model_cfg, train_cfg, tracks, device, min_unmasked=min_unmasked, **kwargs)
    elif isinstance(train_cfg.mask_config, DiffusionMaskConfig):
        return DiffusionDataLoader(dataset, model_cfg, train_cfg, tracks, device, min_unmasked=min_unmasked, **kwargs)
    elif isinstance(train_cfg.mask_config, NoMaskConfig):
        return NoMaskDataLoader(dataset, model_cfg, train_cfg, tracks, device, **kwargs)
    else:
        raise ValueError(f"Unknown mask config type: {type(train_cfg.mask_config)}. Expected SimpleMaskConfig, ComplexMaskConfig, DiffusionMaskConfig, or NoMaskConfig.")
    

class MaskingDataLoader(DataLoader):
    """DataLoader that applies masking during collation."""
    
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, fsq_encoder=None,  
                  propogate_coords_mask=True, min_unmasked=_DEFAULT_MIN_UNMASKED, **kwargs):

        self.device = device if device is not None else torch.device("cpu")
        self.L = model_cfg.max_len
            
        # TODO: get batch size directly from train_cfg.batch_size and apply, if we keep the train_cfg in the constructor argument list.
        # Override collate_fn with our custom one
        kwargs['collate_fn'] = self._mask_collate
        super().__init__(dataset, **kwargs)

    
        # Initialize tokenizers
        self.sequence_tokenizer = SequenceTokenizer(self.L)
        self.coordinates_tokenizer = CoordinatesTokenizer(self.L)
        self.structure_tokenizer = StructureTokenizer(self.L, fsq_encoder, reapply_bos_eos_pad=True)

        if tracks['struct']:
            assert fsq_encoder is not None

        self.propogate_coords_mask = propogate_coords_mask
        self.tracks = tracks
        self.min_unmasked = min_unmasked

    def _mask_collate(self, data):
        data = [item for item in data if item is not None]
        if len(data)==0: return None # Return None if all items were filtered out

        batch = MaskedBatch(data, self.tracks, self.sequence_tokenizer, self.structure_tokenizer, self.coordinates_tokenizer, 
                          device=self.device, min_unmasked=self.min_unmasked, generator=self.generator)
    
        self.sample_masks(batch)

        for track in [t for t in batch.tracks if (batch.tracks[t] and t != 'struct')]:
            #print(f"Iterative over track {track}")
            batch.apply_mask(track, batch.masks[track])

        if self.tracks['struct']:
            batch.apply_mask('struct', batch.masks['coords'])

        return batch
    
    @abstractmethod
    def sample_masks(self, batch):
        raise NotImplementedError("Subclasses must implement this method")

    
class SimpleDataLoader(MaskingDataLoader):
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, fsq_encoder=None, min_unmasked=_DEFAULT_MIN_UNMASKED, **kwargs):
        super(SimpleDataLoader, self).__init__(dataset, model_cfg, train_cfg, tracks, device,fsq_encoder=fsq_encoder, min_unmasked=min_unmasked,  **kwargs)

        assert isinstance(train_cfg.mask_config, SimpleMaskConfig)
        self.simple_mask_prob = {'seq': train_cfg.mask_config.mask_prob_seq, 'coords': train_cfg.mask_config.mask_prob_struct}

    def sample_masks(self, batch):
        for track in [t for t in batch.tracks if (batch.tracks[t] and t != 'struct')]:

            # Create masks with fixed probabilities
            mask = torch.rand(batch.B, batch.L, device=self.device) < self.simple_mask_prob[track]
            batch.masks[track] = mask.bool()
    
class ComplexDataLoader(MaskingDataLoader):
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, fsq_encoder=None, min_unmasked=_DEFAULT_MIN_UNMASKED, **kwargs):
        super(ComplexDataLoader, self).__init__(dataset, model_cfg, train_cfg, tracks, device, fsq_encoder=fsq_encoder, min_unmasked=min_unmasked, **kwargs)

    def sample_masks(self, batch):
        """
        Create masked tokens using complex masking with variable mask rates.
        
        Samples different mask rates for each protein in the batch for structure.
        Within each protein, masking is IID Bernoulli.
        """
        for track in [t for t in batch.tracks if (batch.tracks[t] and t != 'struct')]:

            # Sample mask rates for each protein in batch
            probs = _sample_cosine(batch.B, device=self.device)
            
            # Create masks with per-protein rates
            # Expand rates to [B, L] and sample IID Bernoulli
            probs = probs.unsqueeze(1).expand(batch.B, batch.L)
            
            mask = torch.rand(batch.B, batch.L, device=self.device) < probs
            batch.masks[track] = mask.bool()


class NoMaskDataLoader(MaskingDataLoader):
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, fsq_encoder=None, **kwargs):
        super(NoMaskDataLoader, self).__init__(dataset, model_cfg, train_cfg, tracks, device, fsq_encoder=fsq_encoder, **kwargs)
    
    def sample_masks(self, batch):
        for track in [t for t in batch.tracks if (batch.tracks[t] and t != 'struct')]:
            batch.masks[track] = torch.zeros(batch.B, batch.L, device=self.device).bool()

class DiffusionDataLoader(MaskingDataLoader):
    """DataLoader that applies discrete diffusion noise process during batch collation."""

    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, fsq_encoder=None, min_unmasked=_DEFAULT_MIN_UNMASKED, **kwargs):
        super(DiffusionDataLoader, self).__init__(dataset, model_cfg, train_cfg, tracks, device, fsq_encoder=fsq_encoder, min_unmasked=min_unmasked, **kwargs)

        # Store diffusion config
        self.diffusion_cfg = train_cfg.mask_config
        assert isinstance(self.diffusion_cfg, DiffusionMaskConfig)

        # Pre-compute noise levels
        self.inst_noise_levels, self.cumulative_noise_levels = _get_noise_levels(
            self.diffusion_cfg.sigma_min, 
            self.diffusion_cfg.sigma_max, 
            self.diffusion_cfg.num_timesteps,
            self.diffusion_cfg.noise_schedule
        )
        # Move to device
        self.inst_noise_levels = self.inst_noise_levels.to(self.device)
        self.cumulative_noise_levels = self.cumulative_noise_levels.to(self.device)

    def sample_masks(self, batch):
        # Sample timestep indices uniformly from [0, T-1]
        timestep_indices = torch.randint(0, self.diffusion_cfg.num_timesteps, (batch.B,), device=self.device)
        # Get corresponding noise levels
        cumulative_noise_level = self.cumulative_noise_levels[timestep_indices].unsqueeze(1)  # [B, 1]
        inst_noise_levels = self.inst_noise_levels[timestep_indices].unsqueeze(1)

        batch.metadata.update({'timestep_indices': timestep_indices, 'cumulative_noise': cumulative_noise_level, 'inst_noise': inst_noise_levels})

        for track in [t for t in batch.tracks if (batch.tracks[t] and t != 'struct')]:

            mask_prob = 1 - torch.exp(-cumulative_noise_level)
            mask_prob_expanded = mask_prob.expand(batch.B, batch.L)
            desired_masks = torch.rand(batch.B, batch.L, device=self.device) < mask_prob_expanded
            desired_masks = desired_masks.bool()

            batch.masks[track] = desired_masks

class MaskedBatch():
    def __init__(self, data, tracks, sequence_tokenizer, structure_tokenizer, coordinates_tokenizer,
                  device=None, min_unmasked={'seq': 0, 'struct': 0, 'coords': 0}, generator=None):
        
        """Custom collate function that applies discrete diffusion noise."""
        self.device = device if device is not None else torch.device("cpu")

        #TODO: get rid of 0.0, replace with a constant
        self.mask_tokens = {
            'seq': sequence_tokenizer.mapping['MASK'], 
            'struct': structure_tokenizer.mapping['MASK'],  # Now mapping exists in StructureTokenizer
            'coords': 0.0
        }

        self.tracks = tracks
        self.min_unmasked = min_unmasked
        self.generator = generator

        # Unpack batch from ProteinDataset
        self.masked_data = {'seq': None, 'struct': None, 'coords': None}
        self.unmasked_data = {'seq': None, 'struct': None, 'coords': None}
        self.masks = {'seq': None, 'struct': None, 'coords': None}
        self.beospank = {'seq': None, 'struct': None, 'coords': None}
        self.metadata = {}

        seq_list, coords_list, _ = zip(*data)
        
        # Stack lengths
        self.B = len(seq_list)
        self.L = sequence_tokenizer.full_length

        #########################################################################
        # Tokenize sequences and structures
        #########################################################################
        if tracks['seq']:
            seq_data = []
            for seq in seq_list:
                seq_data.append(sequence_tokenizer.tokenize(seq))
            self.unmasked_data['seq'], self.beospank['seq'] = zip(*seq_data)
            self.unmasked_data['seq'] = torch.stack(self.unmasked_data['seq'], dim=0).to(self.device)
            self.beospank['seq'] = torch.stack(self.beospank['seq'], dim=0).to(self.device).bool()

        if tracks['coords'] or tracks['struct']:
            tok = structure_tokenizer if tracks['struct'] else coordinates_tokenizer
            coords_data = []
            for coords in coords_list:
                coords_data.append(tok.tokenize(coords))
            
            # Unpack the 4-tuple returned by tokenize_from_coords
            coords_results = list(zip(*coords_data))
            self.unmasked_data['coords'] = torch.stack(coords_results[0], dim=0).to(self.device)
            self.beospank['coords'] = torch.stack(coords_results[1], dim=0).to(self.device).bool()
            if tracks['struct']:
                self.unmasked_data['struct'] = torch.stack(coords_results[2], dim=0).to(self.device)
                self.beospank['struct'] = torch.stack(coords_results[3], dim=0).to(self.device).bool()


    def apply_mask(self, track, desired_mask):
        # The following function will take as input self.masks and OVERWRITE these masks
        # This happens if a mask is sampled in a BOS/EOS/PAD position.
        #print(f"Applying mask to track {track}")
        self.masked_data[track], self.masks[track] = self.attempt_mask(self.unmasked_data[track], self.beospank[track], desired_mask, track)

    def attempt_mask(self, unmasked, beospank, desired_mask, track):
        #print(f"Unmasked Data: {unmasked}")
        B, L = unmasked.shape[0], unmasked.shape[1]

        actual_mask = desired_mask & ~beospank
        actual_mask = actual_mask.bool()

        #########################################################################
        # Here, we enforce that there is at least min_unmasked REAL RESIDUES (e.g. not BOS/EOS/PAD or MASK) per row.
        # This is extremely important for some applications -- such as KABSCH and for having nonsingular geometric matrices for geometric/reflexive attn.
        # If you want to 'skip over' this then set min_unmasked to 0 for all tracks.
        for row in range(B):
            # Count positions that are NOT masked AND NOT beospank
            real_residues = (~actual_mask[row] & ~beospank[row]).sum()
            if real_residues < self.min_unmasked[track]:
                num_to_unmask = self.min_unmasked[track] - real_residues
                
                # Find positions that are currently masked but NOT beospank (candidates for unmasking)
                candidate_positions = (actual_mask[row] & ~beospank[row]).nonzero(as_tuple=False).squeeze(-1)
                
                if candidate_positions.numel() < num_to_unmask:
                    raise ValueError(f"Need {self.min_unmasked[track]} unmasked residues, but only have {real_residues} residues in entire protein.")
                
                # Randomly select positions to unmask
                #TODO: use random number generator of the dataloader object.
                if self.generator is None:
                    print("Warning: No generator provided to MaskedBatch. Using default generator.")
                    perm = torch.randperm(candidate_positions.numel(), device=self.device)
                else:
                    # Use generator without device parameter to avoid device mismatch
                    perm = torch.randperm(candidate_positions.numel(), generator=self.generator).to(self.device)
                    
                positions_to_unmask = candidate_positions[perm[:num_to_unmask]]
                
                # Unmask these positions
                actual_mask[row, positions_to_unmask] = False
        
        #########################################################################
        # Now, actually apply the mask to the unmasked data.
        masked = unmasked.clone().to(self.device)
        extended_mask = actual_mask
        if track == 'coords':
            # If the track is coords, we have to extend out the mask due to the tensor dimension     
            H = unmasked.shape[2]       
            extended_mask = actual_mask.unsqueeze(-1).unsqueeze(-1).expand(B, L, H, 3)

        masked[extended_mask] = self.mask_tokens[track]
        return masked, actual_mask # return [B,L] actual mask even for 'coords' track
