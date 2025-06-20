import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple
from src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from src.data_util.tokenizer_bos_eos_pad import SequenceTokenizer, StructureTokenizer, CoordinatesTokenizer
import math
from abc import abstractmethod

# --------------------------------------------------------------------------- #
#  Noise Schedules for complex masking                                        #
# --------------------------------------------------------------------------- #
def _get_noise_levels(s_min, s_max, T):
    """Generate instantaneous and cumulative noise levels for discrete diffusion.
    
    Args:
        s_min: Minimum noise level (sigma_min)
        s_max: Maximum noise level (sigma_max)
        T: Number of timesteps
        
    Returns:
        inst_noise_levels: Tensor of shape (T,) with instantaneous noise at each timestep
        cumulative_noise_levels: Tensor of shape (T,) with cumulative noise up to each timestep
    """
    t = torch.arange(T, dtype=torch.float32)
    # Geometric schedule for instantaneous noise
    inst_noise_levels = s_min**(1-t/T) * s_max**(t/T)
    
    # Cumulative noise: integral of instantaneous noise
    # For geometric schedule: ∫_0^t σ(s) ds
    cumulative_noise_levels = (math.log(s_max/s_min)/T) * (inst_noise_levels - inst_noise_levels[0])

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

def _get_training_dataloader(dataset, model_cfg, train_cfg, tracks, device=None, diffusion_cfg=None, **kwargs):
    #TODO: replace this with a constructor that simply examines the MaskCfg object of the TrainingConfig object
    if train_cfg.masking_strategy == "simple":
        return SimpleDataLoader(dataset, model_cfg, train_cfg, tracks, device=device, **kwargs)
    elif train_cfg.masking_strategy == "complex":
        return ComplexDataLoader(dataset, model_cfg, train_cfg, tracks, device=device, **kwargs)
    elif train_cfg.masking_strategy == "discrete_diffusion":
        assert diffusion_cfg is not None, "Diffusion config is required for discrete diffusion"
        return DiffusionDataLoader(dataset, model_cfg, train_cfg, diffusion_cfg, tracks, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown masking strategy: {train_cfg.masking_strategy}")
    

class MaskingDataLoader(DataLoader):
    """DataLoader that applies masking during collation."""
    
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device=None, 
                  propogate_coords_mask=False, **kwargs):

        self.device = device if device is not None else torch.device("cpu")
        self.L = model_cfg.max_len
            
        # Override collate_fn with our custom one
        kwargs['collate_fn'] = self._mask_collate
        super().__init__(dataset, **kwargs)

        # Initialize tokenizers
        self.sequence_tokenizer = SequenceTokenizer(self.L)
        self.structure_tokenizer = StructureTokenizer(self.L, model_cfg.fsq_encoder, reapply_bos_eos_pad=True)
        self.coordinates_tokenizer = CoordinatesTokenizer(self.L)

        self.propogate_coords_mask = propogate_coords_mask
        self.tracks = tracks

    def _mask_collate(self, data):
        

        batch = MaskedBatch(data, self.tracks, self.sequence_tokenizer, self.structure_tokenizer, self.coordinates_tokenizer, device=self.device)
    

        for t in self.tracks:
            if self.tracks[t]:
                mask = self.sample_masks(batch, t).bool()
                batch.apply_mask(t, mask)

        if self.propogate_coords_mask:
            batch.apply_mask('struct', batch.masks['coords'])

        return batch
    
    @abstractmethod
    def sample_masks(self, batch, track):
        raise NotImplementedError("Subclasses must implement this method")

    
class SimpleDataLoader(MaskingDataLoader):
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device=None, **kwargs):
        super(SimpleDataLoader, self).__init__(dataset, model_cfg, train_cfg, tracks, device=device, **kwargs)

        self.simple_mask_prob = {'seq': train_cfg.mask_prob_seq, 'struct': train_cfg.mask_prob_struct}

    def sample_masks(self, batch, track):
        # Create masks with fixed probabilities
        mask = torch.rand(batch.B, self.L, device=self.device) < self.simple_mask_prob[track]
        return mask.bool()
    
class ComplexDataLoader(MaskingDataLoader):
    def __init__(self, dataset, model_cfg, train_cfg, tracks,device=None, **kwargs):
        super(ComplexDataLoader, self).__init__(dataset, model_cfg, train_cfg, tracks, device=device, **kwargs)

    def sample_masks(self, batch, track):
        """
        Create masked tokens using complex masking with variable mask rates.
        
        Samples different mask rates for each protein in the batch for structure.
        Within each protein, masking is IID Bernoulli.
        """

        # Sample mask rates for each protein in batch
        probs = _sample_cosine(batch.B, device=self.device)
        
        # Create masks with per-protein rates
        # Expand rates to [B, L] and sample IID Bernoulli
        probs = probs.unsqueeze(1).expand(batch.B, batch.L)
        
        mask = torch.rand(batch.B, batch.L, device=self.device) < probs
        return mask.bool()


class NoMaskDataLoader(MaskingDataLoader):
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device=None, **kwargs):
        super(NoMaskDataLoader, self).__init__(dataset, model_cfg, train_cfg, tracks, device=device, **kwargs)
    
    def sample_masks(self, batch, track):
        return torch.zeros(batch.B, batch.L, device=self.device).bool()

class DiffusionDataLoader(MaskingDataLoader):
    """DataLoader that applies discrete diffusion noise process during batch collation."""

    def __init__(self, dataset, model_cfg, train_cfg, diffusion_cfg, tracks, device=None, **kwargs):
        super(DiffusionDataLoader, self).__init__(dataset, model_cfg, train_cfg, tracks, device=device, **kwargs)

        # Store diffusion config
        self.diffusion_cfg = diffusion_cfg

        # Pre-compute noise levels
        self.inst_noise_levels, self.cumulative_noise_levels = _get_noise_levels(
            diffusion_cfg.sigma_min, 
            diffusion_cfg.sigma_max, 
            diffusion_cfg.num_timesteps
        )
        # Move to device
        self.inst_noise_levels = self.inst_noise_levels.to(self.device)
        self.cumulative_noise_levels = self.cumulative_noise_levels.to(self.device)

    def sample_masks(self, batch, track):
        # Sample timestep indices uniformly from [0, T-1]
        timestep_indices = torch.randint(0, self.diffusion_cfg.num_timesteps, (batch.B,), device=self.device)
        # Get corresponding noise levels
        cumulative_noise_level = self.cumulative_noise_levels[timestep_indices].unsqueeze(1)  # [B, 1]
        inst_noise_levels = self.inst_noise_levels[timestep_indices].unsqueeze(1)

        mask_prob = 1-torch.exp(-cumulative_noise_level)
        mask_prob_expanded = mask_prob.expand(batch.B, batch.L)
        desired_masks = torch.rand(batch.B, batch.L, device=self.device) < mask_prob_expanded

        batch.mask_metadata[track] = {'timestep_indices': timestep_indices, 'cumulative_noise': cumulative_noise_level, 'inst_noise': inst_noise_levels}

        return desired_masks.bool()

class MaskedBatch():
    def __init__(self, data, tracks, sequence_tokenizer, structure_tokenizer, coordinates_tokenizer,
                  device=None):
        
        """Custom collate function that applies discrete diffusion noise."""
        self.device = device if device is not None else torch.device("cpu")

        #TODO: get rid of 0.0, replace with a constant
        self.mask_tokens = {
            'seq': sequence_tokenizer.mapping['MASK'], 
            'struct': structure_tokenizer.mapping['MASK'],  # Now mapping exists in StructureTokenizer
            'coords': 0.0
        }

        # Unpack batch from ProteinDataset
        self.masked_data = {'seq': None, 'struct': None, 'coords': None}
        self.unmasked_data = {'seq': None, 'struct': None, 'coords': None}
        self.masks = {'seq': None, 'struct': None, 'coords': None}
        self.beospad = {'seq': None, 'struct': None, 'coords': None}
        self.mask_metadata = {'seq': None, 'struct': None, 'coords': None}

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
            self.unmasked_data['seq'], self.beospad['seq'] = zip(*seq_data)
            self.unmasked_data['seq'] = torch.stack(self.unmasked_data['seq'], dim=0).to(self.device)
            self.beospad['seq'] = torch.stack(self.beospad['seq'], dim=0).to(self.device).bool()



        if tracks['coords'] or tracks['struct']:
            tok = structure_tokenizer if tracks['struct'] else coordinates_tokenizer
            coords_data = []
            for coords in coords_list:
                coords_data.append(tok.tokenize(coords))
            
            # Unpack the 4-tuple returned by tokenize_from_coords
            coords_results = list(zip(*coords_data))
            self.unmasked_data['coords'] = torch.stack(coords_results[0], dim=0).to(self.device)
            self.beospad['coords'] = torch.stack(coords_results[1], dim=0).to(self.device).bool()
            if tracks['struct']:
                self.unmasked_data['struct'] = torch.stack(coords_results[2], dim=0).to(self.device)
                self.beospad['struct'] = torch.stack(coords_results[3], dim=0).to(self.device).bool()
            

    def apply_mask(self, track, desired_mask):
        self.masked_data[track], self.masks[track] = self.attempt_mask(self.unmasked_data[track], self.beospad[track], desired_mask, track)

    def attempt_mask(self, unmasked, beospad, desired_mask, track, min_allowed_unmasked=1):
        B, L = unmasked.shape[0], unmasked.shape[1]

        if desired_mask.sum() < min_allowed_unmasked:
            assert False, 'handle this case better'
            #TODO: handle this case better.           

        actual_mask = desired_mask & ~beospad
        actual_mask = actual_mask.bool()
        
        masked = unmasked.clone().to(self.device)

        extended_mask = actual_mask
        if track == 'coords':
            H = unmasked.shape[2]
            extended_mask = actual_mask.unsqueeze(-1).unsqueeze(-1).expand(B, L, H, 3)

        masked[extended_mask] = self.mask_tokens[track]
        return masked, actual_mask # return [B,L] actual mask even for 'coords' track
