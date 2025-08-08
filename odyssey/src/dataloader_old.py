import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from typing import Tuple
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from odyssey.src.tokenizer import SequenceTokenizer, StructureTokenizer, CoordinatesTokenizer, SS8Tokenizer, SASATokenizer, PLDDTTokenizer
from odyssey.src.tokenizer import OrthologousGroupsTokenizer, SemanticDescriptionTokenizer, DomainsTokenizer, CorruptionMode
import math
from abc import abstractmethod

from odyssey.src.configurations import DiffusionMaskConfig, SimpleMaskConfig, ComplexMaskConfig, NoMaskConfig

def worker_init_fn(worker_id):
    """Initialize each worker with a deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

_DEFAULT_MIN_UNMASKED = {'seq': 0, 'coords': 0}
# --------------------------------------------------------------------------- #
#  Noise Schedules for complex masking                                        #
# --------------------------------------------------------------------------- #
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
        min_cumulative_noise = -torch.log(torch.tensor(1 - 0.05))  # ≈ 0.051293
        cumulative_noise_levels = min_cumulative_noise + s_min * normalized_t + 0.5 * (s_max - s_min) * normalized_t**2
        
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

def _sample_sqrt(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample valid rates from sqrt distribution with linearly increasing PDF from 0 to 2 (favors high mask rates)."""
    u = torch.rand(batch_size, device=device)  # Sample uniform values
    mask_rates = torch.sqrt(u)  # Apply sqrt transformation for PDF f(x) = 2x
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
    
    def __init__(self, dataset, corruption_mode, model_cfg, train_cfg, tracks, device, autoencoder=None,  
                  propagate_coords_mask=True, min_unmasked=_DEFAULT_MIN_UNMASKED, **kwargs):

        self.device = device if device is not None else torch.device("cpu")
        self.L = model_cfg.max_len
        self.K = model_cfg.max_domains_per_residue
        self.G = model_cfg.max_len_orthologous_groups
        self.H = model_cfg.max_len_semantic_description

        # TODO: get batch size directly from train_cfg.batch_size and apply, if we keep the train_cfg in the constructor argument list.
        # Override collate_fn with our custom one
        kwargs['collate_fn'] = self._mask_collate
        
        # Extract generator from kwargs if provided
        generator = kwargs.pop('generator', None)
        
        super().__init__(dataset, **kwargs)
    
        # Initialize tokenizers
        self.sequence_tokenizer = SequenceTokenizer(self.L, min_unmasked=min_unmasked['seq'], generator=generator, corruption_mode=corruption_mode)
        self.coordinates_tokenizer = CoordinatesTokenizer(self.L, min_unmasked=min_unmasked['coords'], generator=generator)
        self.structure_tokenizer = StructureTokenizer(self.L, autoencoder, min_unmasked=min_unmasked['coords'], generator=generator, corruption_mode=corruption_mode)
        self.ss8_tokenizer = SS8Tokenizer(self.L, generator=generator, corruption_mode=corruption_mode)
        self.sasa_tokenizer = SASATokenizer(self.L, generator=generator, corruption_mode=corruption_mode)
        self.orthologous_groups_tokenizer = OrthologousGroupsTokenizer(self.G)
        self.semantic_description_tokenizer = SemanticDescriptionTokenizer(self.H)
        self.domains_tokenizer = DomainsTokenizer(self.L, self.K, generator=generator, corruption_mode=corruption_mode)
        self.plddt_tokenizer = PLDDTTokenizer(self.L, generator=generator, corruption_mode=corruption_mode)
        
        if tracks['struct']:
            assert autoencoder is not None

        self.propagate_coords_mask = propagate_coords_mask
        self.tracks = tracks
        self.min_unmasked = min_unmasked

    def _mask_collate(self, data):
        data = [item for item in data if item is not None]
        if len(data)==0: return None # Return None if all items were filtered out

        batch_len = len(data)

        masks, metadata = self.sample_masks(self.tracks, batch_len)

        if self.tracks['struct']: masks['struct'] = masks['coords']

        batch = MaskedBatch(data, masks, metadata, self.tracks, self.sequence_tokenizer, self.structure_tokenizer, self.coordinates_tokenizer, self.ss8_tokenizer, self.sasa_tokenizer, 
                              self.orthologous_groups_tokenizer, self.semantic_description_tokenizer, self.domains_tokenizer, self.plddt_tokenizer, device=self.device, min_unmasked=self.min_unmasked)

        return batch
    
    @abstractmethod
    def sample_masks(self, tracks, batch_len):
        raise NotImplementedError("Subclasses must implement this method")

    
class SimpleDataLoader(MaskingDataLoader):
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, autoencoder=None, min_unmasked=_DEFAULT_MIN_UNMASKED, **kwargs):
        super(SimpleDataLoader, self).__init__(dataset, CorruptionMode.MASK, model_cfg, train_cfg, tracks, device, autoencoder=autoencoder, min_unmasked=min_unmasked,  **kwargs)

        assert isinstance(train_cfg.mask_config, SimpleMaskConfig)
        # Use mask_prob_seq as default for sequence-like tracks, mask_prob_struct for structure-like tracks
        self.simple_mask_prob = {'seq': train_cfg.mask_config.mask_prob_seq, 'coords': train_cfg.mask_config.mask_prob_struct, 'ss8': train_cfg.mask_config.mask_prob_seq, 'sasa': train_cfg.mask_config.mask_prob_seq, 'plddt': train_cfg.mask_config.mask_prob_seq, 'domains': train_cfg.mask_config.mask_prob_seq}

    def sample_masks(self, tracks, batch_len):

        masks = {}
        for track in [t for t in tracks if (tracks[t] and t != 'struct' and t != 'orthologous_groups' and t != 'semantic_description')]:

            # Create masks with fixed probabilities
            mask = torch.rand(batch_len, self.L, device=self.device) < self.simple_mask_prob[track]
            masks[track] = mask.bool()

        return masks, None
    
class ComplexDataLoader(MaskingDataLoader):
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, autoencoder=None, min_unmasked=_DEFAULT_MIN_UNMASKED, **kwargs):
        super(ComplexDataLoader, self).__init__(dataset, CorruptionMode.MASK, model_cfg, train_cfg, tracks, device, autoencoder=autoencoder, min_unmasked=min_unmasked, **kwargs)

    def sample_masks(self, tracks, batch_len):
        """
        Create masked tokens using complex masking with variable mask rates.
        
        Samples different mask rates for each protein in the batch for structure.
        Within each protein, masking is IID Bernoulli.
        """
        masks = {}
        metadata = {}
        for track in [t for t in tracks if (tracks[t] and t != 'struct' and t != 'orthologous_groups' and t != 'semantic_description')]:

            # Sample mask rates for each protein in batch
            if track == 'seq': probs = _sample_betalinear30(batch_len, device=self.device)
            elif track == 'coords': probs = _sample_cosine(batch_len, device=self.device)
            else: probs = _sample_sqrt(batch_len, device=self.device)
            
            # Create masks with per-protein rates
            # Expand rates to [B, L] and sample IID Bernoulli
            probs_full = probs.unsqueeze(1).expand(batch_len, self.L)
            
            mask = torch.rand(batch_len, self.L, device=self.device) < probs_full
            masks[track] = mask.bool()

            metadata[track] = probs

        return masks, metadata


class NoMaskDataLoader(MaskingDataLoader):
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, autoencoder=None, **kwargs):
        super(NoMaskDataLoader, self).__init__(dataset, CorruptionMode.MASK, model_cfg, train_cfg, tracks, device, autoencoder=autoencoder, **kwargs)
    
    def sample_masks(self, tracks, batch_len):
        masks = {}
        for track in [t for t in tracks if (tracks[t] and t != 'struct' and t != 'orthologous_groups' and t != 'semantic_description')]:
            masks[track] = torch.zeros(batch_len, self.L, device=self.device).bool()

        return masks, None


class DiffusionDataLoader(MaskingDataLoader):
    """DataLoader that applies discrete diffusion noise process during batch collation."""

    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, autoencoder=None, min_unmasked=_DEFAULT_MIN_UNMASKED, **kwargs):

        corruption_mode = CorruptionMode.MASK if train_cfg.mask_config.corruption_mode == "absorb" else CorruptionMode.UNIFORM

        super(DiffusionDataLoader, self).__init__(dataset, corruption_mode, model_cfg, train_cfg, tracks, device, autoencoder=autoencoder, min_unmasked=min_unmasked, **kwargs)

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

    def sample_masks(self, tracks, batch_len):
        # Sample timestep indices uniformly from [0, T-1]
        timestep_indices = torch.randint(0, self.diffusion_cfg.num_timesteps, (batch_len,), device=self.device)
        # Get corresponding noise levels
        cumulative_noise_level = self.cumulative_noise_levels[timestep_indices].unsqueeze(1)  # [B, 1]
        inst_noise_levels = self.inst_noise_levels[timestep_indices].unsqueeze(1)

        metadata = {'timestep_indices': timestep_indices, 'cumulative_noise': cumulative_noise_level, 'inst_noise': inst_noise_levels}
        masks = {}
        for track in [t for t in tracks if (tracks[t] and t != 'struct' and t != 'orthologous_groups' and t != 'semantic_description')]:

            mask_prob = 1 - torch.exp(-cumulative_noise_level)
            mask_prob_expanded = mask_prob.expand(batch_len, self.L)
            desired_masks = torch.rand(batch_len, self.L, device=self.device) < mask_prob_expanded
            masks[track] = desired_masks.bool()

        return masks, metadata


class MaskedBatch():
    def __init__(self, data, masks, metadata, tracks, sequence_tokenizer, structure_tokenizer, coordinates_tokenizer, ss8_tokenizer, sasa_tokenizer, orthologous_groups_tokenizer,
                  semantic_description_tokenizer, domains_tokenizer, plddt_tokenizer, device=None, min_unmasked={'seq': 0, 'coords': 0}):
        
        """Custom collate function that applies discrete diffusion noise."""
        self.device = device if device is not None else torch.device("cpu")
        self.tracks = tracks
        self.min_unmasked = min_unmasked

        # Unpack batch from ProteinDataset
        self.masked_data = {'seq': None, 'struct': None, 'coords': None, 'ss8': None, 'sasa': None, 'domains': None, 'plddt': None}
        self.unmasked_data = {'seq': None, 'struct': None, 'coords': None, 'ss8': None, 'sasa': None, 'orthologous_groups': None, 'semantic_description': None, 'domains': None, 'plddt': None}
        self.beospank = {'seq': None, 'struct': None, 'coords': None, 'ss8': None, 'sasa': None, 'orthologous_groups': None, 'semantic_description': None, 'domains': None, 'plddt': None}
        
        self.metadata = metadata if metadata is not None else {}
        self.masks = {'seq': None, 'struct': None, 'coords': None, 'ss8': None, 'sasa': None, 'domains': None, 'plddt': None}
        self.masks = masks

        seq_list, coords_list, ss8_list, sasa_list, orthologous_groups_list, semantic_description_list, domains_list, plddt_list, _ = zip(*data)
        
        # Stack lengths
        self.B = len(seq_list)
        self.L = sequence_tokenizer.full_length
        self.G = orthologous_groups_tokenizer.full_length
        self.H = semantic_description_tokenizer.full_length

        #########################################################################
        # Tokenize sequences, structures, and context-specific data
        #########################################################################

        if tracks['seq']:
            seq_data = []
            for idx, seq in enumerate(seq_list):
                seq_data.append(sequence_tokenizer.tokenize(seq, masks['seq'][idx]))
            self.unmasked_data['seq'], self.masked_data['seq'], self.beospank['seq'], self.masks['seq'] = zip(*seq_data)
            self.unmasked_data['seq'] = torch.stack(self.unmasked_data['seq'], dim=0).to(self.device)
            self.masked_data['seq'] = torch.stack(self.masked_data['seq'], dim=0).to(self.device)
            self.beospank['seq'] = torch.stack(self.beospank['seq'], dim=0).to(self.device).bool()
            self.masks['seq'] = torch.stack(self.masks['seq'], dim=0).to(self.device).bool()
            
        if tracks['coords'] or tracks['struct']:
            tok = structure_tokenizer if tracks['struct'] else coordinates_tokenizer
            coords_data = []
            for idx, coords in enumerate(coords_list):
                coords_data.append(tok.tokenize(coords.to(self.device), masks['coords'][idx]))
            
            # Unpack the 4-tuple returned by tokenize_from_coords
            coords_results = list(zip(*coords_data))
            self.unmasked_data['coords'] = torch.stack(coords_results[0], dim=0).to(self.device)
            self.masked_data['coords'] = torch.stack(coords_results[1], dim=0).to(self.device)
            self.beospank['coords'] = torch.stack(coords_results[2], dim=0).to(self.device).bool()
            self.masks['coords'] = torch.stack(coords_results[3], dim=0).to(self.device).bool()
            if tracks['struct']:
                self.unmasked_data['struct'] = torch.stack(coords_results[4], dim=0).to(self.device)
                self.masked_data['struct'] = torch.stack(coords_results[5], dim=0).to(self.device)
                self.beospank['struct'] = torch.stack(coords_results[6], dim=0).to(self.device).bool()
                self.masks['struct'] = torch.stack(coords_results[7], dim=0).to(self.device).bool()

        # Tokenize SS8 (secondary structure) data if enabled
        if tracks['ss8']:
            ss8_data = []
            for idx, ss8 in enumerate(ss8_list):
                ss8_data.append(ss8_tokenizer.tokenize(ss8, masks['ss8'][idx]))
            self.unmasked_data['ss8'], self.masked_data['ss8'], self.beospank['ss8'], self.masks['ss8'] = zip(*ss8_data)
            self.unmasked_data['ss8'] = torch.stack(self.unmasked_data['ss8'], dim=0).to(self.device)
            self.masked_data['ss8'] = torch.stack(self.masked_data['ss8'], dim=0).to(self.device)
            self.beospank['ss8'] = torch.stack(self.beospank['ss8'], dim=0).to(self.device).bool()
            self.masks['ss8'] = torch.stack(self.masks['ss8'], dim=0).to(self.device).bool()

        # Tokenize SASA (solvent accessible surface area) data if enabled
        if tracks['sasa']:
            sasa_data = []
            for idx, sasa in enumerate(sasa_list):
                sasa_data.append(sasa_tokenizer.tokenize(sasa, masks['sasa'][idx]))
            self.unmasked_data['sasa'], self.masked_data['sasa'], self.beospank['sasa'], self.masks['sasa'] = zip(*sasa_data)
            self.unmasked_data['sasa'] = torch.stack(self.unmasked_data['sasa'], dim=0).to(self.device)
            self.masked_data['sasa'] = torch.stack(self.masked_data['sasa'], dim=0).to(self.device)
            self.beospank['sasa'] = torch.stack(self.beospank['sasa'], dim=0).to(self.device).bool()
            self.masks['sasa'] = torch.stack(self.masks['sasa'], dim=0).to(self.device).bool()

        # Tokenize orthologous groups data if enabled
        if tracks['orthologous_groups']:
            orthologous_groups_data = []
            for orthologous_groups in orthologous_groups_list:
                orthologous_groups_data.append(orthologous_groups_tokenizer.tokenize(orthologous_groups))
            self.unmasked_data['orthologous_groups'], self.beospank['orthologous_groups'] = zip(*orthologous_groups_data)
            self.unmasked_data['orthologous_groups'] = torch.stack(self.unmasked_data['orthologous_groups'], dim=0).to(self.device)
            self.beospank['orthologous_groups'] = torch.stack(self.beospank['orthologous_groups'], dim=0).to(self.device).bool()

        # Tokenize semantic description data if enabled
        if tracks['semantic_description']:
            semantic_description_data = []
            for semantic_description in semantic_description_list:
                semantic_description_data.append(semantic_description_tokenizer.tokenize(semantic_description))
            self.unmasked_data['semantic_description'], self.beospank['semantic_description'] = zip(*semantic_description_data)
            self.unmasked_data['semantic_description'] = torch.stack(self.unmasked_data['semantic_description'], dim=0).to(self.device)
            self.beospank['semantic_description'] = torch.stack(self.beospank['semantic_description'], dim=0).to(self.device).bool()

        # Tokenize domains data if enabled
        if tracks['domains']:
            domains_data = []
            for idx, domains in enumerate(domains_list):
                domains_data.append(domains_tokenizer.tokenize(domains, masks['domains'][idx]))
            self.unmasked_data['domains'], self.masked_data['domains'], self.beospank['domains'], self.masks['domains'] = zip(*domains_data)
            self.unmasked_data['domains'] = torch.stack(self.unmasked_data['domains'], dim=0).to(self.device)
            self.masked_data['domains'] = torch.stack(self.masked_data['domains'], dim=0).to(self.device)
            self.beospank['domains'] = torch.stack(self.beospank['domains'], dim=0).to(self.device).bool()
            self.masks['domains'] = torch.stack(self.masks['domains'], dim=0).to(self.device).bool()

        # Tokenize pLDDT data if enabled
        if tracks['plddt']:
            plddt_data = []
            for idx, plddt in enumerate(plddt_list):
                plddt_data.append(plddt_tokenizer.tokenize(plddt, masks['plddt'][idx]))
            self.unmasked_data['plddt'], self.masked_data['plddt'], self.beospank['plddt'], self.masks['plddt'] = zip(*plddt_data)
            self.unmasked_data['plddt'] = torch.stack(self.unmasked_data['plddt'], dim=0).to(self.device)
            self.masked_data['plddt'] = torch.stack(self.masked_data['plddt'], dim=0).to(self.device)
            self.beospank['plddt'] = torch.stack(self.beospank['plddt'], dim=0).to(self.device).bool()
            self.masks['plddt'] = torch.stack(self.masks['plddt'], dim=0).to(self.device).bool()