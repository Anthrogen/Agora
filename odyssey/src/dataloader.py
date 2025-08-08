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
from threading import Thread
from queue import Queue, Empty

from odyssey.src.configurations import DiffusionMaskConfig, SimpleMaskConfig, ComplexMaskConfig, NoMaskConfig
from odyssey.src.masking_utils import _get_noise_levels, _sample_betalinear30, _sample_cosine, _sample_sqrt

# Import components from pipeline for DecoupledDataLoader
from odyssey.src.pipeline import (
    RawBatch, RawDataCache, StructureTokenCache,
    raw_collate_fn, CPUTokenizationStage, 
    GPUStructureTokenizationStage, TokenizationPipeline,
    CachedDataLoader
)

def worker_init_fn(worker_id):
    """Initialize each worker with a deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

_DEFAULT_MIN_UNMASKED = {'seq': 0, 'coords': 0}

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

class DecoupledDataLoader:
    """
    Base class for decoupled dataloaders. It decouples raw data fetching from
    tokenization using a cached raw loader and a tokenization pipeline.
    Subclasses must implement sample_masks(tracks, batch_len).
    """
    
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device,
                 autoencoder=None, batch_size=32, num_workers=4, cache_size=1000,
                 min_unmasked=_DEFAULT_MIN_UNMASKED, detect_existing_masks=False, **kwargs):
        self.tracks = tracks
        self.device = device
        self.L = model_cfg.max_len
        self.detect_existing_masks = detect_existing_masks
        
        # Extract generator from kwargs if provided (same as dataloader_old.py)
        generator = kwargs.pop('generator', None)
        
        # Prepare DataLoader kwargs
        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': kwargs.get('pin_memory', True),
            'persistent_workers': kwargs.get('persistent_workers', num_workers > 0),
            'prefetch_factor': kwargs.get('prefetch_factor', 2 if num_workers > 0 else None)
        }
        if 'sampler' in kwargs:
            dataloader_kwargs['sampler'] = kwargs['sampler']
        elif 'shuffle' in kwargs:
            dataloader_kwargs['shuffle'] = kwargs['shuffle']
        
        # Stage 1: cached raw dataloader (uses raw_collate_fn internally)
        self.cached_loader = CachedDataLoader(
            dataset,
            cache_size=cache_size,
            **dataloader_kwargs
        )
        
        # Stage 2: tokenization pipeline
        self.pipeline = TokenizationPipeline(
            model_cfg, train_cfg, tracks,
            autoencoder=autoencoder,
            device=device,
            min_unmasked=min_unmasked,
            generator=generator
        )
        
        self.metadata_queue = Queue(maxsize=100)
        self.running = True
        # Feeder thread is created per epoch in __iter__
        self.feeder_thread = None
    
    def _detect_existing_masks_from_raw_batch(self, raw_batch):
        """Detect pre-existing masks in raw data based on mask notation."""
        batch_size = len(raw_batch)
        masks = {}
        
        # Initialize all masks to False
        for track in self.tracks:
            if self.tracks[track] and track not in ['struct', 'orthologous_groups', 'semantic_description']:
                masks[track] = torch.zeros(batch_size, self.L, device=torch.device('cpu')).bool()
        
        # Detect masks for each sample in batch
        for i, protein in enumerate(raw_batch):
            # Sequence: detect '*' 
            if 'seq' in masks and hasattr(protein, 'seq'):
                for j, aa in enumerate(protein.seq):
                    if j < self.L and aa == '*':
                        masks['seq'][i, j] = True
            
            # Coordinates: align coordinate masks with sequence masks
            # For pre-masked proteins, coordinate masking should follow sequence masking
            if 'coords' in masks and hasattr(protein, 'coords') and hasattr(protein, 'seq'):
                # Check each position's coordinates
                for j in range(len(protein.seq)):
                    if j < self.L and j < protein.coords.shape[0]:
                        # Only mask coordinates where sequence is also masked (has '*')
                        if protein.seq[j] == '*':
                            masks['coords'][i, j] = True
            
            # PLDDT: detect -1
            if 'plddt' in masks and hasattr(protein, 'plddt'):
                for j, val in enumerate(protein.plddt):
                    if j < self.L and val == -1:
                        masks['plddt'][i, j] = True
            
            # SASA: detect -1
            if 'sasa' in masks and hasattr(protein, 'sasa'):
                for j, val in enumerate(protein.sasa):
                    if j < self.L and val == -1:
                        masks['sasa'][i, j] = True
            
            # SS8: detect '*'
            if 'ss8' in masks and hasattr(protein, 'ss8'):
                for j, val in enumerate(protein.ss8):
                    if j < self.L and val == '*':
                        masks['ss8'][i, j] = True
                        
            # Domains: could have '*' but it's multi-dimensional, skip for now
        
        # Structure masks follow coordinate masks
        if self.tracks.get('struct', False) and 'coords' in masks:
            masks['struct'] = masks['coords'].clone()
            
        return masks
    
    def _feeder_loop(self):
        # Assign an epoch_id and target at the start
        epoch_id = id(self) ^ int(torch.empty((), dtype=torch.int64).random_(2**31).item())
        self.pipeline.begin_submission(epoch_id=epoch_id)
        
        for raw_batch in self.cached_loader:
            if not self.running:
                break
            if raw_batch is None:
                continue
            
            # Check if we should detect existing masks
            if self.detect_existing_masks:
                # Detect masks from raw data
                masks = self._detect_existing_masks_from_raw_batch(raw_batch)
                metadata = None
            else:
                # Delegate mask generation to subclass
                masks, metadata = self.sample_masks(self.tracks, len(raw_batch))
                if self.tracks['struct']:
                    masks['struct'] = masks['coords']
                    
            self.metadata_queue.put(metadata)
            self.pipeline.submit_batch(raw_batch, masks)
        
        # Signal no more submissions for this epoch
        self.pipeline.end_submission()
    
    def __iter__(self):
        # Start epoch in pipeline (resets queues, toggles active)
        self.pipeline.start_epoch()
        
        # Start a fresh feeder thread for this epoch
        self.feeder_thread = Thread(target=self._feeder_loop, daemon=True)
        self.feeder_thread.start()
        
        while True:
            batch = self.pipeline.get_tokenized_batch(timeout=2.0)
            if batch is None:
                # End when feeder finished and everything submitted was emitted
                if (self.feeder_thread and not self.feeder_thread.is_alive() and 
                    self.pipeline.emitted >= self.pipeline.submitted_final):
                    break
                continue
            try:
                metadata = self.metadata_queue.get(timeout=0.1)
            except Empty:
                metadata = None
            batch.metadata = metadata if metadata else {}
            

            
            yield batch
        
        # End epoch in pipeline and ensure feeder dies cleanly
        self.pipeline.end_epoch()
        if self.feeder_thread and self.feeder_thread.is_alive():
            self.feeder_thread.join(timeout=2.0)
    
    def __len__(self):
        return len(self.cached_loader)
    
    @property
    def sampler(self):
        """Expose the sampler from the underlying DataLoader for distributed training."""
        return self.cached_loader.base_loader.sampler
    
    def shutdown(self):
        self.running = False
        self.cached_loader.shutdown()
        self.pipeline.shutdown()
        if self.feeder_thread.is_alive():
            self.feeder_thread.join(timeout=2.0)


class SimpleDataLoader(DecoupledDataLoader):
    """Decoupled dataloader with simple masking strategy."""
    
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, autoencoder=None, min_unmasked=_DEFAULT_MIN_UNMASKED, **kwargs):
        super().__init__(dataset, model_cfg, train_cfg, tracks, device,
                         autoencoder=autoencoder, min_unmasked=min_unmasked, **kwargs)
        
        assert isinstance(train_cfg.mask_config, SimpleMaskConfig)
        # Use mask_prob_seq as default for sequence-like tracks, mask_prob_struct for structure-like tracks
        self.simple_mask_prob = {'seq': train_cfg.mask_config.mask_prob_seq, 'coords': train_cfg.mask_config.mask_prob_struct, 'ss8': train_cfg.mask_config.mask_prob_seq, 'sasa': train_cfg.mask_config.mask_prob_seq, 'plddt': train_cfg.mask_config.mask_prob_seq, 'domains': train_cfg.mask_config.mask_prob_seq}

    def sample_masks(self, tracks, batch_len):
        masks = {}
        for track in [t for t in tracks if (tracks[t] and t != 'struct' and t != 'orthologous_groups' and t != 'semantic_description')]:

            # Create masks with fixed probabilities
            mask = torch.rand(batch_len, self.L, device=torch.device('cpu')) < self.simple_mask_prob[track]
            masks[track] = mask.bool()

        return masks, None


class ComplexDataLoader(DecoupledDataLoader):
    """Decoupled dataloader with complex masking strategy."""
    
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, autoencoder=None, min_unmasked=_DEFAULT_MIN_UNMASKED, **kwargs):
        super().__init__(dataset, model_cfg, train_cfg, tracks, device,
                         autoencoder=autoencoder, min_unmasked=min_unmasked, **kwargs)
        
        assert isinstance(train_cfg.mask_config, ComplexMaskConfig)

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
            if track == 'seq': probs = _sample_betalinear30(batch_len, device=torch.device('cpu'))
            elif track == 'coords': probs = _sample_cosine(batch_len, device=torch.device('cpu'))
            else: probs = _sample_sqrt(batch_len, device=torch.device('cpu'))
            
            # Create masks with per-protein rates
            # Expand rates to [B, L] and sample IID Bernoulli
            probs_full = probs.unsqueeze(1).expand(batch_len, self.L)
            
            mask = torch.rand(batch_len, self.L, device=torch.device('cpu')) < probs_full
            masks[track] = mask.bool()

            metadata[track] = probs

        return masks, metadata


class NoMaskDataLoader(DecoupledDataLoader):
    """Decoupled dataloader with no masking (for evaluation/generation)."""
    
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, autoencoder=None, **kwargs):
        # NoMaskDataLoader doesn't use min_unmasked
        super().__init__(dataset, model_cfg, train_cfg, tracks, device,
                         autoencoder=autoencoder, **kwargs)
        
        assert isinstance(train_cfg.mask_config, NoMaskConfig)
    
    def sample_masks(self, tracks, batch_len):
        masks = {}
        for track in [t for t in tracks if (tracks[t] and t != 'struct' and t != 'orthologous_groups' and t != 'semantic_description')]:
            masks[track] = torch.zeros(batch_len, self.L, device=torch.device('cpu')).bool()

        return masks, None


class DiffusionDataLoader(DecoupledDataLoader):
    """DataLoader that applies discrete diffusion noise process during batch collation."""

    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, autoencoder=None, min_unmasked=_DEFAULT_MIN_UNMASKED, **kwargs):

        corruption_mode = CorruptionMode.MASK if train_cfg.mask_config.corruption_mode == "absorb" else CorruptionMode.UNIFORM

        super().__init__(dataset, model_cfg, train_cfg, tracks, device,
                         autoencoder=autoencoder, min_unmasked=min_unmasked, **kwargs)

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
        # Move to CPU for worker compatibility
        self.inst_noise_levels = self.inst_noise_levels.to(torch.device('cpu'))
        self.cumulative_noise_levels = self.cumulative_noise_levels.to(torch.device('cpu'))

    def sample_masks(self, tracks, batch_len):
        # Sample timestep indices uniformly from [0, T-1]
        timestep_indices = torch.randint(0, self.diffusion_cfg.num_timesteps, (batch_len,), device=torch.device('cpu'))
        # Get corresponding noise levels
        cumulative_noise_level = self.cumulative_noise_levels[timestep_indices].unsqueeze(1)  # [B, 1]
        inst_noise_levels = self.inst_noise_levels[timestep_indices].unsqueeze(1)

        metadata = {'timestep_indices': timestep_indices, 'cumulative_noise': cumulative_noise_level, 'inst_noise': inst_noise_levels}
        masks = {}
        for track in [t for t in tracks if (tracks[t] and t != 'struct' and t != 'orthologous_groups' and t != 'semantic_description')]:

            mask_prob = 1 - torch.exp(-cumulative_noise_level)
            mask_prob_expanded = mask_prob.expand(batch_len, self.L)
            desired_masks = torch.rand(batch_len, self.L, device=torch.device('cpu')) < mask_prob_expanded
            masks[track] = desired_masks.bool()

        return masks, metadata
   