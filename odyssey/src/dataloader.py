import torch
import numpy as np
import random
import os
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
                 min_unmasked=_DEFAULT_MIN_UNMASKED, use_fp16=False, **kwargs):
        self.tracks = tracks
        self.device = device
        self.L = model_cfg.max_len
        
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
        if 'sampler' in kwargs: dataloader_kwargs['sampler'] = kwargs['sampler']
        elif 'shuffle' in kwargs: dataloader_kwargs['shuffle'] = kwargs['shuffle']
        
        # Stage 1: cached raw dataloader (uses raw_collate_fn internally)
        self.cached_loader = CachedDataLoader(dataset, cache_size=cache_size, **dataloader_kwargs)
        
        # Stage 2: tokenization pipeline
        self.pipeline = TokenizationPipeline(model_cfg, train_cfg, tracks, autoencoder=autoencoder, device=device, 
                                             min_unmasked=min_unmasked, generator=generator, use_fp16=use_fp16)

        self.metadata_queue = Queue(maxsize=100)
        self.running = True
        self.feeder_thread = None # Feeder thread is created per epoch in __iter__
    
    def _print_cache_saturation(self, tag: str = ""):
        """Print current cache and queue saturation metrics."""
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Only print from rank 5 (or rank 0 if running with fewer ranks)
        if rank != 5 and not (rank == 0 and world_size <= 5): return
            
        try:
            # Raw cache stats
            raw_cache_obj = self.cached_loader.cache
            raw_stats = self.cached_loader.get_stats()['cache']
            raw_batches = raw_stats.get('current_batches', 0)
            raw_proteins = raw_stats.get('current_proteins', 0)
            raw_max_batches = getattr(raw_cache_obj, 'max_batches', None)
            raw_max_size = getattr(raw_cache_obj, 'max_size', None)
            
            # Pipeline queue sizes
            raw_q = self.pipeline.raw_queue.qsize()
            cpu_q = self.pipeline.cpu_queue.qsize()
            ready_q = self.pipeline.ready_queue.qsize()

            # Structure token cache stats (if exists)
            struct_size = struct_max = struct_hit_rate = None
            if getattr(self.pipeline, 'gpu_stage', None) and getattr(self.pipeline.gpu_stage, 'cache', None):
                s_stats = self.pipeline.gpu_stage.cache.get_stats()
                struct_size = s_stats.get('size', None)
                struct_max = s_stats.get('max_size', None)
                struct_hit_rate = s_stats.get('hit_rate', None)

            # Format and print
            parts = []
            parts.append(f"RawCache proteins={raw_proteins}{'/' + str(raw_max_size) if raw_max_size is not None else ''} batches={raw_batches}{'/' + str(raw_max_batches) if raw_max_batches is not None else ''}")
            parts.append(f"Queues raw={raw_q} cpu={cpu_q} ready={ready_q}")
            if struct_size is not None:
                hr = f"{struct_hit_rate*100:.1f}%" if isinstance(struct_hit_rate, (int, float)) else "n/a"
                parts.append(f"StructTokenCache size={struct_size}{'/' + str(struct_max) if struct_max is not None else ''} hit_rate={hr}")
                
        except Exception as e: pass
    
    def _feeder_loop(self):
        # Assign an epoch_id and target at the start
        epoch_id = id(self) ^ int(torch.empty((), dtype=torch.int64).random_(2**31).item())
        self.pipeline.begin_submission(epoch_id=epoch_id)
        
        for raw_batch in self.cached_loader:
            if not self.running: break
            if raw_batch is None: continue
            
            # Delegate mask generation to subclass
            masks, metadata = self.sample_masks(self.tracks, len(raw_batch))
            if self.tracks['struct']: masks['struct'] = masks['coords']
                    
            self.metadata_queue.put(metadata)
            self.pipeline.submit_batch(raw_batch, masks)
        
        # Signal no more submissions for this epoch
        self.pipeline.end_submission()
    
    def __iter__(self):
        # Start epoch in pipeline (resets queues, toggles active)
        self.pipeline.start_epoch()
        # Print initial cache saturation
        self._print_cache_saturation(tag="epoch_start")
        
        # Start a fresh feeder thread for this epoch
        self.feeder_thread = Thread(target=self._feeder_loop, daemon=True)
        self.feeder_thread.start()
        
        step_count = 0
        max_wait_iterations = 10  # Maximum iterations to wait after feeder completes
        wait_iterations = 0
        
        try:
            while True:
                batch = self.pipeline.get_tokenized_batch(timeout=2.0)
                if batch is None:
                    # Check if feeder thread has finished
                    if self.feeder_thread and not self.feeder_thread.is_alive():
                        # Feeder is done, check if we've emitted everything
                        if self.pipeline.emitted >= self.pipeline.submitted_final: break # All batches have been emitted, we're done
                        else:
                            # Still waiting for GPU processing to finish
                            wait_iterations += 1
                            if wait_iterations >= max_wait_iterations:
                                print(f"[WARNING] Epoch ending: submitted={self.pipeline.submitted_final}, emitted={self.pipeline.emitted}, forcing exit")
                                break
                    continue
                
                # Reset wait counter since we got a batch
                wait_iterations = 0
                
                try: metadata = self.metadata_queue.get(timeout=0.1)
                except Empty: metadata = None
                batch.metadata = metadata if metadata else {}
                
                step_count += 1
                if step_count % 50 == 0:
                    self._print_cache_saturation(tag=f"step_{step_count}")
                    self._print_data_flow_stats()
                
                yield batch

        finally:
            # End epoch in pipeline and ensure feeder dies cleanly
            self._print_cache_saturation(tag="epoch_end")
            self._print_data_flow_stats()  # Final stats at epoch end
            try: self.pipeline.end_epoch()
            except Exception: pass
            if self.feeder_thread and self.feeder_thread.is_alive():
                try: self.feeder_thread.join(timeout=2.0)
                except Exception: pass
    
    def __len__(self):
        return len(self.cached_loader)
    
    @property
    def sampler(self):
        """Expose the sampler from the underlying DataLoader for distributed training."""
        return self.cached_loader.base_loader.sampler
    
    def _print_data_flow_stats(self):
        """Print data flow statistics through the pipeline."""
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Only print from rank 5 (or rank 0 if running with fewer ranks)
        if rank != 5 and not (rank == 0 and world_size <= 5): return
    
    def shutdown(self):
        self.running = False
        self.cached_loader.shutdown()
        self.pipeline.shutdown()
        if self.feeder_thread and self.feeder_thread.is_alive():
            self.feeder_thread.join(timeout=2.0)


class ContentBasedDataLoader(DecoupledDataLoader):
    """
    DataLoader that creates masks based on the actual content of the data.
    
    Masking rules:
    - seq: mask positions where sequence contains "*"  
    - coords: mask positions where coordinates are [-1, -1, -1]
    - plddt: mask positions where plddt is -1
    - sasa: mask positions where sasa is -1
    - struct: copies coords mask (handled automatically in base class)
    """
    def _feeder_loop(self):
        """Override to pass raw_batch to sample_masks for content-based masking."""
        epoch_id = id(self) ^ int(torch.empty((), dtype=torch.int64).random_(2**31).item())
        self.pipeline.begin_submission(epoch_id=epoch_id)
        
        for raw_batch in self.cached_loader:
            if not self.running: break
            if raw_batch is None: continue
            
            masks, metadata = self.sample_masks(self.tracks, len(raw_batch), raw_batch)
            if self.tracks['struct']: masks['struct'] = masks['coords']
                    
            self.metadata_queue.put(metadata)
            self.pipeline.submit_batch(raw_batch, masks)
        
        self.pipeline.end_submission()

    def sample_masks(self, tracks, batch_len, raw_batch=None):
        """Generate content-based masks: seq='*', coords=[-1,-1,-1], plddt/sasa=-1"""
        if raw_batch is None:
            return {track: torch.zeros(batch_len, self.L, device=torch.device('cpu')).bool() 
                    for track in tracks if tracks[track] and track not in ['struct', 'orthologous_groups', 'semantic_description']}, None
        
        masks = {}
        for track in [t for t in tracks if tracks[t] and t not in ['struct', 'orthologous_groups', 'semantic_description']]:
            mask = torch.zeros(batch_len, self.L, device=torch.device('cpu')).bool()
            
            if track == 'seq':
                for b, seq in enumerate(raw_batch.sequences):
                    # Account for BOS token - shift mask positions by 1
                    seq_mask = torch.tensor([c == '*' for c in seq[:self.L-2]])  # -2 for BOS/EOS
                    if len(seq_mask) > 0:
                        mask[b, 1:1+len(seq_mask)] = seq_mask  # Start at position 1 (after BOS)
            
            elif track == 'coords':
                # Check coordinates directly - mask positions where coords are [-1,-1,-1]
                for b, coords in enumerate(raw_batch.coords):
                    for pos in range(min(coords.shape[0], self.L-2)):  # -2 for BOS/EOS
                        # Check if ALL atoms in this position are [-1,-1,-1]
                        if all(torch.allclose(coords[pos, atom], torch.tensor([-1.0, -1.0, -1.0])) for atom in range(coords.shape[1])):
                            mask[b, pos+1] = True  # +1 to account for BOS token
            
            elif track in ['plddt', 'sasa']:
                data_list = raw_batch.plddt if track == 'plddt' else raw_batch.sasa
                for b, vals in enumerate(data_list):
                    # Account for BOS token - shift mask positions by 1
                    val_mask = torch.tensor([v == -1 for v in vals[:self.L-2]])  # -2 for BOS/EOS
                    if len(val_mask) > 0:
                        mask[b, 1:1+len(val_mask)] = val_mask  # Start at position 1 (after BOS)
            
            masks[track] = mask
        
        return masks, None


class SimpleDataLoader(DecoupledDataLoader):
    """Decoupled dataloader with simple masking strategy."""
    
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, autoencoder=None, min_unmasked=_DEFAULT_MIN_UNMASKED, use_fp16=False, **kwargs):
        super().__init__(dataset, model_cfg, train_cfg, tracks, device,
                         autoencoder=autoencoder, min_unmasked=min_unmasked, use_fp16=use_fp16, **kwargs)

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
    
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, autoencoder=None, min_unmasked=_DEFAULT_MIN_UNMASKED, use_fp16=False, **kwargs):
        super().__init__(dataset, model_cfg, train_cfg, tracks, device,
                         autoencoder=autoencoder, min_unmasked=min_unmasked, use_fp16=use_fp16, **kwargs)
        
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

    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, autoencoder=None, min_unmasked=_DEFAULT_MIN_UNMASKED, use_fp16=False, **kwargs):

        corruption_mode = CorruptionMode.MASK if train_cfg.mask_config.corruption_mode == "absorb" else CorruptionMode.UNIFORM

        super().__init__(dataset, model_cfg, train_cfg, tracks, device,
                         autoencoder=autoencoder, min_unmasked=min_unmasked, use_fp16=use_fp16, **kwargs)

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
   