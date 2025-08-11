"""
Data pipeline components for decoupled fetch and tokenization.

This module provides a clean separation between data loading and tokenization,
enabling multi-worker data loading even when using GPU-based autoencoder tokenization.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from queue import Queue, Empty
from threading import Thread, Lock
import time
import hashlib

# Import tokenizers
from odyssey.src.tokenizer import (
    SequenceTokenizer, StructureTokenizer, CoordinatesTokenizer,
    SS8Tokenizer, SASATokenizer, PLDDTTokenizer,
    OrthologousGroupsTokenizer, SemanticDescriptionTokenizer,
    DomainsTokenizer, CorruptionMode
)

@dataclass
class RawBatch:
    """
    Container for raw, untokenized batch data directly from the dataset.
    This is what DataLoader workers produce - no tokenization yet.
    """
    sequences: List  # Raw sequence strings
    coords: List[torch.Tensor]  # Raw coordinate tensors
    ss8: List  # Secondary structure strings
    sasa: List  # Solvent accessible surface area
    orthologous_groups: List  # Orthologous group strings
    semantic_description: List  # Semantic descriptions
    domains: List  # Domain annotations
    plddt: List  # pLDDT scores
    lengths: List[int]  # Original lengths
    
    @property
    def batch_size(self) -> int: return len(self.sequences)
    
    def __len__(self) -> int: return self.batch_size
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get a single item from the batch."""
        return (self.sequences[idx], self.coords[idx], self.ss8[idx], self.sasa[idx], self.orthologous_groups[idx], 
                self.semantic_description[idx], self.domains[idx], self.plddt[idx], self.lengths[idx])

class RawDataCache:
    """
    Bounded cache for prefetched raw data batches.
    This cache sits between DataLoader workers and the tokenization pipeline,
    allowing workers to run ahead of GPU processing.
    """
    
    def __init__(self, max_size: int = 1000, max_batches: int = 50):
        """
        Args:
            max_size: Maximum number of individual proteins to cache
            max_batches: Maximum number of batches to cache
        """
        self.max_size = max_size
        self.max_batches = max_batches
        self.batch_queue = deque(maxlen=max_batches)
        self.protein_count = 0
        self.lock = Lock()
        
        # Statistics
        self.stats = {'batches_added': 0, 'batches_retrieved': 0, 'proteins_cached': 0, 'cache_full_events': 0, 'total_wait_time': 0.0}
    
    def can_add_batch(self, batch: RawBatch) -> bool:
        """Check if there's room for another batch."""
        with self.lock:
            if len(self.batch_queue) >= self.max_batches: return False
            if self.protein_count + len(batch) > self.max_size: return False
            return True
    
    def add_batch(self, batch: RawBatch, timeout: float = 1.0) -> bool:
        """
        Add a batch to the cache.
        Returns True if added successfully, False if cache is full.
        """
        start_time = time.time()
        
        # Wait for space if needed
        while not self.can_add_batch(batch):
            time.sleep(0.01)
            if time.time() - start_time > timeout:
                with self.lock: self.stats['cache_full_events'] += 1
                return False
        
        with self.lock:
            self.batch_queue.append(batch)
            self.protein_count += len(batch)
            self.stats['batches_added'] += 1
            self.stats['proteins_cached'] = self.protein_count
            
        return True
    
    def get_batch(self, timeout: float = 5.0) -> Optional[RawBatch]:
        """
        Retrieve a batch from the cache.
        Returns RawBatch if available, None if timeout.
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                if self.batch_queue:
                    batch = self.batch_queue.popleft()
                    self.protein_count -= len(batch)
                    self.stats['batches_retrieved'] += 1
                    self.stats['total_wait_time'] += time.time() - start_time
                    return batch
            
            if time.time() - start_time > timeout: return None
            time.sleep(0.01)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats['current_batches'] = len(self.batch_queue)
            stats['current_proteins'] = self.protein_count
            stats['cache_usage'] = self.protein_count / self.max_size if self.max_size > 0 else 0
            
            if stats['batches_retrieved'] > 0: stats['avg_wait_time'] = stats['total_wait_time'] / stats['batches_retrieved']
            else: stats['avg_wait_time'] = 0.0
                
        return stats
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.batch_queue.clear()
            self.protein_count = 0

class StructureTokenCache:
    """
    Cache for structure tokens to avoid redundant autoencoder inference.
    Uses a simple LRU cache with a hash of coordinates as the key.
    """
    def __init__(self, max_size: int = 500):
        """
        Args:
            max_size: Maximum number of structure tokens to cache
        """
        self.max_size = max_size
        self.cache = {}  # hash -> tokens
        self.access_order = deque(maxlen=max_size)
        self.lock = Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def _compute_hash(self, coords) -> str:
        """Compute a hash for the coordinates tensor."""
        # Use first 10 residues for hash (usually stable)
        key_coords = coords[:min(10, coords.shape[0])].cpu()
        coords_bytes = key_coords.numpy().tobytes()
        return hashlib.md5(coords_bytes).hexdigest()
    
    def get(self, coords) -> Optional[torch.Tensor]:
        """
        Retrieve cached tokens for given coordinates.
        Returns cached tokens if found, None otherwise.
        """
        key = self._compute_hash(coords)
        
        with self.lock:
            if key in self.cache:
                self.hits += 1
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key].clone()
            else:
                self.misses += 1
                return None
    
    def put(self, coords, tokens):
        """Store tokens in cache."""
        key = self._compute_hash(coords)
        
        with self.lock:
            # Check if we need to evict
            if key not in self.cache and len(self.cache) >= self.max_size:
                # Evict least recently used
                if self.access_order:
                    lru_key = self.access_order[0]
                    del self.cache[lru_key]
            
            # Add to cache
            self.cache[key] = tokens.clone()
            if key in self.access_order: self.access_order.remove(key)
            self.access_order.append(key)
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        if total == 0: return 0.0
        return self.hits / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {'size': len(self.cache), 'max_size': self.max_size, 'hits': self.hits, 'misses': self.misses,
                    'hit_rate': self.get_hit_rate(), 'usage': len(self.cache) / self.max_size if self.max_size > 0 else 0}
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0

def raw_collate_fn(batch_list: List[Tuple]) -> RawBatch:
    """
    Collate function that creates RawBatch without any tokenization.
    This replaces the tokenization-heavy _mask_collate function for the
    first stage of our pipeline.
    
    Args:
        batch_list: List of tuples from Dataset.__getitem__
        
    Returns:
        RawBatch containing unprocessed data
    """
    # Filter out None items
    batch_list = [item for item in batch_list if item is not None]
    
    if not batch_list: return None
    
    # Unpack all items based on Dataset.__getitem__ return format:
    # (seq, coords, ss8, sasa, orthologous_groups, semantic_description, domains, plddt, length)
    sequences = []; coords = []; ss8 = []; sasa = []; orthologous_groups = []; semantic_description = []; domains = []; plddt = []; lengths = []
    
    for item in batch_list:
        seq, coord, ss8_item, sasa_item, og, sd, dom, plddt_item, length = item
        sequences.append(seq)
        coords.append(coord)
        ss8.append(ss8_item)
        sasa.append(sasa_item)
        orthologous_groups.append(og)
        semantic_description.append(sd)
        domains.append(dom)
        plddt.append(plddt_item)
        lengths.append(length.item() if torch.is_tensor(length) else length)
    
    return RawBatch(sequences=sequences, coords=coords, ss8=ss8, sasa=sasa, orthologous_groups=orthologous_groups,
                    semantic_description=semantic_description, domains=domains, plddt=plddt, lengths=lengths)

class CachedDataLoader:
    """
    DataLoader wrapper that prefetches and caches raw batches.
    This allows workers to run ahead of the main training loop without
    being blocked by GPU tokenization.
    """
    def __init__(self, dataset, batch_size: int = 32, num_workers: int = 4, cache_size: int = 1000, max_batches: int = 50, **kwargs):
        """
        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            num_workers: Number of worker processes
            cache_size: Maximum proteins to cache
            max_batches: Maximum batches to cache
            **kwargs: Additional DataLoader arguments
        """
        self.cache = RawDataCache(max_size=cache_size, max_batches=max_batches)
        
        # Create base DataLoader with raw collate function
        kwargs['collate_fn'] = raw_collate_fn
        kwargs['num_workers'] = num_workers
        kwargs['batch_size'] = batch_size
        kwargs['pin_memory'] = kwargs.get('pin_memory', True)
        kwargs['persistent_workers'] = kwargs.get('persistent_workers', True) if num_workers > 0 else False
        
        self.base_loader = DataLoader(dataset, **kwargs)
        
        # Start prefetch thread
        self.prefetch_thread = Thread(target=self._prefetch_loop, daemon=True)
        self.running = True
        self.prefetch_thread.start()
    
    def _prefetch_loop(self):
        """Background thread that continuously prefetches batches, tolerant to DataLoader shutdown."""
        try:
            for batch in self.base_loader:
                if not self.running: break
                if batch is None: continue
                # Try to add to cache, wait if full
                while self.running and not self.cache.add_batch(batch, timeout=0.1): time.sleep(0.01)
        except AssertionError: return # PyTorch DataLoader is shutting down; exit quietly
        except AttributeError as e:
            # This happens when _iterator is deleted while we're using it - expected during shutdown
            if '_iterator' in str(e): return
            print(f"Prefetch thread attribute error: {e}")
        except RuntimeError as e:
            # Some shutdown paths raise RuntimeError; treat as graceful exit
            if 'shutdown' in str(e).lower(): return
            print(f"Prefetch thread runtime error: {e}")
        except Exception as e: print(f"Prefetch thread error: {e}")
    
    def __iter__(self):
        """Iterate over cached batches. Restart prefetch/cache each epoch."""
        # Restart cache and prefetch thread for a new epoch
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.running = False
            self.prefetch_thread.join(timeout=5.0)  # Increase timeout
            
            # Force kill if still alive
            if self.prefetch_thread.is_alive():
                print(f"Warning: Prefetch thread did not terminate cleanly")
        
        
        # Recreate cache and prefetch thread
        self.cache = RawDataCache(max_size=self.cache.max_size, max_batches=self.cache.max_batches)
        self.running = True
        self.prefetch_thread = Thread(target=self._prefetch_loop, daemon=True)
        self.prefetch_thread.start()
        
        while True:
            batch = self.cache.get_batch(timeout=2.0)
            if batch is None:
                # Check if prefetch thread is still running
                if not self.prefetch_thread.is_alive(): break
                continue
            yield batch
    
    def __len__(self):
        """Return length of base loader."""
        return len(self.base_loader)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {'cache': self.cache.get_stats(), 'prefetch_running': self.prefetch_thread.is_alive()}
    
    def shutdown(self):
        """Shutdown the loader."""
        self.running = False
        if self.prefetch_thread.is_alive(): self.prefetch_thread.join(timeout=5.0)
        self.cache.clear()

# --------------------------------------------------------------------------- #
#  Tokenization Pipeline Components                                           #
# --------------------------------------------------------------------------- #
class PartiallyTokenizedBatch:
    """
    Intermediate batch with CPU tokenization complete but structure tokens pending.
    
    This is the output of CPU tokenization and input to GPU tokenization.
    """
    def __init__(self):
        # Tokenized data (everything except structure)
        self.masked_data = {}
        self.unmasked_data = {}
        self.beospank = {}
        self.masks = {}
        
        # Metadata
        self.metadata = {}
        self.tracks = {}
        self.device = torch.device('cpu')
        self.B = 0  # Batch size
        self.L = 0  # Sequence length
            
    def to(self, device):
        """Move all batch data to the specified device - matching MaskedBatch.to()"""
        # Update device attribute
        self.device = device
        
        # Helper function to safely move tensors
        def safe_to_device(tensor_dict, key):
            if key in tensor_dict and tensor_dict[key] is not None:
                tensor_dict[key] = tensor_dict[key].to(device)
        
        # Move all data dictionaries to device
        for key in self.masked_data.keys(): safe_to_device(self.masked_data, key)
        for key in self.unmasked_data.keys(): safe_to_device(self.unmasked_data, key)
        for key in self.beospank.keys(): safe_to_device(self.beospank, key)
        for key in self.masks.keys(): safe_to_device(self.masks, key)
        
        # Move metadata tensors to device (for diffusion loader compatibility)
        if hasattr(self, 'metadata') and self.metadata:
            for key, value in self.metadata.items():
                if isinstance(value, torch.Tensor): self.metadata[key] = value.to(device)
                    
        return self


class CPUTokenizationStage:
    """
    Handles all CPU-based tokenization operations.
    This can run safely in DataLoader workers.
    """
    def __init__(self, model_cfg, train_cfg, tracks, device='cpu', min_unmasked=None, generator=None):        
        self.device = torch.device(device)
        self.tracks = tracks
        self.L = model_cfg.max_len
        self.K = model_cfg.max_domains_per_residue
        self.G = model_cfg.max_len_orthologous_groups
        self.H = model_cfg.max_len_semantic_description
        
        # min_unmasked is guaranteed to be provided (always has 'seq' and 'coords' keys)
        self.min_unmasked = min_unmasked
        
        # Determine corruption mode
        if hasattr(train_cfg.mask_config, 'corruption_mode'):
            if train_cfg.mask_config.corruption_mode == "absorb": corruption_mode = CorruptionMode.MASK
            else: corruption_mode = CorruptionMode.UNIFORM
        else: corruption_mode = CorruptionMode.MASK
        
        # Initialize CPU tokenizers with proper min_unmasked and generator (same as dataloader_old.py)
        self.sequence_tokenizer = SequenceTokenizer(self.L, min_unmasked=self.min_unmasked['seq'], generator=generator, corruption_mode=corruption_mode)
        self.coordinates_tokenizer = CoordinatesTokenizer(self.L, min_unmasked=self.min_unmasked['coords'], generator=generator)
        self.ss8_tokenizer = SS8Tokenizer(self.L, generator=generator, corruption_mode=corruption_mode)
        self.sasa_tokenizer = SASATokenizer(self.L, generator=generator, corruption_mode=corruption_mode)
        self.plddt_tokenizer = PLDDTTokenizer(self.L, generator=generator, corruption_mode=corruption_mode)
        self.orthologous_groups_tokenizer = OrthologousGroupsTokenizer(self.G)
        self.semantic_description_tokenizer = SemanticDescriptionTokenizer(self.H)
        self.domains_tokenizer = DomainsTokenizer(self.L, self.K, generator=generator, corruption_mode=corruption_mode)
    
    def tokenize_batch_cpu(self, raw_batch: RawBatch, masks: Dict) -> PartiallyTokenizedBatch:
        """
        Perform all CPU tokenization on a raw batch.
        
        Args:
            raw_batch: Raw data from dataset
            masks: Pre-generated masks for each track
            
        Returns:
            PartiallyTokenizedBatch ready for GPU tokenization
        """
        result = PartiallyTokenizedBatch()
        result.tracks = self.tracks
        result.masks = masks
        result.device = self.device
        result.B = len(raw_batch)
        
        # Tokenize sequences
        if self.tracks['seq']:
            seq_data = []
            for idx, seq in enumerate(raw_batch.sequences):
                seq_data.append(self.sequence_tokenizer.tokenize(seq, masks['seq'][idx]))
            
            unmasked, masked, beospank, seq_masks = zip(*seq_data)
            result.unmasked_data['seq'] = torch.stack(unmasked, dim=0)
            result.masked_data['seq'] = torch.stack(masked, dim=0)
            result.beospank['seq'] = torch.stack(beospank, dim=0).bool()
            result.masks['seq'] = torch.stack(seq_masks, dim=0).bool()
        
        # Tokenize coordinates
        if self.tracks['coords']:
            coords_data = []
            for idx, coords in enumerate(raw_batch.coords): coords_data.append(self.coordinates_tokenizer.tokenize(coords, masks['coords'][idx]))
            padded_coords, masked_coords, coords_beospank, coords_masks = zip(*coords_data)
            result.unmasked_data['coords'] = torch.stack(padded_coords, dim=0)
            result.masked_data['coords'] = torch.stack(masked_coords, dim=0)
            result.beospank['coords'] = torch.stack(coords_beospank, dim=0).bool()
            result.masks['coords'] = torch.stack(coords_masks, dim=0).bool()
                    
        # Tokenize SS8
        if self.tracks['ss8']:
            ss8_data = []
            for idx, ss8 in enumerate(raw_batch.ss8): ss8_data.append(self.ss8_tokenizer.tokenize(ss8, masks['ss8'][idx]))
            unmasked, masked, beospank, ss8_masks = zip(*ss8_data)
            result.unmasked_data['ss8'] = torch.stack(unmasked, dim=0)
            result.masked_data['ss8'] = torch.stack(masked, dim=0)
            result.beospank['ss8'] = torch.stack(beospank, dim=0).bool()
            result.masks['ss8'] = torch.stack(ss8_masks, dim=0).bool()
        
        # Tokenize SASA
        if self.tracks['sasa']:
            sasa_data = []
            for idx, sasa in enumerate(raw_batch.sasa): sasa_data.append(self.sasa_tokenizer.tokenize(sasa, masks['sasa'][idx]))
            unmasked, masked, beospank, sasa_masks = zip(*sasa_data)
            result.unmasked_data['sasa'] = torch.stack(unmasked, dim=0)
            result.masked_data['sasa'] = torch.stack(masked, dim=0)
            result.beospank['sasa'] = torch.stack(beospank, dim=0).bool()
            result.masks['sasa'] = torch.stack(sasa_masks, dim=0).bool()
        
        # Tokenize pLDDT
        if self.tracks['plddt']:
            plddt_data = []
            for idx, plddt in enumerate(raw_batch.plddt): plddt_data.append(self.plddt_tokenizer.tokenize(plddt, masks['plddt'][idx]))
            unmasked, masked, beospank, plddt_masks = zip(*plddt_data)
            result.unmasked_data['plddt'] = torch.stack(unmasked, dim=0)
            result.masked_data['plddt'] = torch.stack(masked, dim=0)
            result.beospank['plddt'] = torch.stack(beospank, dim=0).bool()
            result.masks['plddt'] = torch.stack(plddt_masks, dim=0).bool()
        
        # Tokenize orthologous groups
        if self.tracks['orthologous_groups']:
            og_data = []
            for og in raw_batch.orthologous_groups: og_data.append(self.orthologous_groups_tokenizer.tokenize(og))
            unmasked, beospank = zip(*og_data)
            result.unmasked_data['orthologous_groups'] = torch.stack(unmasked, dim=0)
            result.beospank['orthologous_groups'] = torch.stack(beospank, dim=0).bool()
        
        # Tokenize semantic descriptions
        if self.tracks['semantic_description']:
            sd_data = []
            for sd in raw_batch.semantic_description: sd_data.append(self.semantic_description_tokenizer.tokenize(sd))
            unmasked, beospank = zip(*sd_data)
            result.unmasked_data['semantic_description'] = torch.stack(unmasked, dim=0)
            result.beospank['semantic_description'] = torch.stack(beospank, dim=0).bool()
        
        # Tokenize domains
        if self.tracks['domains']:
            domains_data = []
            for idx, domains in enumerate(raw_batch.domains): domains_data.append(self.domains_tokenizer.tokenize(domains, masks['domains'][idx]))
            unmasked, masked, beospank, domain_masks = zip(*domains_data)
            result.unmasked_data['domains'] = torch.stack(unmasked, dim=0)
            result.masked_data['domains'] = torch.stack(masked, dim=0)
            result.beospank['domains'] = torch.stack(beospank, dim=0).bool()
            result.masks['domains'] = torch.stack(domain_masks, dim=0).bool()
        
        return result

class GPUStructureTokenizationStage:
    """
    Handles GPU-based structure tokenization via autoencoder.
    This must run in the main process on GPU.
    """
    def __init__(self, model_cfg, train_cfg, autoencoder, device='cuda', min_unmasked=None, generator=None, use_fp16=False):
        self.device = torch.device(device)
        self.cache = StructureTokenCache(max_size=500)
        self.L = model_cfg.max_len
        
        # Check if autoencoder is already in FP16 or convert if requested
        if autoencoder is not None:
            # Check current dtype of autoencoder
            first_param = next(autoencoder.parameters())
            is_already_fp16 = (first_param.dtype == torch.float16)
            
            if is_already_fp16: self.use_fp16 = True
            elif use_fp16 and self.device.type == 'cuda':
                # Convert model to FP16 for faster inference
                autoencoder = autoencoder.half()
                self.use_fp16 = True
            else: self.use_fp16 = False
            
            self.autoencoder = autoencoder
            self.min_unmasked = min_unmasked
            
            # Determine corruption mode (same logic as CPU stage)
            if hasattr(train_cfg.mask_config, 'corruption_mode'):
                if train_cfg.mask_config.corruption_mode == "absorb": corruption_mode = CorruptionMode.MASK
                else: corruption_mode = CorruptionMode.UNIFORM
            else: corruption_mode = CorruptionMode.MASK
            
            # Initialize structure tokenizer with proper configuration and generator (same as dataloader_old.py)
            self.structure_tokenizer = StructureTokenizer(self.L, self.autoencoder, min_unmasked=self.min_unmasked['coords'], generator=generator, corruption_mode=corruption_mode)
    
    def tokenize_batch_gpu(self, partial_batch: PartiallyTokenizedBatch) -> None:
        """
        Complete structure tokenization for a partially tokenized batch.
        Uses pre-computed coordinate tokenization results for efficiency.
        Now uses the batch method from StructureTokenizer.
        
        Args:
            partial_batch: Batch with CPU tokenization complete (including coords)
        """
        # Check if structure tokenization is needed
        if self.autoencoder is None: raise ValueError("Autoencoder required for structure tokenization")
        
        # Prepare precomputed coords for batch processing
        precomputed_coords_list = []
        
        for b in range(partial_batch.B):
            # Get pre-computed coordinate data from CPU stage and move to GPU
            padded_coords = partial_batch.unmasked_data['coords'][b].to(self.device)
            masked_coords = partial_batch.masked_data['coords'][b].to(self.device)
            coords_beospank = partial_batch.beospank['coords'][b].to(self.device)
            coords_masks = partial_batch.masks['coords'][b].to(self.device)
            
            # Convert coords to FP16 if autoencoder is in FP16
            if self.use_fp16:
                padded_coords = padded_coords.half()
                masked_coords = masked_coords.half()
            
            # Add to list as tuple (matching tokenize() input format)
            precomputed_coords_list.append((padded_coords, masked_coords, coords_beospank, coords_masks))
        
        # Use the batch tokenization method from StructureTokenizer
        self.structure_tokenizer._structure_cache = self.cache
        batch_results = self.structure_tokenizer.tokenize_batch(precomputed_coords_list)
        
        # Extract structure tokens from results and move to CPU
        struct_data = []
        for result in batch_results:
            # result format: (padded_coords, masked_coords, coords_beospank, coords_masks,
            #                 padded_struct_tokens, masked_struct_tokens, struct_beospank, struct_masks)
            struct_output = (result[4].cpu(), result[5].cpu(), result[6].cpu(), result[7].cpu())
            struct_data.append(struct_output)
        
        # Store structure results
        unmasked, masked, beospank, struct_masks = zip(*struct_data)
        partial_batch.unmasked_data['struct'] = torch.stack(unmasked, dim=0)
        partial_batch.masked_data['struct'] = torch.stack(masked, dim=0)
        partial_batch.beospank['struct'] = torch.stack(beospank, dim=0).bool()
        partial_batch.masks['struct'] = torch.stack(struct_masks, dim=0).bool()

class TokenizationPipeline:
    """
    Orchestrates the complete tokenization pipeline from raw data to final batches.
    This pipeline:
    1. Receives raw batches from CachedDataLoader
    2. Performs CPU tokenization in parallel
    3. Batches data for efficient GPU tokenization
    4. Produces final MaskedBatch objects ready for training
    """
    def __init__(self, model_cfg, train_cfg, tracks, autoencoder=None, 
                 device='cuda', cpu_buffer_size=200, gpu_buffer_size=100, min_unmasked=None, generator=None, use_fp16=False):
        """
        Args:
            model_cfg: Model configuration
            train_cfg: Training configuration  
            tracks: Which data tracks to tokenize
            autoencoder: Autoencoder for structure tokenization (can be None)
            device: Device for final batch
            cpu_buffer_size: Max batches in CPU tokenization queue
            gpu_buffer_size: Max batches in GPU tokenization queue
            use_fp16: If True, use FP16 for autoencoder inference
        """
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.tracks = tracks
        self.device = torch.device(device)
        
        # Create pipeline stages with min_unmasked and generator
        self.cpu_stage = CPUTokenizationStage(model_cfg, train_cfg, tracks, device='cpu', min_unmasked=min_unmasked, generator=generator)
        
        if autoencoder is not None and tracks['struct']:
            self.gpu_stage = GPUStructureTokenizationStage(model_cfg, train_cfg, autoencoder, device=device, min_unmasked=min_unmasked, generator=generator, use_fp16=use_fp16)
        else: self.gpu_stage = None
        
        # Pipeline queues
        self.raw_queue = Queue(maxsize=cpu_buffer_size)
        self.cpu_queue = Queue(maxsize=cpu_buffer_size)
        self.ready_queue = Queue(maxsize=gpu_buffer_size)
        
        # Epoch control
        self.epoch_active = False
        self.submission_open = False
        self.epoch_id = 0
        self.epoch_target = 0
        self.submitted = 0
        self.emitted = 0
        
        # Tracking counters for data flow
        self.cpu_batches_received = 0
        self.cpu_batches_sent = 0
        self.gpu_batches_received = 0
        self.gpu_batches_processed = 0
        
        # Worker threads
        self.running = True
        self.cpu_thread = Thread(target=self._cpu_worker, daemon=True)
        self.gpu_thread = Thread(target=self._gpu_worker, daemon=True) if self.gpu_stage else None
        
        # Start workers
        self.cpu_thread.start()
        if self.gpu_thread: self.gpu_thread.start()
    
    def begin_submission(self, epoch_id: int):
        """Called by feeder at epoch start to set identifiers and reset counters."""
        self.epoch_id = epoch_id
        self.submitted = 0
        self.emitted = 0
        self.submitted_final = 0
        self.submission_open = True
    
    def end_submission(self):
        """Called by feeder when no more batches will be submitted this epoch."""
        self.submission_open = False
        self.submitted_final = self.submitted
    
    def start_epoch(self):
        """Mark start of an epoch and ensure queues are empty."""
        # If threads are dead, restart them
        if not self.cpu_thread.is_alive():
            self.running = True
            self.cpu_thread = Thread(target=self._cpu_worker, daemon=True)
            self.cpu_thread.start()
        
        if self.gpu_thread and not self.gpu_thread.is_alive():
            self.running = True
            self.gpu_thread = Thread(target=self._gpu_worker, daemon=True)
            self.gpu_thread.start()
        
        self._drain_all_queues()
        self.epoch_active = True
        # do not reset counters here; feeder sets them via begin_submission
    
    def end_epoch(self):
        """Mark end of an epoch and stop accepting/producing until next start."""
        # Wait until we have emitted everything that was submitted
        # Prevent late enqueues by marking inactive
        self.epoch_active = False
        self._drain_all_queues()
    
    def _drain(self, q):
        try: 
            while True: q.get_nowait()
        except Empty: return
    
    def _drain_all_queues(self):
        self._drain(self.raw_queue)
        self._drain(self.cpu_queue)
        self._drain(self.ready_queue)
    
    def _attach_epoch(self, obj):
        # Helper to tag tuples with epoch
        return (self.epoch_id, obj)
    
    def _cpu_worker(self):
        """Worker thread for CPU tokenization."""
        self.cpu_batches_received = 0
        self.cpu_batches_sent = 0
        
        while self.running:
            try:
                raw_batch, masks = self.raw_queue.get(timeout=1.0)
                if not (self.epoch_active and (self.submission_open or self.submitted > 0)): continue
                    
                self.cpu_batches_received += 1
                partial_batch = self.cpu_stage.tokenize_batch_cpu(raw_batch, masks)
                
                # Tag with epoch_id
                if self.gpu_stage:
                    tagged = (self.epoch_id, partial_batch)
                    self.cpu_queue.put(tagged)
                    self.cpu_batches_sent += 1
                else:
                    tagged = (self.epoch_id, partial_batch)
                    self.ready_queue.put(tagged)
                    self.emitted += 1
                        
            except Empty: continue
            except Exception as e:
                print(f"Error in CPU tokenization: {e}")
                import traceback; traceback.print_exc()
    
    def _gpu_worker(self):
        """Worker thread for GPU structure tokenization."""
        batch_accumulator = []
        self.gpu_batches_received = 0
        self.gpu_batches_processed = 0
        
        while self.running:
            try:
                item = self.cpu_queue.get(timeout=0.1)
                epoch_tag, partial_batch = item
                if epoch_tag != self.epoch_id or not self.epoch_active: continue
                    
                batch_accumulator.append(partial_batch)
                self.gpu_batches_received += 1
                
                if len(batch_accumulator) >= 4:
                    self._process_gpu_batch(batch_accumulator)
                    self.gpu_batches_processed += len(batch_accumulator)
                    batch_accumulator = []
                            
            except Empty:
                # Process any remaining batches when queue is empty
                # This is critical for end of epoch when we have < 4 remaining batches
                if batch_accumulator:
                    # Check if we should process: either still active OR no more submissions coming
                    if self.epoch_active or (not self.submission_open and self.submitted > self.emitted):
                        self._process_gpu_batch(batch_accumulator)
                        self.gpu_batches_processed += len(batch_accumulator)
                        batch_accumulator = []

            except Exception as e:
                print(f"Error in GPU tokenization: {e}")
                import traceback; traceback.print_exc()
    
    def _process_gpu_batch(self, batches):
        """Process multiple batches together for efficient GPU utilization."""
        if len(batches) == 1:
            # Single batch - process normally
            batch = batches[0]
            self.gpu_stage.tokenize_batch_gpu(batch)
            self.ready_queue.put((self.epoch_id, batch))
            self.emitted += 1
        else:
            # Multiple batches - can process together for better GPU utilization
            # Process each batch (they're already accumulated for efficiency)
            for batch in batches:
                self.gpu_stage.tokenize_batch_gpu(batch)
                self.ready_queue.put((self.epoch_id, batch))
                self.emitted += 1
    
    def submit_batch(self, raw_batch: RawBatch, masks: Dict):
        if self.epoch_active:
            self.raw_queue.put((raw_batch, masks))
            self.submitted += 1
    
    def get_tokenized_batch(self, timeout: float = 5.0):
        if not self.epoch_active: return None
        try:
            epoch_tag, batch = self.ready_queue.get(timeout=timeout)
            if epoch_tag != self.epoch_id: return None
            return batch
        except Empty: return None
    
    def shutdown(self):
        """Shutdown the pipeline."""
        self.running = False
        self.epoch_active = False
        # Drain queues to unblock threads
        self._drain_all_queues()
        if self.cpu_thread.is_alive(): self.cpu_thread.join(timeout=2.0)
        if self.gpu_thread and self.gpu_thread.is_alive(): self.gpu_thread.join(timeout=2.0)