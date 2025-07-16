import os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field, asdict, replace
from typing import Optional, Tuple, Callable, Dict
import random
from types import SimpleNamespace
import argparse

# Import the model and data loader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from odyssey.src.models.autoencoder import Autoencoder, StandardTransformerBlock
from odyssey.src.models.transformer import TransformerTrunk
from odyssey.src.models.autoencoder import FSQEncoder
from odyssey.src.dataloader import MaskedBatch, SimpleDataLoader, ComplexDataLoader, DiffusionDataLoader, NoMaskDataLoader, _get_training_dataloader, worker_init_fn
from odyssey.src.dataset import ProteinDataset
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from odyssey.src.losses import kabsch_rmsd_loss, squared_kabsch_rmsd_loss
from odyssey.src.configurations import TransformerConfig, TrainingConfig
from odyssey.train.fsq_step import stage_1_step, stage_2_step
from odyssey.train.mlm_step import mlm_step
from odyssey.train.discrete_diffusion_step import discrete_diffusion_step
from odyssey.src.configurations import *
from odyssey.src.config_loader import load_config, load_multi_configs
from odyssey.src.model_librarian import ensure_identical_parameters_transformers, ensure_identical_parameters_autoencoders, load_model_from_empty, load_model_from_checkpoint, save_model_checkpoint, save_summary_history

from odyssey.train.yaml_expander import expand_yaml_to_directory
from odyssey.train.generate_experiment_map import generate_experiment_map


def generate(model_checkpoint, callback=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg, train_cfg = load_model_from_checkpoint(model_checkpoint, device)

    transformer, fsq_encoder, fsq_decoder = None, None, None

    assert isinstance(model, TransformerTrunk), "Model must be a TransformerTrunk"




    # Use different masking strategies for stage 1 vs stage 2
    elif model_cfg.style == "mlm":
        tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'global_annotation': True, 'per_residue_annotation': True, 'plddt': True}
        min_unmasked = {'seq': 0, 'coords': 1}
    elif model_cfg.style == "discrete_diffusion":
        tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'global_annotation': True, 'per_residue_annotation': True, 'plddt': True}
        min_unmasked = {'seq': 0, 'coords': 1}
    
    ###########################################################################
    #  Data Loading
    ###########################################################################
    # Set seed for dataset split
    data_seed = model_cfg.reference_model_seed
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    random.seed(data_seed)

    # Create DataLoaders with fixed seed for consistent masking
    g_train = torch.Generator()
    g_train.manual_seed(data_seed)
    g_val = torch.Generator()
    g_val.manual_seed(data_seed + 5000)

    dataset = ProteinDataset(train_cfg.data_dir, mode=dataset_modes[i], max_length=model_cfg.max_len - 2, max_length_global=model_cfg.max_len_global - 2)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size

    # We use g_val as the generator of the split
    _, data = random_split(dataset, [train_size, val_size], generator=g_val)


    # Do not mask anything.
    val_loader = NoMaskDataLoader(data, model_cfg, train_cfg, tracks, device, batch_size=train_cfg.batch_size, shuffle=False, generator=g_val, 
                                        worker_init_fn=worker_init_fn, min_unmasked=min_unmasked_list[i], 
                                        fsq_encoder=fsq_encoder)

    
    for batch in val_loader:
        generate_mlm(model, model_cfg, train_cfg, batch)





def generate_mlm(model, model_cfg, train_cfg, batch):
    """
    Hard-coded parameters:
    unmask_per_pass = 3
    unmask_strategy = "max_confidence" # or "uniform" or "prob_confidence"
    # max_confidence = unmask the three tokens of which you are most confident.
    # prob_confidence = unmask the three tokens probabilistically, with probability proportional to confidence.
    # uniform = unmask three tokens uniformly at random.
    remask_per_pass # Not yet implemented.
    """

    # How to directly modify the batch? 
    logits = model(batch.masked_data['seq'], batch.masked_data['struct'], batch.masked_data['coords'], batch.masked_data['ss8'], batch.masked_data['sasa'], batch.masked_data['global_annotation'], batch.masked_data['per_residue_annotation'], batch.masked_data['plddt'])



    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Odyssey models')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    args = parser.parse_args()
    
    # Create expanded directory name based on config file - save to configs folder
    config_path = Path(args.config)
    yaml_name = config_path.stem
    # Go up to the project root and into configs/expanded (base directory)
    expanded_base_dir = Path(__file__).parent.parent.parent / "checkpoints"
    expanded_yaml_dir = expanded_base_dir / yaml_name
    
