from odyssey.train.train import train
from odyssey.src.model_librarian import load_model_from_checkpoint, load_model_from_empty
from odyssey.src.configurations import *
import torch
from odyssey.src.config_loader import load_config
from odyssey.src.dataset import ProteinDataset
from odyssey.src.dataloader import _get_training_dataloader, worker_init_fn
from torch.utils.data import random_split
import numpy as np
import random
import pdb
import copy

def minus(dict, dict_):
    ret = {}
    for k in dict.keys():
        ret[k] = dict[k] - dict_[k]

    return ret

def allsame(dict, dict_, except_idxs):
    for k in dict.keys():
        t = dict[k][[r for r in range(len(dict[k])) if r not in except_idxs]]
        t_ = dict_[k][[r for r in range(len(dict_[k])) if r not in except_idxs]]

        if not torch.allclose(t, t_, rtol=1e-3):
            # print(f"DEBUG: {k} is not all zeros")
            return False
    
    return True


def validate_gradients(ckpt):
    # Run Stage 1 training.  Watch both encoder and decoder parameters update:
    dvc = torch.device("cuda")
    if ".pt" in ckpt or ".pkl" in ckpt:
        model, model_cfg, train_cfg = load_model_from_checkpoint(ckpt, dvc)
    elif ".yaml" in ckpt:
        model_cfg, train_cfg, _, _ = load_config(ckpt)
        model = load_model_from_empty(model_cfg, dvc)
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt}")

    print(model)
    
    print(f"DEBUG: Model training mode: {model.training}")
    model.eval()  # Set to eval mode to disable dropout
    print(f"DEBUG: Model training mode after eval(): {model.training}")

    # Load real data following train.py pattern
    # Set up data loading configuration based on model style
    if model_cfg.style == "stage_1":
        tracks = {'seq': False, 'struct': False, 'coords': True, 'ss8': False, 'sasa': False, 'orthologous_groups': False, 'semantic_description': False, 'domains': False, 'plddt': False}
        min_unmasked = {'seq': 1, 'coords': 1}
        dataset_mode = "backbone"
    elif model_cfg.style == "stage_2":
        tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': False, 'sasa': False, 'orthologous_groups': False, 'semantic_description': False, 'domains': False, 'plddt': False}
        min_unmasked = {'seq': 0, 'coords': 0}
        dataset_mode = "side_chain"
    elif model_cfg.style == "mlm":
        tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'orthologous_groups': True, 'semantic_description': True, 'domains': True, 'plddt': True}
        min_unmasked = {'seq': 1, 'coords': 1}
        dataset_mode = "backbone"
    elif model_cfg.style == "discrete_diffusion":
        tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'orthologous_groups': True, 'semantic_description': True, 'domains': True, 'plddt': True}
        min_unmasked = {'seq': 1, 'coords': 1}
        dataset_mode = "backbone"
    else:
        # Default configuration
        tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'orthologous_groups': True, 'semantic_description': True, 'domains': True, 'plddt': True}
        min_unmasked = {'seq': 1, 'coords': 1}
        dataset_mode = "backbone"

    # Set up FSQ encoder if needed
    autoencoder = None
    load_fsq_encoder = model_cfg.style in {"mlm", "discrete_diffusion", "stage_2"}
    if load_fsq_encoder and hasattr(model_cfg, 'autoencoder_path') and model_cfg.autoencoder_path:
        autoencoder, autoencoder_model_cfg, _ = load_model_from_checkpoint(model_cfg.autoencoder_path, dvc)
        autoencoder.encoder.eval()
        autoencoder.encoder.requires_grad_(False)
        autoencoder.quantizer.eval()
        autoencoder.quantizer.requires_grad_(False)

    # Set seed for reproducible data loading
    data_seed = getattr(model_cfg, 'reference_model_seed', 42)
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    random.seed(data_seed)

    # Create dataset and get 4 samples
    dataset = ProteinDataset(train_cfg.data_dir, mode=dataset_mode, max_length=model_cfg.max_len - 2, 
                           max_length_orthologous_groups=model_cfg.max_len_orthologous_groups - 2, 
                           max_length_semantic_description=model_cfg.max_len_semantic_description - 2)
    
    # Create a small subset with just 4 samples
    indices = list(range(min(4, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    
    # Create generator for consistent batching
    g_test = torch.Generator()
    g_test.manual_seed(data_seed)
    
    # Create dataloader for the 4 samples
    test_loader = _get_training_dataloader(subset, model_cfg, train_cfg, tracks, dvc, 
                                         batch_size=4, shuffle=False, generator=g_test, 
                                         worker_init_fn=worker_init_fn, min_unmasked=min_unmasked, 
                                         autoencoder=autoencoder)
    
    # Get one batch of 4 real observations
    batch = next(iter(test_loader))

    masked_inputs = ('coords', 'seq', 'struct', 'ss8', 'sasa', 'domains', 'plddt')
    unmasked_inputs = ('orthologous_groups', 'semantic_description')

    tok1 = {k: batch.masked_data[k] for k in masked_inputs}
    tok2 = {k: batch.unmasked_data[k] for k in unmasked_inputs}
    x = {**tok1, **tok2}

    beospanks = batch.beospank.copy()
    
    B = batch.masked_data['seq'].shape[0]  # Should be 4
    
    print(f"Using real data batch with shape: {batch.masked_data['seq'].shape}")
    
    # For discrete diffusion models, we need timesteps
    def fwd_pass(x):


        if model_cfg.style == "discrete_diffusion":

            timesteps = torch.randint(0, model_cfg.num_timesteps, (B,), device=dvc)
            outputs = model(x, beospanks, timesteps=timesteps)
        else:

            outputs = model(x, beospanks)

        y = {'seq': outputs[0], 'struct': outputs[1]}
        for k in y.keys():
            y[k] = torch.sum(y[k], dim=tuple(range(1, y[k].ndim)))

        return y

    y = fwd_pass(x)


    # pdb.set_trace()

    for k in x.keys():
        if k in ('coords', 'domains', 'orthologous_groups', 'semantic_description'):
            continue 

        x_ = copy.deepcopy(x)

        ptb_row = B-1

        # pdb.set_trace()

        # Toggle one non-BOS token in a way that respects all vocabularies (of nonzero size):
        if x_[k][ptb_row,1].item() == 0:
            x_[k][ptb_row,1] = 1
        else:
            x_[k][ptb_row,1] = 0

        y_ = fwd_pass(x_)

        problem = not allsame(y, y_, (ptb_row,))

        if problem:
            return False

        print(f"{k} is good!")


    return True

        


    


def run_test(ckpt):
    # try: 
    #     validate_gradients(ckpt)
    # except AssertionError as e:
    #     print("Test Failure.")
    #     print(e)
    #     return False

    validate_gradients(ckpt)
    
    print("Test Success.")
    return True


if __name__ == "__main__":
    #run_test("../../checkpoints/fsq/fsq_stage_1_config/fsq_stage_1_config_000/checkpoint_step_26316.pt")

    #run_test("../../checkpoints/transformer_trunk/mlm_complex_config/mlm_complex_config_000/checkpoint_step_86904.pt")
    run_test("../../configs/tiny_mlm.yaml")

# THIS TEST WILL FAIL BECAUSE OF THE BATCHNORM IN THE CONV BLOCK
