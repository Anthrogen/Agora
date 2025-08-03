import torch 
import numpy as np
import random

from odyssey.src.model_librarian import load_model_from_checkpoint
from odyssey.src.dataset import ProteinDataset
from odyssey.src.dataloader import ComplexDataLoader, worker_init_fn
from odyssey.src.losses import cross_entropy_loss, calculate_accuracy

from scipy.stats import binned_statistic
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt = '/workspace/demo/Odyssey/checkpoints/transformer_trunk/mlm_complex_config/mlm_complex_config_000/checkpoint_step_12240.pt'

model, model_cfg, train_cfg = load_model_from_checkpoint(ckpt, device)
autoencoder, autoencoder_model_cfg, autoencoder_train_cfg = load_model_from_checkpoint(model_cfg.autoencoder_path, device)

dataset_mode = "backbone"
data_dir = "/workspace/demo/Odyssey/sample_data/3k.csv"

# Use comprehensive tracks like in generate.py
tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'orthologous_groups': True, 'semantic_description': True, 'domains': True, 'plddt': True}
min_unmasked = {'seq': 0, 'coords': 1}

# Use the reference model seed for consistency
data_seed = model_cfg.reference_model_seed
torch.manual_seed(data_seed)
np.random.seed(data_seed)
random.seed(data_seed)

# Create dataset and validation split
g_val = torch.Generator()
g_val.manual_seed(data_seed + 5000)
val_ds = ProteinDataset(data_dir, mode=dataset_mode, max_length=model_cfg.max_len - 2, max_length_orthologous_groups=model_cfg.max_len_orthologous_groups - 2, max_length_semantic_description=model_cfg.max_len_semantic_description - 2)
val_dl = ComplexDataLoader(val_ds, model_cfg, train_cfg, tracks, device, autoencoder=autoencoder, min_unmasked=min_unmasked, batch_size=train_cfg.batch_size, shuffle=True, generator=g_val, worker_init_fn=worker_init_fn)

scatter_struct = []
scatter_seq = []

count = 0
for batch in val_dl:

    if batch is None:
        continue

    count += 1

    B = batch.unmasked_data['seq'].shape[0]


    masked_seq, masked_struct, masked_coords = batch.masked_data['seq'], batch.masked_data['struct'], batch.masked_data['coords']
    ss8_tokens, sasa_tokens = batch.masked_data['ss8'], batch.masked_data['sasa']
    orthologous_groups_tokens, semantic_description_tokens, domains_tokens = batch.unmasked_data['orthologous_groups'], batch.unmasked_data['semantic_description'], batch.masked_data['domains']
    plddt_tokens = batch.masked_data['plddt']
    B, L = masked_seq.shape

    # Create mask for GA/RA/SA/SC models and for SS8/SASA
    content_elements = ~batch.masks['coords'] & ~batch.beospank['coords']
    nonbeospank = ~batch.beospank['coords'] & ~batch.beospank['seq']
    nonbeospank_ss8 = ~batch.beospank['ss8']
    nonbeospank_sasa = ~batch.beospank['sasa']
    nonbeospank_orthologous_groups = ~batch.beospank['orthologous_groups']
    nonbeospank_semantic_description = ~batch.beospank['semantic_description']
    nonbeospank_domains = ~batch.beospank['domains']
    nonbeospank_plddt = ~batch.beospank['plddt']
    assert content_elements.any(dim=1).all()

    inputs = (masked_seq, masked_struct, ss8_tokens, sasa_tokens, orthologous_groups_tokens, semantic_description_tokens, domains_tokens, plddt_tokens) # Prepare model input
    nonbeospanks_all = {'nonbeospank': nonbeospank,
                        'nonbeospank_ss8': nonbeospank_ss8,
                        'nonbeospank_sasa': nonbeospank_sasa,
                        'nonbeospank_orthologous_groups': nonbeospank_orthologous_groups,
                        'nonbeospank_semantic_description': nonbeospank_semantic_description,
                        'nonbeospank_domains': nonbeospank_domains,
                        'nonbeospank_plddt': nonbeospank_plddt}



    model_type = model.cfg.first_block_cfg.initials()
    if model_type in ("GA", "RA"): outputs = model(inputs, masked_coords, content_elements, **nonbeospanks_all)
    else: outputs = model(inputs, **nonbeospanks_all)

    seq_logits, struct_logits = outputs

    if train_cfg.loss_config.loss_elements == "masked":
        loss_elements_seq = batch.masks['seq']
        loss_elements_struct = batch.masks['struct']
    elif train_cfg.loss_config.loss_elements == "non_beospank":
        # Compute loss over all non-BOS/EOS/PAD positions, including masks.
        loss_elements_seq = ~batch.beospank['seq']
        loss_elements_struct = ~batch.beospank['struct']
    elif train_cfg.loss_config.loss_elements == "non_special":
        loss_elements_seq = ~batch.beospank['seq'] & ~batch.masks['seq']
        loss_elements_struct = ~batch.beospank['struct'] & ~batch.masks['struct']
    else: raise ValueError(f"What is {train_cfg.loss_config.loss_elements}?")

    # Find which batch elements have valid loss elements and count effective batch sizes
    valid_seq_mask = loss_elements_seq.any(dim=1)  # [B]
    valid_struct_mask = loss_elements_struct.any(dim=1)  # [B]
    effective_batch_size_seq = valid_seq_mask.sum().item()
    effective_batch_size_struct = valid_struct_mask.sum().item()

    probs_seq = batch.metadata['seq'][valid_seq_mask]
    probs_struct = batch.metadata['coords'][valid_struct_mask] # struct uses metadata of coords

    if effective_batch_size_seq > 0:
        seq_logits_valid = seq_logits[valid_seq_mask]
        seq_labels_valid = batch.unmasked_data['seq'][valid_seq_mask]
        loss_elements_seq_valid = loss_elements_seq[valid_seq_mask]
        loss_seq = cross_entropy_loss(seq_logits_valid, seq_labels_valid, loss_elements_seq_valid, return_all=True)
        seq_acc = calculate_accuracy(seq_logits_valid, seq_labels_valid, loss_elements_seq_valid, return_all=True)
    else:
        loss_seq = torch.tensor([0.0]*B, device=seq_logits.device)
        seq_acc = torch.tensor([0.0]*B, device=seq_logits.device)
    
    if effective_batch_size_struct > 0:
        struct_logits_valid = struct_logits[valid_struct_mask]
        struct_labels_valid = batch.unmasked_data['struct'][valid_struct_mask]
        loss_elements_struct_valid = loss_elements_struct[valid_struct_mask]
        loss_struct = cross_entropy_loss(struct_logits_valid, struct_labels_valid, loss_elements_struct_valid, return_all=True)
        struct_acc = calculate_accuracy(struct_logits_valid, struct_labels_valid, loss_elements_struct_valid, return_all=True)
    else:
        loss_struct = torch.tensor([0.0]*B, device=struct_logits.device)
        struct_acc = torch.tensor([0.0]*B, device=struct_logits.device)



    for idx in range(effective_batch_size_seq):
        scatter_seq.append((probs_seq[idx].item(), loss_seq[idx].item()))
    for idx in range(effective_batch_size_struct):
        scatter_struct.append((probs_struct[idx].item(), loss_struct[idx].item()))

    print('looping...')
    if count > 1000:
        break


scatter_seq = np.array(scatter_seq)
scatter_struct = np.array(scatter_struct)



# Original scatter plots
plt.scatter(scatter_seq[:,0], scatter_seq[:,1], label="Sequence", alpha=0.5)
plt.scatter(scatter_struct[:,0], scatter_struct[:,1], label="Structure", alpha=0.5)

# Calculate conditional expectation for sequence data
bin_means_seq, bin_edges_seq, _ = binned_statistic(
    scatter_seq[:,0], scatter_seq[:,1], statistic='mean', bins=20
)
bin_centers_seq = (bin_edges_seq[:-1] + bin_edges_seq[1:]) / 2

# Calculate conditional expectation for structure data
bin_means_struct, bin_edges_struct, _ = binned_statistic(
    scatter_struct[:,0], scatter_struct[:,1], statistic='mean', bins=20
)
bin_centers_struct = (bin_edges_struct[:-1] + bin_edges_struct[1:]) / 2

# Plot conditional expectations
plt.plot(bin_centers_seq, bin_means_seq, 'b-', linewidth=2, label="E[Loss|Mask] Sequence")
plt.plot(bin_centers_struct, bin_means_struct, 'red', linewidth=2, label="E[Loss|Mask] Structure")

plt.xlabel("Mask Probability")
plt.ylabel("Cross Entropy Loss")
plt.legend()
plt.savefig('results/complex_mask_rate.png')