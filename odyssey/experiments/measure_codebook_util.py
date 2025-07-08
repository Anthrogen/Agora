import torch
from odyssey.src.models.autoencoder import Autoencoder
from odyssey.src.model_librarian import load_model_from_checkpoint
from odyssey.src.dataset import ProteinDataset
from odyssey.src.dataloader import _get_training_dataloader, worker_init_fn

ckpt = "../../checkpoints/fsq/SC_stage_1_discrete_diffusion_model.pt"
dvc = torch.device("cuda")
model, model_cfg, train_cfg = load_model_from_checkpoint(ckpt, dvc)

# print(f"Loaded model of type {type(model)}.")
# quit()

dataset = ProteinDataset(train_cfg.data_dir, mode="backbone", max_length=model_cfg.max_len - 2)

train_loader = _get_training_dataloader(dataset, model_cfg, train_cfg, {'seq': False, 'struct': False, 'coords': True}, dvc, 
                                               batch_size=train_cfg.batch_size, shuffle=True, 
                                               worker_init_fn=worker_init_fn, min_unmasked={'seq': 0, 'struct': 0, 'coords': 1})

prod = 1
for i in range(len(model_cfg.fsq_levels)):
    prod *= model_cfg.fsq_levels[i]

print(f"Length of indices: {prod}")

util = torch.zeros(prod, device=dvc)

for batch in train_loader:

    x = batch.masked_data['coords'][:,:,:3,:]

    _, y = model(x)

    for j in y:
        util[j] = 1

print(f"Utilization: {util.sum() / prod}")
