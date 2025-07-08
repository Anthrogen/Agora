from test_cfg_objs import _stage_1_model_cfg, _stage_1_train_cfg, _stage_2_model_cfg, _stage_2_train_cfg
from odyssey.train.train_tensorized import train
from odyssey.src.model_librarian import load_model_from_checkpoint
import torch


# Run Stage 1 training.  Watch both encoder and decoder parameters update:
ckpt = "../../checkpoints/fsq/SC_stage_1_discrete_diffusion_model.pt"
dvc = torch.device("cuda")
model, model_cfg, train_cfg = load_model_from_checkpoint(ckpt, dvc)



B = 4
L = _stage_1_model_cfg.max_len

x = torch.randn(B, L, 3, 3, device=dvc)
x.requires_grad = True

y, _ = model(x)


y = torch.sum(y, dim=(1, 2))

for i in range(B):
    (grad,) = torch.autograd.grad(y[i], x, create_graph=True)

    print(grad)
