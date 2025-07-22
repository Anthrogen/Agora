from odyssey.train.train import train
from odyssey.src.model_librarian import load_model_from_checkpoint
from odyssey.src.configurations import *
import torch

def validate_gradients(ckpt):
    # Run Stage 1 training.  Watch both encoder and decoder parameters update:
    dvc = torch.device("cuda")
    if ".pt" in ckpt or ".pkl" in ckpt:
        model, model_cfg, train_cfg = load_model_from_checkpoint(ckpt, dvc)
    elif ".yaml" in ckpt:
        model, model_cfg, train_cfg = load_model_from_empty(dvc)
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt}")

    B = 4
    L = model_cfg.max_len

    print(model)

    x = torch.randn(B, L, 3, 3, device=dvc)
    x.requires_grad = True
    y, _ = model(x)

    y = torch.sum(y, dim=(1, 2))
    for i in range(B):
        scalar_output = y[i].sum()  # Ensure scalar output
        (grad,) = torch.autograd.grad(scalar_output, x, create_graph=True)
        grad = grad.abs().sum(dim=(1, 2, 3))
        print(grad.tolist())

        # Now ensure all elements outside of position "i" are zeros:
        outside_i = grad.sum() - grad[i]
        assert torch.allclose(outside_i, torch.zeros_like(outside_i))


def run_test(ckpt):
    try: 
        validate_gradients(ckpt)
    except AssertionError as e:
        print("Test Failure.")
        return False
    
    print("Test Success.")
    return True


if __name__ == "__main__":
    run_test("../../checkpoints/fsq/fsq_stage_1_config_SA/fsq_stage_1_config_SA_000/checkpoint_step_1.pt")


# THIS TEST WILL FAIL BECAUSE OF THE BATCHNORM IN THE CONV BLOCK
