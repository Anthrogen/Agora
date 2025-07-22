from odyssey.src.model_librarian import load_model_from_checkpoint, load_model_from_checkpoint
from odyssey.src.configurations import *
import torch
from odyssey.src.losses import kabsch_rmsd_loss, _kabsch_align

def validate_gradients(ckpt):
    # Run Stage 1 training.  Watch both encoder and decoder parameters update:
    dvc = torch.device("cuda")
    model, model_cfg, train_cfg = load_model_from_checkpoint(ckpt, dvc)

    return

    model = model.to(dvc)

    L = model_cfg.max_len
    base = torch.nn.functional.one_hot(torch.Tensor([0, 1, 2]).long(), num_classes=3).float()

    x = []
    for idx in range(L):
        x.append(base.clone() + torch.ones_like(base) * idx)

    x = torch.stack(x,dim=0)
    x = x.unsqueeze(0)
    x = x.to(dvc)

    y, _ = model(x)

    print("Input:")
    print(x)

    print("Output:")
    print(y)

    print("KABSCH Loss:")



def run_test(ckpt):
    try: 
        validate_gradients(ckpt)
    except AssertionError as e:
        print("Test Failure.")
        return False
    
    print("Test Success.")
    return True


if __name__ == "__main__":
    ckpt = "../../checkpoints/fsq/fsq_stage_2_config/fsq_stage_2_config_000/model.pt"
    run_test(ckpt)
