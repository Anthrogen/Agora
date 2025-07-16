from odyssey.src.losses import kabsch_rmsd_loss, squared_kabsch_rmsd_loss
import torch

def make_random_rotation_matrix(dimension, device='cpu', dtype=torch.float32):
    """
    Generate random 3x3 rotation matrices using QR decomposition.
    
    Args:
        batch_size: Number of rotation matrices to generate
        device: torch device ('cpu' or 'cuda')
        dtype: torch dtype (float32 or float64)
    
    Returns:
        Tensor of shape (batch_size, 3, 3) containing rotation matrices
    """
    # Generate random 3x3 matrices
    random_matrices = torch.randn(1, dimension, dimension, device=device, dtype=dtype)
    
    # Perform QR decomposition
    Q, R = torch.linalg.qr(random_matrices)
    
    # Ensure determinant is +1 (not -1)
    # If det(Q) = -1, multiply first column by -1
    det = torch.det(Q)
    Q[:, :, 0] *= det.sign().unsqueeze(-1)
    
    return Q.squeeze(0)

def kabsch(x, y):
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    return squared_kabsch_rmsd_loss([x], [y])


if __name__ == "__main__":
    x = torch.randn(20,3)
    x2 = torch.randn(10, 3)

    R = make_random_rotation_matrix(3, device='cpu', dtype=torch.float32)
    print(R)

    # rot_self = x @ R
    # noise = torch.randn_like(x)
    # rot_self_w_noise = rot_self + 0.01*noise

    # print(f"RMSD w/ rotated self: {kabsch(x, rot_self)}")
    # print(f"RMSD w/ pure noise: {kabsch(x, noise)}")
    # print(f"RMSD w/ rotated self + noise: {kabsch(x, rot_self_w_noise)}")

    # print(f"RMSD w/ rotated self: {kabsch(x, rot_self)}")
    # print(f"RMSD w/ pure noise: {kabsch(x, noise)}")
    # print(f"RMSD w/ rotated self + noise: {kabsch(x, rot_self_w_noise)}")


    y = x @ R
    y2 = x2 @ R

    x_ = [x, x2]
    y_ = [y, y2]

    noise = torch.randn_like(y)
    noise2 = torch.randn_like(y2)

    noise_ = [noise, noise2]
    y_plus_noise_ = [y + noise, y2 + noise2]


    print(f"RMSD w/ rotated self: {kabsch(x_, y_)}")
    print(f"RMSD w/ pure noise: {kabsch(x_, noise_)}")
    print(f"RMSD w/ rotated self + noise: {kabsch(x_, y_plus_noise_)}")
