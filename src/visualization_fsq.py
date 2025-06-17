"""
Visualization utilities for backbone reconstruction.

Functions:
  - write_backbone_pdb: dump [L,3,3] coords to a PDB file
  - save_backbone_comparison: write two PDBs & a combined 3D‐scatter PNG
"""

import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  used by plt projection
from src.losses import _kabsch_align

def write_backbone_pdb(coords,
                       pdb_path: str,
                       chain_id: str = "A",
                       res_name: str = "GLY"):
    """
    Write a backbone-only PDB file from coords.
    Args:
      coords: np.ndarray or torch.Tensor, shape [L,3,3] or [1,L,3,3]
      pdb_path: output filename (*.pdb)
      chain_id: one‐letter chain ID
      res_name: three‐letter residue name
    """
    # unwrap tensor
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    # remove batch dim if present
    if coords.ndim == 4 and coords.shape[0] == 1:
        coords = coords[0]
    if coords.ndim != 3 or coords.shape[1:] != (3, 3):
        raise ValueError(f"Expected coords [L,3,3], got {coords.shape}")

    atom_names = ["N", "CA", "C"]
    lines = []
    atom_index = 1
    for i, residue in enumerate(coords, start=1):
        for atom_i, atom_name in enumerate(atom_names):
            x, y, z = residue[atom_i]
            lines.append(
                f"ATOM  {atom_index:5d} {atom_name:^4s}{res_name:>3s} "
                f"{chain_id}{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                "  1.00  0.00           C\n"
            )
            atom_index += 1

    os.makedirs(os.path.dirname(pdb_path) or ".", exist_ok=True)
    with open(pdb_path, "w") as f:
        f.writelines(lines)


def save_backbone_comparison(orig,
                             recon,
                             out_dir: str,
                             prefix: str,
                             png_path: str):
    """
    For one sample, write:
      {out_dir}/{prefix}_orig.pdb
      {out_dir}/{prefix}_recon.pdb
    and a single 3D‐scatter PNG at png_path.
    """
    # — align reconstruction to original via Kabsch —
    # ensure orig/recon are torch tensors on CPU
    if not torch.is_tensor(orig):
        orig_t = torch.tensor(orig, dtype=torch.float32)
    else:
        orig_t = orig.detach().cpu()
    if not torch.is_tensor(recon):
        recon_t = torch.tensor(recon, dtype=torch.float32)
    else:
        recon_t = recon.detach().cpu()

    # flatten to [1, N, 3] where N = L * 3 atoms
    orig_flat  = orig_t.reshape(1, -1, 3)
    recon_flat = recon_t.reshape(1, -1, 3)

    # rotate centered recon → orig
    recon_flat_aligned = _kabsch_align(recon_flat, orig_flat)

    # add back orig centroid
    orig_centroid = orig_flat.mean(dim=1, keepdim=True)    # [1,1,3]
    recon_aligned  = recon_flat_aligned + orig_centroid   # [1,N,3]

    # reshape back to [L,3,3]
    L     = orig_t.shape[0]
    recon = recon_aligned.squeeze(0).reshape(L, 3, 3)

    # 1) Write PDBs (orig stays unmodified; recon is now aligned)
    os.makedirs(out_dir, exist_ok=True)
    orig_pdb  = os.path.join(out_dir, f"{prefix}_orig.pdb")
    recon_pdb = os.path.join(out_dir, f"{prefix}_recon.pdb")
    write_backbone_pdb(orig_t, orig_pdb)
    write_backbone_pdb(recon,  recon_pdb)
    print(f"Wrote PDBs:\n  {orig_pdb}\n  {recon_pdb}")

    # 2) Build scatter plot
    # flatten to [N,3]
    if isinstance(orig, torch.Tensor):
        orig_pts = orig.detach().cpu().numpy().reshape(-1, 3)
        recon_pts = recon.detach().cpu().numpy().reshape(-1, 3)
    else:
        orig_pts = orig.reshape(-1, 3)
        recon_pts = recon.reshape(-1, 3)

    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(orig_pts[:,0], orig_pts[:,1], orig_pts[:,2],
               c="blue", s=4, label="orig")
    ax.scatter(recon_pts[:,0], recon_pts[:,1], recon_pts[:,2],
               c="red",  s=4, label="recon")
    ax.set_title(f"{prefix} reconstruction")
    ax.legend()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved scatter PNG to {png_path}") 