import json
from pathlib import Path
from typing import Union, Sequence

# -----------------------------------------------------------------------------
# Main conversion routine
# -----------------------------------------------------------------------------
def convert_json_to_cif(
    json_source: Union[str, Path, dict],
    cif_destination: Union[str, Path],
    *,
    occupancy: float = 1.0,
    b_factor: float = 0.0,
    model_num: int = 1
) -> None:
    """
    Convert a JSON structure (as in the example) to an mmCIF file.

    Parameters
    ----------
    json_source : str | Path | dict
        Path to the JSON file or a pre‑loaded dict with the same keys.
    cif_destination : str | Path
        Where to write the mmCIF file.
    occupancy : float, optional
        Occupancy value written for every atom (default 1.0).
    b_factor : float, optional
        B‑factor (temperature factor) written for every atom (default 0.0).
    model_num : int, optional
        The model number inserted in the _atom_site.pdbx_PDB_model_num column.
    """
    # ---------------------------------------------------------------------
    # Parse the JSON
    # ---------------------------------------------------------------------
    if isinstance(json_source, (str, Path)):
        with open(json_source, "r") as fh:
            data = json.load(fh)
    else:  # assume dict‑like
        data = json_source

    struct_id = data.get("structure_id", "structure")
    chain_id = data["chain_id"]
    atom_names: Sequence[str] = data["all_atoms"]["atom_names"]
    coords: Sequence[Sequence[float]] = data["all_atoms"]["atom_coordinates"]
    residue_ids: Sequence[str] = data["all_atoms"]["residue_ids"]

    if not (len(atom_names) == len(coords) == len(residue_ids)):
        raise ValueError("atom_names, atom_coordinates and residue_ids lengths differ!")

    # ---------------------------------------------------------------------
    # Build CIF lines
    # ---------------------------------------------------------------------
    cif_lines = []

    # Header
    cif_lines.append(f"data_{struct_id}")
    cif_lines.append("#")
    cif_lines.append("loop_")
    cif_lines.extend([
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_PDB_model_num",
    ])

    # Atom records
    for idx, (atom, xyz, res) in enumerate(zip(atom_names, coords, residue_ids), start=1):
        # Attempt to split residue id into name + number (e.g. "ASP893" -> "ASP", 893)
        res_name, res_num = _split_residue(res)
        type_symbol = atom[0].upper()  # crude: first letter of atom name
        line = (
            f"ATOM {idx} {type_symbol:>2} {atom:<4} {res_name:<3} "
            f"{chain_id} {res_num:>4} "
            f"{xyz[0]:>9.3f} {xyz[1]:>8.3f} {xyz[2]:>8.3f} "
            f"{occupancy:>6.2f} {b_factor:>6.2f} {model_num}"
        )
        cif_lines.append(line)

    cif_lines.append("#")  # mmCIF delimiter

    # ---------------------------------------------------------------------
    # Write the file
    # ---------------------------------------------------------------------
    cif_path = Path(cif_destination)
    cif_path.write_text("\n".join(cif_lines))
    print(f"Wrote {len(atom_names)} atoms to {cif_path.resolve()}")

# -----------------------------------------------------------------------------
# Helper: split residue identifier
# -----------------------------------------------------------------------------
def _split_residue(res_id: str):
    """
    Heuristically split residue identifier into (res_name, res_num).

    Examples
    --------
    "ASP893" -> ("ASP", 893)
    "D893"   -> ("UNK", 893)   # if three‑letter code missing
    """
    import re

    m = re.match(r"([A-Za-z]+)(\d+)", res_id)
    if not m:  # fallback
        return ("UNK", 0)
    name, num = m.groups()
    name = name.upper()
    if len(name) < 3:  # one‑letter or chain prefix
        name = "UNK"
    return name[:3], int(num)

# -----------------------------------------------------------------------------
# Convenience CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Convert custom JSON to mmCIF")
    ap.add_argument("json_in", help="input JSON file")
    ap.add_argument("cif_out", help="output mmCIF file")
    args = ap.parse_args()
    convert_json_to_cif(args.json_in, args.cif_out)
