"""DFT calculation data processing pipeline.

Replicates the workflow from 1.Calu_Data_Processing.ipynb:
1. Read DFT calculation data (CSV) and Fermi energy
2. Coordinate transformation (reciprocal to Cartesian)
3. Brillouin zone expansion (6-fold rotation + reflection)
4. Interpolation to uniform grid
5. Save as band_map.h5
"""

import os

import numpy as np

from pimre.config import load_config
from pimre.dft.reader import (
    build_band_map_3d,
    expand_bz,
    interpolate_to_grid,
    read_band_gap,
    read_dft_csv,
    read_fermi_energy,
    save_band_map_h5,
    save_band_map_mat,
)


def _k_grid(config):
    """Read nkx/nky from config, supporting both top-level (defaults.yaml)
    and nested ``dft:`` (pimre_config.yaml) layouts."""
    kg = config.get("k_grid", None) or config.get("dft", {}).get("k_grid", [20, 20])
    if isinstance(kg, dict):
        return int(kg["nkx"]), int(kg["nky"])
    return int(kg[0]), int(kg[1])


def _bz_params(config):
    """Read BZ grid parameters, defaulting to the pimre_config defaults."""
    bz = config.get("bz", {}) or config.get("dft", {}).get("bz", {})
    return {
        "n_x": int(bz.get("n_x", 70)),
        "n_y": int(bz.get("n_y", 70)),
        "n_plus": int(bz.get("n_plus", 3)),
        "scaling_factor": float(bz.get("scaling_factor", 2)),
    }


def run_dft_pipeline(dft_csv, config_path="configs/defaults.yaml",
                     fermi_file=None, band_gap_file=None,
                     output="band_map.h5", output_format="h5",
                     method="grid_cell"):
    """Process DFT calculation data into a band map.

    Parameters
    ----------
    dft_csv : str
        Path to DFT CSV file.
    config_path : str
        Path to config YAML.
    fermi_file : str or None
        Path to FERMI_ENERGY file.
    band_gap_file : str or None
        Path to BAND_GAP file.
    output : str
        Output file path.
    output_format : str
        'h5' or 'mat'.
    method : str
        'grid_cell' or 'griddata'.

    Returns
    -------
    band_map_path : str
        Path to the saved band map.
    """
    config = load_config(config_path)
    nkx, nky = _k_grid(config)

    if fermi_file:
        fermi_energy = read_fermi_energy(fermi_file)
        print(f"FERMI_ENERGY = {fermi_energy}")
        vbm_index = None; cbm_index = None
    elif band_gap_file:
        fermi_energy, vbm_index, cbm_index = read_band_gap(band_gap_file)
        print(f"Fermi Energy = {fermi_energy}")
        print(f"VBM Band Index = {vbm_index}")
        print(f"CBM Band Index = {cbm_index}")
    else:
        fermi_energy = 0.0; vbm_index = None; cbm_index = None
        print("Warning: No Fermi energy file provided, using 0.0")

    cartesian_coords, energy_bands, ebands = read_dft_csv(dft_csv, fermi_energy, nkx, nky)
    if ebands.shape == energy_bands.shape:
        print(f"Energy bands shape: {ebands.shape} (scattered k-points, no regular {nkx}x{nky} grid)")
    else:
        print(f"Energy bands shape: {ebands.shape}")

    bz_coords, repeated_bands = expand_bz(cartesian_coords, energy_bands)
    print(f"BZ coordinates: {bz_coords.shape[0]} points")

    has_fermi = bool(fermi_file or band_gap_file)
    if has_fermi:
        gamma_row = int(np.argmin(np.linalg.norm(cartesian_coords, axis=1)))
        gamma_energies = energy_bands[gamma_row]
        gap_id = int(np.sum(gamma_energies < 0))
        print(f"Occupied bands at Gamma (E < Fermi): {gap_id}/{energy_bands.shape[1]}")
    else:
        gap_id = vbm_index + 1 if vbm_index is not None else None
        print("Warning: No Fermi energy provided, gap split falls back to band index/VBM")

    if method == "grid_cell":
        bz = _bz_params(config)
        BANDMAP, CARCOO = build_band_map_3d(bz_coords, repeated_bands, **bz)
        print(f"Band map shape: {BANDMAP.shape}")
        if gap_id is None:
            gap_id = BANDMAP.shape[0] // 2
        evb = BANDMAP[:gap_id][::-1]
        ecb = BANDMAP[gap_id:]
        kx_grid = CARCOO[0]
        ky_grid = CARCOO[1]
    else:
        mapping, kx_grid, ky_grid = interpolate_to_grid(bz_coords, repeated_bands)
        if gap_id is None:
            gap_id = mapping.shape[0] // 2
        evb = mapping[:gap_id][::-1]
        ecb = mapping[gap_id:]

    if output_format == "mat":
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        save_band_map_mat(output, evb, ecb, kx_grid, ky_grid)
    else:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        save_band_map_h5(output, evb, ecb, kx_grid, ky_grid)

    print(f"Band map saved to {output}")
    return output
