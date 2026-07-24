"""DFT calculation data processing pipeline.

Replicates the workflow from 1.Calu_Data_Processing.ipynb:
1. Read DFT calculation data (CSV) and Fermi energy
2. Coordinate transformation (reciprocal to Cartesian)
3. Brillouin zone expansion (6-fold rotation + reflection)
4. Interpolation to uniform grid
5. Save as band_map.h5
"""

import os

import yaml
import numpy as np

from pimre.dft.reader import (
    read_fermi_energy,
    read_band_gap,
    read_dft_csv,
    expand_bz,
    interpolate_to_grid,
    build_band_map_3d,
    save_band_map_mat,
    save_band_map_h5,
)


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
    with open(config_path) as f:
        config = yaml.safe_load(f)

    nkx = config["k_grid"]["nkx"]
    nky = config["k_grid"]["nky"]

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
    print(f"Energy bands shape: {ebands.shape}")

    bz_coords, repeated_bands = expand_bz(cartesian_coords, energy_bands)
    print(f"BZ coordinates: {bz_coords.shape[0]} points")

    if method == "grid_cell":
        n_x = config["bz"]["n_x"]
        n_y = config["bz"]["n_y"]
        n_plus = config["bz"]["n_plus"]
        scaling_factor = config["bz"]["scaling_factor"]
        BANDMAP, CARCOO = build_band_map_3d(bz_coords, repeated_bands, n_x, n_y, n_plus, scaling_factor)
        print(f"Band map shape: {BANDMAP.shape}")
        gap_id = vbm_index + 1 if vbm_index is not None else BANDMAP.shape[0] // 2
        evb = BANDMAP[:gap_id][::-1]
        ecb = BANDMAP[gap_id:]
        kx_grid = CARCOO[0]
        ky_grid = CARCOO[1]
    else:
        mapping, kx_grid, ky_grid = interpolate_to_grid(bz_coords, repeated_bands)
        gap_id = vbm_index + 1 if vbm_index is not None else mapping.shape[0] // 2
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