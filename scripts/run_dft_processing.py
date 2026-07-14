#!/usr/bin/env python3
"""DFT Calculation Data Processing Pipeline.

Replicates the workflow from 1.Calu_Data_Processing.ipynb:
1. Read DFT calculation data (CSV) and Fermi energy
2. Coordinate transformation (reciprocal to Cartesian)
3. Brillouin zone expansion (6-fold rotation + reflection)
4. Interpolation to uniform grid
5. Save as band_map.mat or band_map.h5
"""

import argparse
import os
import sys
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pimre.dft.reader import (
    read_fermi_energy,
    read_band_gap,
    read_dft_csv,
    expand_bz,
    interpolate_to_grid,
    build_band_map_3d,
    extract_high_symmetry_path,
    save_band_map_mat,
    save_band_map_h5,
    reciprocal_to_cartesian_matrix,
)


def main():
    parser = argparse.ArgumentParser(description="Process DFT calculation data into band map.")
    parser.add_argument("--config", default="configs/defaults.yaml", help="Path to default config")
    parser.add_argument("--dft_csv", required=True, help="Path to DFT CSV file")
    parser.add_argument("--fermi_file", default=None, help="Path to FERMI_ENERGY file")
    parser.add_argument("--band_gap_file", default=None, help="Path to BAND_GAP file")
    parser.add_argument("--output", default="band_map.mat", help="Output file path")
    parser.add_argument("--output_format", default="mat", choices=["mat", "h5"], help="Output format")
    parser.add_argument("--method", default="grid_cell", choices=["grid_cell", "griddata"], help="Interpolation method")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    nkx = config["k_grid"]["nkx"]
    nky = config["k_grid"]["nky"]

    # Read Fermi energy
    if args.fermi_file:
        fermi_energy = read_fermi_energy(args.fermi_file)
        print(f"FERMI_ENERGY = {fermi_energy}")
        vbm_index = None
        cbm_index = None
    elif args.band_gap_file:
        fermi_energy, vbm_index, cbm_index = read_band_gap(args.band_gap_file)
        print(f"Fermi Energy = {fermi_energy}")
        print(f"VBM Band Index = {vbm_index}")
        print(f"CBM Band Index = {cbm_index}")
    else:
        fermi_energy = 0.0
        vbm_index = None
        cbm_index = None
        print("Warning: No Fermi energy file provided, using 0.0")

    # Read DFT data
    cartesian_coords, energy_bands, ebands = read_dft_csv(args.dft_csv, fermi_energy, nkx, nky)
    print(f"Energy bands shape: {ebands.shape}")

    # BZ expansion
    bz_coords, repeated_bands = expand_bz(cartesian_coords, energy_bands)
    print(f"BZ coordinates: {bz_coords.shape[0]} points")

    if args.method == "grid_cell":
        # Build 3D band map using grid cell averaging
        n_x = config["bz"]["n_x"]
        n_y = config["bz"]["n_y"]
        n_plus = config["bz"]["n_plus"]
        scaling_factor = config["bz"]["scaling_factor"]

        BANDMAP, CARCOO = build_band_map_3d(bz_coords, repeated_bands, n_x, n_y, n_plus, scaling_factor)
        print(f"Band map shape: {BANDMAP.shape}")

        if vbm_index is not None:
            gap_id = vbm_index + 1
        else:
            gap_id = BANDMAP.shape[0] // 2

        evb = BANDMAP[:gap_id][::-1]
        ecb = BANDMAP[gap_id:]
        kx_grid = CARCOO[0]
        ky_grid = CARCOO[1]
    else:
        # Interpolate using griddata
        mapping, kx_grid, ky_grid = interpolate_to_grid(bz_coords, repeated_bands)
        if vbm_index is not None:
            gap_id = vbm_index + 1
        else:
            gap_id = mapping.shape[0] // 2
        evb = mapping[:gap_id][::-1]
        ecb = mapping[gap_id:]

    # Save
    if args.output_format == "mat":
        save_band_map_mat(args.output, evb, ecb, kx_grid, ky_grid)
    else:
        save_band_map_h5(args.output, evb, ecb, kx_grid, ky_grid)

    print(f"Band map saved to {args.output}")


if __name__ == "__main__":
    main()