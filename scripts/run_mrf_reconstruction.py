#!/usr/bin/env python3
"""MRF Band Reconstruction Pipeline.

Replicates the workflow from 4.mrf.ipynb:
1. Load experimental preprocessed data
2. Load DFT band map data
3. Expand DFT data to match experimental grid
4. Find high-symmetry K, M points
5. Align DFT coordinates to experimental coordinates
6. MRF reconstruction with hyperparameters
7. Band structure symmetrization
8. Evaluation and plotting
"""

import argparse
import os
import sys
import yaml
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pimre.utils.io import loadHDF
from pimre.mrf.model import MrfRec
from pimre.mrf.evaluation import (
    expand_dft_bands,
    align_dft_to_exp,
    run_mrf_reconstruction_single_band,
)
from pimre.mrf.symmetry import sym_band
from pimre.kpath.symmetry import Get_G_M_K, dft_KM
from pimre.kpath.corrector import find_MM


def main():
    parser = argparse.ArgumentParser(description="MRF band reconstruction.")
    parser.add_argument("--config", default="configs/mrf_hyperparams.yaml", help="MRF hyperparams config")
    parser.add_argument("--exp_data", required=True, help="Preprocessed experimental HDF5 file")
    parser.add_argument("--band_map", required=True, help="DFT band map .mat file")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--use_momentum_corrector", action="store_true", help="Use MomentumCorrector for M points")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load experimental data
    data = loadHDF(args.exp_data)
    E = data["E"]
    kx = data["kx"][1:-1]
    ky = data["ky"][1:-1]
    I = data["V"][1:-1, 1:-1, 1:-1]

    # Create MRF model
    mrf = MrfRec(E=E, kx=kx, ky=ky, I=I, eta=0.12)
    mrf.smoothenI(sigma=(0.5, 0.5, 0.5 / 5))

    # Load DFT band map
    mat_data = sio.loadmat(args.band_map)
    evb = mat_data["evb"][:]
    ecb = mat_data["ecb"][:]
    E_dft = np.nan_to_num(np.vstack((ecb[::-1, :, :], evb)))
    E_dft = E_dft[33:]

    if np.abs(np.sum(np.diff(mat_data["kxxsc"][:, 0]))) > np.abs(np.sum(np.diff(mat_data["kxxsc"][0, :]))):
        ky_dft = mat_data["kxxsc"][:, 0]
        kx_dft = mat_data["kyysc"][0, :]
    else:
        ky_dft = mat_data["kxxsc"][0, :]
        kx_dft = mat_data["kyysc"][:, 0]

    print(f"DFT bands: {E_dft.shape}, kx_dft: {kx_dft.shape}, ky_dft: {ky_dft.shape}")

    # Expand DFT bands
    E_dft_expanded, kx_dft_expanded, ky_dft_expanded = expand_dft_bands(E_dft, kx_dft, ky_dft, kx, ky)
    kx_dft = kx_dft_expanded
    ky_dft = ky_dft_expanded
    E_dft = E_dft_expanded

    # Find high-symmetry points
    if args.use_momentum_corrector:
        pouter_ord = find_MM(I)
        M = pouter_ord[2]
        M1 = pouter_ord[5]
        # Use gamma from grid center
        G = (np.argmin(np.abs(kx)), np.argmin(np.abs(ky)))
        K = (G[0] + int(M[0] * 0.58), G[1] + int(M[1] * 0.58))
        K1 = (G[0] - int(M[0] * 0.58), G[1] + int(M[1] * 0.58))
    else:
        crystallographic_data = cfg.get("crystallographic_data", [5.8077, 5.8077, 9.1297, 90, 90, 120])
        G, M, M1, K, K1, K2, K3, K4, K5 = Get_G_M_K(crystallographic_data, kx, ky)
        print(f"G={G}, M={M}, M1={M1}, K={K}, K1={K1}")

    # DFT K,M points
    KP_dft, MP_dft = dft_KM(kx_dft, ky_dft)

    # Align DFT coordinates
    kx_dft_aligned, ky_dft_aligned = align_dft_to_exp(kx, ky, kx_dft, ky_dft, M, MP_dft, G)

    # Re-expand DFT bands on aligned coordinates
    E_dft_aligned = np.zeros((E_dft.shape[0], kx.shape[0], ky.shape[0]))
    for ind in range(E_dft.shape[0]):
        from pimre.mrf.evaluation import theory_data_expand
        E_dft_aligned[ind] = theory_data_expand(ind, kx_dft_aligned, ky_dft_aligned, E_dft, kx, ky, kx.shape[0])

    # Run MRF reconstruction
    hyperparams = cfg["bands"]
    colors = ["r", "y", "b", "g", "w"]
    n_bands = len(hyperparams)
    recon = np.zeros((n_bands, kx.shape[0], ky.shape[0]))

    for ind_band in range(n_bands):
        hp = hyperparams[ind_band]
        hyperparam = np.array([hp["index"], hp["k_scale"], hp["offset"], hp["eta"]])

        print(f"Reconstructing band {ind_band} (eta={hp['eta']})...")

        mrf.eta = hyperparam[3]
        Einterp = theory_data_expand(
            ind_band * 2, kx_dft_aligned, ky_dft_aligned, E_dft_aligned, kx, ky, kx.shape[0]
        )
        E0 = np.reshape(Einterp + hyperparam[2], (kx.shape[0], ky.shape[0]))
        EE, EE0 = np.meshgrid(E, E0)
        ind1d = np.argmin(np.abs(EE - EE0), 1)
        mrf.indEb = ind1d.reshape(E0.shape)

        recon[ind_band, ...] = mrf.getEb()
        sym_band(ind_band, recon, kx, ky, mrf.lengthKx, mrf.lengthKy)

    # Save results
    np.save(os.path.join(args.output_dir, "recon_bands.npy"), recon)
    np.savez(os.path.join(args.output_dir, "recon_bands.npz"), recon=recon, kx=kx, ky=ky, E=E)
    print(f"Reconstruction saved to {args.output_dir}")

    # Plot reconstructed bands
    fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 4))
    if n_bands == 1:
        axes = [axes]
    for i in range(n_bands):
        im = axes[i].imshow(recon[i], aspect="auto", origin="lower", cmap="viridis",
                            extent=[ky[0], ky[-1], kx[0], kx[-1]])
        axes[i].set_title(f"Band {i}")
        axes[i].set_xlabel(r"$k_y$ ($\AA^{-1}$)")
        axes[i].set_ylabel(r"$k_x$ ($\AA^{-1}$)")
        plt.colorbar(im, ax=axes[i], label="E (eV)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "recon_bands.png"), dpi=150)
    print("Done.")


if __name__ == "__main__":
    main()