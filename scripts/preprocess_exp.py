#!/usr/bin/env python3
"""Experimental data preprocessing pipeline.

Usage:
    uv run python scripts/preprocess_exp.py
    uv run python scripts/preprocess_exp.py --skip-calib
    uv run python scripts/preprocess_exp.py --calib-only
    uv run python scripts/preprocess_exp.py --config configs/pimre_config.yaml
"""

import argparse
import os
import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pimre.config import load_config, save_config
from pimre.gui.calibration import GammaCalibrator, GridCalibrator
from pimre.pipeline.preprocess import compute_grid, preprocess_full
from pimre.experiment.calibration import KDInterp

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "configs", "pimre_config.yaml")
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive preprocessing with Gamma calibration.")
    parser.add_argument("--config", default=CONFIG_PATH, help="Config file path")
    parser.add_argument("--skip-calib", action="store_true",
                        help="Use saved calibration values from config")
    parser.add_argument("--calib-only", action="store_true",
                        help="Only calibrate, skip preprocessing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cal = cfg["calibration"]
    ar = cfg["arpes"]

    with h5py.File(ar["path"], "r") as f:
        parts = ar["dataset"].split("/")
        d = f
        for p in parts:
            d = d[p]
        bands = d[:]
    kx_angle = np.linspace(ar["kx_angle"]["start"],
                           ar["kx_angle"]["start"] + ar["kx_angle"]["delta"] * (ar["kx_angle"]["npts"] - 1),
                           ar["kx_angle"]["npts"])
    ky_angle = np.linspace(ar["ky_angle"]["start"],
                           ar["ky_angle"]["start"] + ar["ky_angle"]["delta"] * (ar["ky_angle"]["npts"] - 1),
                           ar["ky_angle"]["npts"])

    if not args.skip_calib:
        print(f"  Current calibration: kx_shift={cal['kx_shift']:.4f}, "
              f"ky_shift={cal['ky_shift']:.4f}")
        gc = GammaCalibrator(bands, kx_angle, ky_angle)
        kx_s, ky_s = gc.run()
        cal["kx_shift"] = kx_s
        cal["ky_shift"] = ky_s
        save_config(cfg, args.config)

    if args.calib_only:
        print("\nCalibration done.  Run without --calib-only to preprocess.")
        return

    E_grid, bands, kx_angle, ky_angle, bands_rep, KX_rot, KY_rot, kx_out, ky_out = compute_grid(cfg)

    if not args.skip_calib:
        kxm, kym = np.meshgrid(kx_out, ky_out, indexing="ij")
        print(f"  KD-interp 1 layer for calibration ...")
        E_temp = KDInterp(bands_rep[10], KX_rot[10], KY_rot[10],
                          radius=cfg["preprocessing"]["kd_radius"],
                          kx_grid=kxm, ky_grid=kym)
        gg = GridCalibrator(E_temp, kx_out, ky_out)
        kx_gs, ky_gs = gg.run()
        cal["kx_grid_shift"] = kx_gs
        cal["ky_grid_shift"] = ky_gs
        save_config(cfg, args.config)
        E_grid, bands, kx_angle, ky_angle, bands_rep, KX_rot, KY_rot, kx_out, ky_out = compute_grid(cfg)

    E_Mon = preprocess_full(cfg, E_grid, bands_rep, KX_rot, KY_rot, kx_out, ky_out)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(bands[10], aspect="auto", cmap="plasma", origin="lower",
               extent=[ky_angle[0], ky_angle[-1], kx_angle[0], kx_angle[-1]])
    ax1.set(title="Raw data (layer 10)", xlabel="ky angle", ylabel="kx angle")
    ax1.axvline(cal["ky_shift"], color="red", ls="--")
    ax1.axhline(cal["kx_shift"], color="green", ls="--")
    ax2.imshow(E_Mon[10], aspect="auto", cmap="plasma", origin="lower",
               extent=[ky_out[0], ky_out[-1], kx_out[0], kx_out[-1]])
    ax2.set(title="Preprocessed (layer 10)",
            xlabel=r"$k_y$ ($\AA^{-1}$)", ylabel=r"$k_x$ ($\AA^{-1}$)")
    ax2.axvline(0, color="red", ls="--", lw=1)
    ax2.axhline(0, color="green", ls="--", lw=1)
    plt.tight_layout()
    fig.savefig(os.path.join(TEST_DIR, "preprocess_preview.png"), dpi=150)
    plt.close()
    print(f"\n  Preview saved → {os.path.join(TEST_DIR, 'preprocess_preview.png')}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()