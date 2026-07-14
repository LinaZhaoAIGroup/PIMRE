#!/usr/bin/env python3
"""Experimental Data Preprocessing Pipeline.

Replicates the workflow from 2.exp_data_pre.ipynb:

1. Load raw experimental HDF5 data
2. (Optional) Interactive Gamma point calibration
3. Angle-to-momentum conversion
4. (Optional) Interactive grid shift calibration
5. Multi-layer rotation and expansion
6. KD-interpolation for all layers
7. Save preprocessed data as HDF5

Usage:
    # Full pipeline
    uv run python scripts/run_exp_preprocessing.py --config configs/experiment_rbtibi.yaml

    # Interactive calibration only
    uv run python scripts/run_exp_preprocessing.py --config configs/experiment_rbtibi.yaml --calibrate

    # Calibrate grid shifts after angle-to-momentum conversion
    uv run python scripts/run_exp_preprocessing.py --config configs/experiment_rbtibi.yaml --calibrate-grid
"""

import argparse
import json
import os
import sys
import yaml
import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pimre.experiment.calibration import (
    Angle2Mon,
    KDInterp,
    RotateCoordinates,
    save_preprocessed_h5,
)

CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test", "calibration.json")


def load_raw_data(filepath, dataset):
    """Load raw experimental HDF5 data, supports nested paths like 'group/subgroup'."""
    parts = dataset.split("/") if isinstance(dataset, str) else [dataset]
    with h5py.File(filepath, "r") as f:
        d = f
        for p in parts:
            d = d[p]
        bands = d[:]
    return bands


def build_axes(cfg, bands):
    """Build energy and angle axes from config and data shape."""
    E_grid = np.linspace(
        cfg["energy_start"],
        cfg["energy_start"] + cfg["energy_delta"] * (bands.shape[0] - 1),
        bands.shape[0],
    )
    if cfg.get("energy_flip", False):
        E_grid = E_grid[::-1]

    kx_angle = np.linspace(
        cfg["kx_angle_start"],
        cfg["kx_angle_start"] + cfg["kx_angle_delta"] * (bands.shape[1] - 1),
        bands.shape[1],
    )
    if cfg.get("kx_angle_flip", False):
        kx_angle = kx_angle[::-1]

    ky_angle = np.linspace(
        cfg["ky_angle_start"],
        cfg["ky_angle_start"] + cfg["ky_angle_delta"] * (bands.shape[2] - 1),
        bands.shape[2],
    )
    if cfg.get("ky_angle_flip", False):
        ky_angle = ky_angle[::-1]

    return E_grid, kx_angle, ky_angle


def run_calibration(bands, kx_angle, ky_angle, layer=10):
    """Interactive Gamma point calibration.

    Opens a matplotlib window with draggable lines.  Drag the red
    vertical line and green horizontal line to center the Gamma point,
    then close the window.
    """
    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    from pimre.utils.interaction import DraggableVLine, DraggableHLine

    print("\n" + "=" * 60)
    print("GAMMA POINT CALIBRATION")
    print("=" * 60)
    print("Drag the red vertical line and green horizontal line to")
    print("center the Gamma point (Γ). Close the window when done.")
    print("The shift values will be printed below.\n")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel("ky angle (deg)")
    ax.set_ylabel("kx angle (deg)")

    im = ax.imshow(
        bands[layer],
        aspect="auto",
        extent=[ky_angle[0], ky_angle[-1], kx_angle[0], kx_angle[-1]],
        cmap="plasma",
        origin="lower",
    )
    plt.colorbar(im, ax=ax, label="Intensity")

    cx = (ky_angle[0] + ky_angle[-1]) / 2
    cy = (kx_angle[0] + kx_angle[-1]) / 2

    ax.set_title(f"Layer {layer} — Drag lines to Γ point, then close window")
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.grid(True)
    fig.canvas.draw()
    vline = DraggableVLine(ax, x=cx, color="red", linestyle="--", linewidth=2)
    hline = DraggableHLine(ax, y=cy, color="green", linestyle="--", linewidth=2)
    plt.show(block=True)

    kx_shift = hline.y
    ky_shift = vline.x
    print(f"\nkx_shift = {kx_shift:.4f}   (horizontal line y-position)")
    print(f"ky_shift = {ky_shift:.4f}   (vertical line x-position)")
    return kx_shift, ky_shift


def run_grid_calibration(E_Mon_layer, kx, ky):
    """Interactive grid shift calibration in momentum space.

    Opens a matplotlib window with draggable lines on the
    momentum-space image.  Drag lines to center the Gamma point
    in momentum space.
    """
    import matplotlib.pyplot as plt
    from pimre.utils.interaction import DraggableVLine, DraggableHLine

    print("\n" + "=" * 60)
    print("GRID SHIFT CALIBRATION (momentum space)")
    print("=" * 60)
    print("Drag the red vertical line and green horizontal line to")
    print("center the Gamma point (Γ) in momentum space.")
    print("Close the window when done.\n")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel(r"$k_y$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$k_x$ ($\AA^{-1}$)")

    im = ax.imshow(
        E_Mon_layer,
        aspect="auto",
        extent=[ky[0], ky[-1], kx[0], kx[-1]],
        cmap="plasma",
        origin="lower",
    )
    plt.colorbar(im, ax=ax, label="Intensity")

    cx = (ky[0] + ky[-1]) / 2
    cy = (kx[0] + kx[-1]) / 2

    ax.set_title("Drag lines to Γ point in momentum space, then close window")
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.grid(True)
    fig.canvas.draw()
    vline = DraggableVLine(ax, x=cx, color="red", linestyle="--", linewidth=2)
    hline = DraggableHLine(ax, y=cy, color="green", linestyle="--", linewidth=2)
    plt.show(block=True)

    kx_grid_shift = hline.y
    ky_grid_shift = vline.x
    print(f"\nkx_grid_shift = {kx_grid_shift:.4f}   (horizontal line y-position)")
    print(f"ky_grid_shift = {ky_grid_shift:.4f}   (vertical line x-position)")
    return kx_grid_shift, ky_grid_shift


def main():
    parser = argparse.ArgumentParser(description="Preprocess experimental ARPES data.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to experiment config")
    parser.add_argument("--exp_data", default=None, help="Path to experimental HDF5 file (overrides config)")
    parser.add_argument("--output", default=None, help="Output file path (overrides config)")
    parser.add_argument("--calibrate", action="store_true", help="Interactive Gamma point calibration")
    parser.add_argument("--calibrate-grid", action="store_true", help="Interactive grid shift calibration")
    parser.add_argument("--layer", type=int, default=10, help="Energy layer for calibration (default: 10)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load calibration from JSON if available (overrides YAML defaults)
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE) as f:
            cal = json.load(f)
        if cal.get("kx_shift", 0) != 0:
            cfg.setdefault("kx_shift", cal["kx_shift"])
        if cal.get("ky_shift", 0) != 0:
            cfg.setdefault("ky_shift", cal["ky_shift"])
        if cal.get("kx_grid_shift", 0) != 0:
            cfg.setdefault("kx_grid_shift", cal["kx_grid_shift"])
        if cal.get("ky_grid_shift", 0) != 0:
            cfg.setdefault("ky_grid_shift", cal["ky_grid_shift"])

    work_function = cfg.get("work_function", 16.03)

    # Load raw data
    filepath = args.exp_data or cfg.get("exp_data", os.path.join(cfg["exp_path"], f"{cfg['exp_name']}.h5"))
    dataset = cfg.get("dataset", cfg["wave_name"] if "wave_name" in cfg else "mp")
    bands = load_raw_data(filepath, dataset)
    print(f"Loaded data: {bands.shape}")

    # Build axes
    E_grid, kx_angle, ky_angle = build_axes(cfg, bands)
    print(f"E_grid: [{E_grid[0]:.4f}, {E_grid[-1]:.4f}] ({len(E_grid)} pts)")
    print(f"kx_angle: [{kx_angle[0]:.4f}, {kx_angle[-1]:.4f}] ({len(kx_angle)} pts)")
    print(f"ky_angle: [{ky_angle[0]:.4f}, {ky_angle[-1]:.4f}] ({len(ky_angle)} pts)")

    # --- Sort axes (notebook-style: flip both label and data) ---
    if cfg.get("sort_axes", False):
        if E_grid[0] > E_grid[-1]:
            bands = np.flip(bands, axis=0)
            E_grid = E_grid[::-1]
        if kx_angle[0] > kx_angle[-1]:
            bands = np.flip(bands, axis=1)
            kx_angle = kx_angle[::-1]
        if ky_angle[0] > ky_angle[-1]:
            bands = np.flip(bands, axis=2)
            ky_angle = ky_angle[::-1]
        print("  Sorted axes (flipped data where needed)")

    # --- Calibration Step ---
    if args.calibrate:
        kx_shift, ky_shift = run_calibration(bands, kx_angle, ky_angle, layer=args.layer)
        cfg["kx_shift"] = kx_shift
        cfg["ky_shift"] = ky_shift
        with open(args.config, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Updated {args.config} with calibration values.")
        return  # calibration-only mode

    # Apply angle shifts
    angle_offset = cfg.get("ky_angle_offset", 0)
    ky_angle = ky_angle - angle_offset
    kx_angle = kx_angle - cfg.get("kx_shift", 0)
    ky_angle = ky_angle - cfg.get("ky_shift", 0)

    # Angle to momentum conversion
    KX, KY = Angle2Mon(E_grid, kx_angle, ky_angle, work_function=work_function)
    print(f"KX shape: {KX.shape}")

    if cfg.get("sign_correct", False):
        xmask = kx_angle < 0
        ymask = ky_angle < 0
        KX_abs, KY_abs = Angle2Mon(E_grid, np.abs(kx_angle), np.abs(ky_angle), work_function=work_function)
        KX_abs[:, xmask, :] *= -1
        KY_abs[:, :, ymask] *= -1
        KX, KY = KX_abs, KY_abs
        print("  Applied sign correction")

    # Multi-layer expansion
    bands_rep = np.repeat(bands[:, :, np.newaxis], 6, axis=2)
    bands_rep = bands_rep.reshape(bands.shape[0], bands.shape[1], bands.shape[2] * 6)

    KX_rot = np.zeros((KX.shape[0], KX.shape[1], KX.shape[2] * 6))
    KY_rot = np.zeros_like(KX_rot)
    KX_rot[:, :, : KX.shape[2]] = KX
    KY_rot[:, :, : KY.shape[2]] = KY
    for i in range(1, 6):
        kxr, kyr = RotateCoordinates(KX, KY, theta=60 * i)
        KX_rot[:, :, i * KX.shape[2] : (i + 1) * KX.shape[2]] = kxr
        KY_rot[:, :, i * KY.shape[2] : (i + 1) * KY.shape[2]] = kyr

    # Build output grid
    n_out = np.max(KX_rot.shape)  # use max dimension for square grid
    kx = np.linspace(np.min(KX_rot), np.max(KX_rot), n_out)
    ky = np.linspace(np.min(KY_rot), np.max(KY_rot), n_out)

    # --- Grid Calibration Step ---
    if args.calibrate_grid:
        # Do a quick single-layer KD-interp for the calibration image
        kxm, kym = np.meshgrid(kx, ky, indexing="ij")
        E_temp = KDInterp(bands_rep[args.layer], KX_rot[args.layer], KY_rot[args.layer], radius=0.05, kx_grid=kxm, ky_grid=kym)
        kx_grid_shift, ky_grid_shift = run_grid_calibration(E_temp, kx, ky)
        cfg["kx_grid_shift"] = kx_grid_shift
        cfg["ky_grid_shift"] = ky_grid_shift
        with open(args.config, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Updated {args.config} with grid shift values.")
        return  # calibration-only mode

    # Apply grid shifts
    kx = kx - cfg.get("kx_grid_shift", 0)
    ky = ky - cfg.get("ky_grid_shift", 0)
    kxm, kym = np.meshgrid(kx, ky, indexing="ij")

    # KD-interpolation
    print(f"KD-interpolation on {bands_rep.shape[0]} layers...")
    E_Mon = np.zeros((bands_rep.shape[0], n_out, n_out))
    for i in range(bands_rep.shape[0]):
        if i % 20 == 0:
            print(f"  layer {i}/{bands_rep.shape[0]}")
        E_Mon[i] = KDInterp(bands_rep[i], KX_rot[i], KY_rot[i], radius=0.05, kx_grid=kxm, ky_grid=kym)

    print(f"Preprocessed data: {E_Mon.shape}")
    print(f"kx range: [{kx.min():.4f}, {kx.max():.4f}]")
    print(f"ky range: [{ky.min():.4f}, {ky.max():.4f}]")

    # Save
    output_path = args.output or cfg.get("output", os.path.join(cfg["exp_path"], f"{cfg['exp_name']}_MON_PIMRE.h5"))
    save_preprocessed_h5(output_path, E_grid, kx, ky, E_Mon)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()