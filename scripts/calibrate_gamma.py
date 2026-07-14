#!/usr/bin/env python3
"""Gamma point calibration tool.

Opens an interactive matplotlib window showing a slice of the raw experimental data.
Drag the red vertical line and green horizontal line to center the Gamma point (Γ).
The shift values are printed when you close the window.

Usage:
    uv run python scripts/calibrate_gamma.py --data /path/to/raw.h5

Output:
    The script prints the kx_shift and ky_shift values that should be used
    in the experimental preprocessing configuration.
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
from pimre.utils.interaction import DraggableVLine, DraggableHLine


def main():
    parser = argparse.ArgumentParser(description="Calibrate Gamma point position.")
    parser.add_argument("--data", required=True, help="Path to raw experimental HDF5 file")
    parser.add_argument("--dataset", default="mp", help="HDF5 dataset name (default: mp)")
    parser.add_argument("--layer", type=int, default=10, help="Energy layer index to display (default: 10)")
    parser.add_argument("--kx_range", default=None, help="kx_angle range as 'start,stop,delta,npts'")
    parser.add_argument("--ky_range", default=None, help="ky_angle range as 'start,stop,delta,npts'")
    parser.add_argument("--config", default=None, help="Path to experiment YAML config to update")
    args = parser.parse_args()

    # Load raw data
    with h5py.File(args.data, "r") as f:
        bands = f[args.dataset][:]
    print(f"Loaded data: {bands.shape} (E, kx_angle, ky_angle)")

    # Determine axis ranges
    if args.kx_range:
        parts = [float(x) for x in args.kx_range.split(",")]
        kx_angle = np.linspace(parts[0], parts[0] + parts[1] * (int(parts[2]) - 1), int(parts[2]))
    else:
        kx_angle = np.arange(bands.shape[1])
    if args.ky_range:
        parts = [float(x) for x in args.ky_range.split(",")]
        ky_angle = np.linspace(parts[0], parts[0] + parts[1] * (int(parts[2]) - 1), int(parts[2]))
    else:
        ky_angle = np.arange(bands.shape[2])

    # Show interactive calibration window
    print("\nDrag the red vertical line and green horizontal line to center the Gamma point.")
    print("Close the window when done. The final shift values will be printed.\n")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.figure.set_size_inches(8, 8)
    ax.set_xlabel(f"ky angle ({'deg' if args.ky_range else 'pixel'})")
    ax.set_ylabel(f"kx angle ({'deg' if args.kx_range else 'pixel'})")

    im = ax.imshow(
        bands[args.layer],
        aspect="auto",
        extent=[ky_angle[0], ky_angle[-1], kx_angle[0], kx_angle[-1]],
        cmap="plasma",
        origin="lower",
    )
    plt.colorbar(im, ax=ax, label="Intensity")

    # Initial guess: center of the image
    cx = (ky_angle[0] + ky_angle[-1]) / 2
    cy = (kx_angle[0] + kx_angle[-1]) / 2

    ax.set_title(f"Layer {args.layer} — Drag lines to Γ point, then close window")
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.grid(True)
    fig.canvas.draw()  # ensure canvas initialised before interactive widgets

    vline = DraggableVLine(ax, x=cx, color="red", linestyle="--", linewidth=2)
    hline = DraggableHLine(ax, y=cy, color="green", linestyle="--", linewidth=2)

    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        pass

    kx_shift = hline.y
    ky_shift = vline.x
    print(f"\n=== Calibration Result ===")
    print(f"kx_shift = {kx_shift:.4f}  (horizontal line y-position)")
    print(f"ky_shift = {ky_shift:.4f}  (vertical line x-position)")

    # Update config if requested
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cfg["kx_shift"] = kx_shift
        cfg["ky_shift"] = ky_shift
        with open(args.config, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Updated {args.config} with calibration values.")


if __name__ == "__main__":
    main()