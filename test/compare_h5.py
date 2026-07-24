#!/usr/bin/env python3
"""Compare two preprocessed HDF5 files side-by-side with interactive controls.

Usage:
    uv run python test/compare_h5.py
    uv run python test/compare_h5.py --file1 /path/to/file1.h5 --file2 /path/to/file2.h5
"""

import argparse
import os
import sys

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Qt5Agg")
from matplotlib.widgets import RadioButtons, Slider

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pimre.kpath.symmetry import find_hsps_robust
from pimre.utils.io import loadHDF

CRYSTAL_DATA = [5.8077, 5.8077, 9.1297, 90, 90, 120]


def load_and_normalize(filepath, dataset=None):
    """Load HDF5 data and return (E, kx, ky, V) in (E, kx, ky) format."""
    if dataset is not None:
        with h5py.File(filepath, "r") as f:
            d = f
            for p in dataset.split("/"):
                d = d[p]
            V = d[:]
        return None, None, None, V, "raw"

    data = loadHDF(filepath)
    E = data["E"]
    kx = data["kx"]
    ky = data["ky"]
    V = data["V"]

    if V.shape[0] == E.shape[0]:
        V_t = V
        fmt = "(E, kx, ky)"
    elif V.shape[0] == kx.shape[0]:
        V_t = np.transpose(V, (2, 0, 1))
        fmt = "(kx, ky, E)"
    else:
        V_t = np.transpose(V, (2, 0, 1))
        fmt = "unknown"

    return E, kx, ky, V_t, fmt


def draw_hsps(ax, hsps, kx, ky):
    """Draw high-symmetry point markers on an axis."""
    def _style(name):
        if name == "G":
            return ("o", "white", 12, "Γ")
        if name.startswith("K"):
            return ("D", "magenta", 8, name)
        if name.startswith("M"):
            return ("s", "cyan", 10, name)
        return ("o", "gray", 6, name)

    for name, (ix, iy) in hsps.items():
        if ix < 0 or ix >= len(kx) or iy < 0 or iy >= len(ky):
            continue
        marker, color, size, label = _style(name)
        ax.scatter(kx[ix], ky[iy], marker=marker, c=color, s=size**2,
                   edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(label, (kx[ix], ky[iy]), textcoords="offset points",
                    xytext=(3, 3), fontsize=8, color="white",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.6))


def main():
    parser = argparse.ArgumentParser(description="Compare two HDF5 files")
    parser.add_argument("--file1", default="/home/dengxw/ARPES/Data/HPES_preprocessed_new.h5")
    parser.add_argument("--file2", default=os.path.join(os.path.dirname(__file__), "..", "test", "exp_preprocessed.h5"))
    parser.add_argument("--name1", default="HPES (ref)")
    parser.add_argument("--name2", default="PIMRE (test)")
    parser.add_argument("--crystal", nargs=6, type=float, default=CRYSTAL_DATA,
                        help="Lattice: a b c alpha beta gamma")
    args = parser.parse_args()

    print(f"Loading file 1: {args.file1}")
    E1, kx1, ky1, V1, fmt1 = load_and_normalize(args.file1)
    print(f"  {args.name1}: V={V1.shape}, E={E1.shape if E1 is not None else 'N/A'}, "
          f"kx={kx1.shape if kx1 is not None else 'N/A'}, ky={ky1.shape if ky1 is not None else 'N/A'}, format={fmt1}")

    print(f"Loading file 2: {args.file2}")
    E2, kx2, ky2, V2, fmt2 = load_and_normalize(args.file2)
    print(f"  {args.name2}: V={V2.shape}, E={E2.shape if E2 is not None else 'N/A'}, "
          f"kx={kx2.shape if kx2 is not None else 'N/A'}, ky={ky2.shape if ky2 is not None else 'N/A'}, format={fmt2}")

    def normalize_intensity(V):
        v = V - V.min()
        v = v / v.max() if v.max() > 0 else v
        return v

    V1_norm = normalize_intensity(V1)
    V2_norm = normalize_intensity(V2)

    V1_sum = np.sum(V1_norm, axis=0)
    V2_sum = np.sum(V2_norm, axis=0)
    V1_sum_norm = normalize_intensity(V1_sum)
    V2_sum_norm = normalize_intensity(V2_sum)

    if kx1 is not None and ky1 is not None and V1_norm is not None:
        result1 = find_hsps_robust(V1_norm, kx1, ky1, args.crystal, E1)
        hsps1 = result1.hsps
        G_idx = hsps1["G"]
        print(f"  {args.name1} HSPs (confidence={result1.confidence:.3f}, "
              f"source={result1.source}, layer={result1.best_layer}):")
        print(f"    G=({kx1[G_idx[0]]:.3f},{ky1[G_idx[1]]:.3f})  "
              f"K0=({kx1[hsps1['K0'][0]]:.3f},{ky1[hsps1['K0'][1]]:.3f})  "
              f"M0=({kx1[hsps1['M0'][0]]:.3f},{ky1[hsps1['M0'][1]]:.3f})")
    else:
        hsps1 = None

    if kx2 is not None and ky2 is not None and V2_norm is not None:
        result2 = find_hsps_robust(V2_norm, kx2, ky2, args.crystal, E2)
        hsps2 = result2.hsps
        G_idx = hsps2["G"]
        print(f"  {args.name2} HSPs (confidence={result2.confidence:.3f}, "
              f"source={result2.source}, layer={result2.best_layer}):")
        print(f"    G=({kx2[G_idx[0]]:.3f},{ky2[G_idx[1]]:.3f})  "
              f"K0=({kx2[hsps2['K0'][0]]:.3f},{ky2[hsps2['K0'][1]]:.3f})  "
              f"M0=({kx2[hsps2['M0'][0]]:.3f},{ky2[hsps2['M0'][1]]:.3f})")
    else:
        hsps2 = None

    fig = plt.figure(figsize=(21, 7))
    fig.suptitle(f"File Comparison: {args.name1}  vs  {args.name2}", fontsize=14)

    ax1 = fig.add_axes([0.04, 0.20, 0.28, 0.72])
    ax1.set_title(args.name1)
    ax2 = fig.add_axes([0.34, 0.20, 0.28, 0.72])
    ax2.set_title(args.name2)
    ax3 = fig.add_axes([0.64, 0.55, 0.32, 0.37])
    ax3.set_title(f"{args.name1}  Σ E (all layers)")
    ax4 = fig.add_axes([0.64, 0.20, 0.32, 0.37])
    ax4.set_title(f"{args.name2}  Σ E (all layers)")

    ax_slider = fig.add_axes([0.15, 0.08, 0.70, 0.03])
    max_layer = max(V1.shape[0], V2.shape[0]) - 1
    slider = Slider(ax_slider, "Layer", 0, max_layer, valinit=0, valstep=1)

    ax_radio = fig.add_axes([0.02, 0.02, 0.08, 0.12])
    radio = RadioButtons(ax_radio, ["E", "kx", "ky"], active=0)

    current_axis = 0

    def update_axis(label):
        nonlocal current_axis
        current_axis = {"E": 0, "kx": 1, "ky": 2}[label]
        update_plot(slider.val)

    radio.on_clicked(update_axis)

    def update_plot(layer):
        layer = int(layer)

        ax1.clear()
        ax2.clear()

        if current_axis == 0 and layer < V1_norm.shape[0]:
            slice1 = V1_norm[layer, :, :].T
            ext1 = [kx1[0], kx1[-1], ky1[0], ky1[-1]] if kx1 is not None else [0, V1.shape[1], 0, V1.shape[2]]
            xl1, yl1 = "kx", "ky"
        elif current_axis == 1 and layer < V1_norm.shape[1]:
            slice1 = V1_norm[:, layer, :].T
            ext1 = [E1[0], E1[-1], ky1[0], ky1[-1]] if E1 is not None and ky1 is not None else [0, V1.shape[0], 0, V1.shape[2]]
            xl1, yl1 = "E", "ky"
        elif current_axis == 2 and layer < V1_norm.shape[2]:
            slice1 = V1_norm[:, :, layer].T
            ext1 = [E1[0], E1[-1], kx1[0], kx1[-1]] if E1 is not None and kx1 is not None else [0, V1.shape[0], 0, V1.shape[1]]
            xl1, yl1 = "E", "kx"
        else:
            slice1 = V1_norm[min(layer, V1_norm.shape[0]-1), :, :].T
            ext1 = [0, V1.shape[1], 0, V1.shape[0]]
            xl1, yl1 = "dim1", "dim2"

        ax1.imshow(slice1, aspect="auto", origin="lower", cmap="plasma", extent=ext1)
        ax1.set_title(f"{args.name1}  (layer {layer}, axis={['E','kx','ky'][current_axis]})")
        ax1.set_xlabel(xl1)
        ax1.set_ylabel(yl1)
        if current_axis == 0 and hsps1 is not None:
            draw_hsps(ax1, hsps1, kx1, ky1)

        if current_axis == 0 and layer < V2_norm.shape[0]:
            slice2 = V2_norm[layer, :, :].T
            ext2 = [kx2[0], kx2[-1], ky2[0], ky2[-1]] if kx2 is not None else [0, V2.shape[1], 0, V2.shape[2]]
            xl2, yl2 = "kx", "ky"
        elif current_axis == 1 and layer < V2_norm.shape[1]:
            slice2 = V2_norm[:, layer, :].T
            ext2 = [E2[0], E2[-1], ky2[0], ky2[-1]] if E2 is not None and ky2 is not None else [0, V2.shape[0], 0, V2.shape[2]]
            xl2, yl2 = "E", "ky"
        elif current_axis == 2 and layer < V2_norm.shape[2]:
            slice2 = V2_norm[:, :, layer].T
            ext2 = [E2[0], E2[-1], kx2[0], kx2[-1]] if E2 is not None and kx2 is not None else [0, V2.shape[0], 0, V2.shape[1]]
            xl2, yl2 = "E", "kx"
        else:
            slice2 = V2_norm[min(layer, V2_norm.shape[0]-1), :, :].T
            ext2 = [0, V2.shape[1], 0, V2.shape[0]]
            xl2, yl2 = "dim1", "dim2"

        ax2.imshow(slice2, aspect="auto", origin="lower", cmap="plasma", extent=ext2)
        ax2.set_title(f"{args.name2}  (layer {layer}, axis={['E','kx','ky'][current_axis]})")
        ax2.set_xlabel(xl2)
        ax2.set_ylabel(yl2)
        if current_axis == 0 and hsps2 is not None:
            draw_hsps(ax2, hsps2, kx2, ky2)

        ax3.clear()
        ax4.clear()
        ext_sum1 = [kx1[0], kx1[-1], ky1[0], ky1[-1]] if kx1 is not None else [0, V1_sum.shape[1], 0, V1_sum.shape[0]]
        ax3.imshow(V1_sum_norm.T, aspect="auto", origin="lower", cmap="plasma", extent=ext_sum1)
        ax3.set_xlabel("kx")
        ax3.set_ylabel("ky")
        if kx1 is not None and hsps1 is not None:
            draw_hsps(ax3, hsps1, kx1, ky1)

        ext_sum2 = [kx2[0], kx2[-1], ky2[0], ky2[-1]] if kx2 is not None else [0, V2_sum.shape[1], 0, V2_sum.shape[0]]
        ax4.imshow(V2_sum_norm.T, aspect="auto", origin="lower", cmap="plasma", extent=ext_sum2)
        ax4.set_xlabel("kx")
        ax4.set_ylabel("ky")
        if kx2 is not None and hsps2 is not None:
            draw_hsps(ax4, hsps2, kx2, ky2)

        fig.canvas.draw_idle()

    slider.on_changed(update_plot)
    update_plot(0)

    print("\nControls:")
    print("  Slider: change layer index")
    print("  Radio (E/kx/ky): switch slice axis — HSPs shown on E slice")
    print("  Close window to exit")
    plt.show()


if __name__ == "__main__":
    main()
