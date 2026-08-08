#!/usr/bin/env python3
"""Compare two band_map files (.h5 or .mat) side-by-side with interactive controls.

Usage:
    uv run python test/compare_band_map.py
    uv run python test/compare_band_map.py --file1 /path/to/ref.h5 --file2 /path/to/test.h5
"""

import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

matplotlib.use("Qt5Agg")
from matplotlib.widgets import RadioButtons, Slider

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pimre.dft.reader import load_band_map_h5
from pimre.kpath.symmetry import _rotate_around_center, dft_KM


def load_band_map(filepath):
    """Load band_map, return stacked E_dft, evb, ecb, kx, ky.

    Supports .h5 (new format) and .mat (legacy format).
    """
    if filepath.endswith(".h5"):
        return load_band_map_h5(filepath)

    mat = sio.loadmat(filepath)
    evb = np.nan_to_num(mat["evb"][:])
    ecb = np.nan_to_num(mat["ecb"][:])
    kxx = mat["kxxsc"]
    kyy = mat["kyysc"]

    if np.abs(np.sum(np.diff(kxx[:, 0]))) > np.abs(np.sum(np.diff(kxx[0, :]))):
        ky = kxx[:, 0]
        kx = kyy[0, :]
    else:
        ky = kxx[0, :]
        kx = kyy[:, 0]

    E_dft = np.vstack((ecb[::-1], evb))
    return E_dft, evb, ecb, kx, ky


def compute_dft_hsps(kx, ky):
    """Compute all high-symmetry points in the DFT k-grid via C6 rotation.

    Returns dict with keys: G, K0..K5, M0..M5.
    Values are (x_idx, y_idx) tuples.
    """
    (KP_x, KP_y), (MP_x, MP_y) = dft_KM(kx, ky)
    G = (np.argmin(np.abs(kx)), np.argmin(np.abs(ky)))

    K_base = (kx[KP_x], ky[KP_y])
    M_base = (kx[MP_x], ky[MP_y])
    center = (kx[G[0]], ky[G[1]])

    K_pts = [_rotate_around_center(K_base, center, i * 60) for i in range(6)]
    M_pts = [_rotate_around_center(M_base, center, i * 60) for i in range(6)]

    hsps = {"G": G}
    for i in range(6):
        hsps[f"K{i}"] = (np.argmin(np.abs(kx - K_pts[i][0])),
                           np.argmin(np.abs(ky - K_pts[i][1])))
        hsps[f"M{i}"] = (np.argmin(np.abs(kx - M_pts[i][0])),
                           np.argmin(np.abs(ky - M_pts[i][1])))
    return hsps


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
    parser = argparse.ArgumentParser(description="Compare two band_map.mat files")
    parser.add_argument("--file1", default="/home/dengxw/ARPES/Data/band_map.mat")
    parser.add_argument("--file2", default=os.path.join(os.path.dirname(__file__), "band_map.h5"))
    parser.add_argument("--name1", default="Standard (ref)")
    parser.add_argument("--name2", default="PIMRE (test)")
    args = parser.parse_args()

    print(f"Loading: {args.file1}")
    E1, evb1, ecb1, kx1, ky1 = load_band_map(args.file1)
    print(f"  {args.name1}: E_dft={E1.shape}, evb={evb1.shape}, ecb={ecb1.shape}, "
          f"kx=[{kx1[0]:.4f}, {kx1[-1]:.4f}], ky=[{ky1[0]:.4f}, {ky1[-1]:.4f}]")

    print(f"Loading: {args.file2}")
    E2, evb2, ecb2, kx2, ky2 = load_band_map(args.file2)
    print(f"  {args.name2}: E_dft={E2.shape}, evb={evb2.shape}, ecb={ecb2.shape}, "
          f"kx=[{kx2[0]:.4f}, {kx2[-1]:.4f}], ky=[{ky2[0]:.4f}, {ky2[-1]:.4f}]")

    n_bands = max(E1.shape[0], E2.shape[0])
    print(f"  Stacked bands (evb + ecb[::-1]): {n_bands}")

    hsps1 = compute_dft_hsps(kx1, ky1)
    hsps2 = compute_dft_hsps(kx2, ky2)
    print(f"  HSPs: G=({kx1[hsps1['G'][0]]:.3f},{ky1[hsps1['G'][1]]:.3f})  "
          f"K0=({kx1[hsps1['K0'][0]]:.3f},{ky1[hsps1['K0'][1]]:.3f})  "
          f"M0=({kx1[hsps1['M0'][0]]:.3f},{ky1[hsps1['M0'][1]]:.3f})")

    # ── Figure ──
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(f"Band Map Comparison: {args.name1}  vs  {args.name2}", fontsize=14)

    ax1 = fig.add_axes([0.04, 0.20, 0.28, 0.72])
    ax1.set_title(args.name1)
    ax2 = fig.add_axes([0.36, 0.20, 0.28, 0.72])
    ax2.set_title(args.name2)
    ax3 = fig.add_axes([0.68, 0.20, 0.28, 0.72])
    ax3.set_title("Difference (test − ref)")

    # ── Slider ──
    ax_slider = fig.add_axes([0.15, 0.08, 0.70, 0.03])
    slider = Slider(ax_slider, "Band", 0, n_bands - 1, valinit=0, valstep=1)

    # ── Radio buttons ──
    ax_radio = fig.add_axes([0.02, 0.02, 0.10, 0.14])
    radio = RadioButtons(ax_radio, ["Stacked", "evb", "ecb"], active=0)

    current_mode = "Stacked"

    def update_mode(label):
        nonlocal current_mode
        current_mode = label
        update_plot(slider.val)

    radio.on_clicked(update_mode)

    def get_band_data(E, evb, ecb, band_idx, mode):
        if mode == "Stacked":
            if band_idx < E.shape[0]:
                return E[band_idx]
            return np.full_like(E[0], np.nan)
        elif mode == "evb":
            if band_idx < evb.shape[0]:
                return evb[band_idx]
            return np.full_like(evb[0], np.nan)
        elif mode == "ecb":
            if band_idx < ecb.shape[0]:
                return ecb[-1 - band_idx] if band_idx < ecb.shape[0] else ecb[0]
            return np.full_like(ecb[0], np.nan)
        else:
            return None

    # ── Create persistent images & colorbars ──
    extent1 = [kx1[0], kx1[-1], ky1[0], ky1[-1]]
    extent2 = [kx2[0], kx2[-1], ky2[0], ky2[-1]]
    dummy = np.zeros((10, 10))

    im1 = ax1.imshow(dummy, aspect="auto", origin="lower", cmap="RdBu_r", extent=extent1)
    im2 = ax2.imshow(dummy, aspect="auto", origin="lower", cmap="RdBu_r", extent=extent2)
    im3 = ax3.imshow(dummy, aspect="auto", origin="lower", cmap="RdBu_r", extent=extent2)
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    def update_plot(band_idx):
        band_idx = int(band_idx)

        data1 = get_band_data(E1, evb1, ecb1, band_idx, current_mode)
        data2 = get_band_data(E2, evb2, ecb2, band_idx, current_mode)
        if data1 is None or data2 is None:
            fig.canvas.draw_idle()
            return

        vmin = min(np.nanmin(data1), np.nanmin(data2))
        vmax = max(np.nanmax(data1), np.nanmax(data2))

        im1.set_data(data1)
        im1.set_clim(vmin, vmax)
        ax1.set_title(f"{args.name1}  (band {band_idx}, {current_mode})")
        draw_hsps(ax1, hsps1, kx1, ky1)

        im2.set_data(data2)
        im2.set_clim(vmin, vmax)
        ax2.set_title(f"{args.name2}  (band {band_idx}, {current_mode})")
        draw_hsps(ax2, hsps2, kx2, ky2)

        diff = data2 - data1
        v_abs = max(abs(np.nanmin(diff)), abs(np.nanmax(diff))) or 1
        im3.set_data(diff)
        im3.set_clim(-v_abs, v_abs)
        ax3.set_title(f"Diff (band {band_idx})")
        draw_hsps(ax3, hsps2, kx2, ky2)

        fig.canvas.draw_idle()

    slider.on_changed(update_plot)
    update_plot(0)

    print("\nControls:")
    print("  Slider:      change band index")
    print("  Radio (Stacked/evb/ecb): view mode")
    print("  Close window to exit")
    plt.show()


if __name__ == "__main__":
    main()
