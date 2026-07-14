#!/usr/bin/env python3
"""RbTiBi end-to-end pipeline test with interactive Gamma point calibration.

The pipeline pauses at the calibration step to let you drag lines to
center the Gamma point. Close the calibration window to continue.

Usage:
    uv run python scripts/test_rbtibi_pipeline.py           # interactive mode
    uv run python scripts/test_rbtibi_pipeline.py --skip-calibrate  # use saved shifts
"""

import argparse
import os
import sys
import json
import numpy as np
import h5py
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import scipy.io as sio
from decimal import Decimal
import math
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pimre.config import load_config, save_config as save_cfg, crystallographic_data
from pimre.dft.reader import read_band_gap, read_dft_csv, expand_bz, interpolate_to_grid, save_band_map_mat
from pimre.experiment.calibration import Angle2Mon, KDInterp, RotateCoordinates, save_preprocessed_h5
from pimre.utils.io import loadHDF
from pimre.mrf.model import MrfRec
from pimre.mrf.symmetry import sym_band
from pimre.mrf.evaluation import expand_dft_bands, theory_data_expand
from pimre.kpath.symmetry import Get_G_M_K, dft_KM
from pimre.kpath.path import points2path, bandpath_map as bpm
from pimre.utils.interaction import DraggableVLine, DraggableHLine

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "configs", "pimre_config.yaml")
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test")
os.makedirs(TEST_DIR, exist_ok=True)

# ── helpers ──

def load_calibration():
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE) as f:
            return json.load(f)
    return {"kx_shift": 0.0, "ky_shift": 0.0, "ky_angle_offset": 0.0, "kx_grid_shift": 0.0, "ky_grid_shift": 0.0}


def save_calibration(calib):
    with open(CALIB_FILE, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"  Calibration saved to {CALIB_FILE}")


def interactive_calibrate(bands, kx_angle, ky_angle, layer=10):
    """Open an interactive matplotlib window for Gamma point calibration.

    Drag the red vertical line and green horizontal line to center the
    Gamma point.  The printed values are the shifts to apply.
    """
    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │  Drag lines to center the Gamma point (Γ).  │")
    print("  │  Close the window to continue.              │")
    print("  └─────────────────────────────────────────────┘\n")

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

    ax.set_title("Drag lines to Γ point, then close window")
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.grid(True)
    fig.canvas.draw()
    vline = DraggableVLine(ax, x=cx, color="red", linestyle="--", linewidth=2)
    hline = DraggableHLine(ax, y=cy, color="green", linestyle="--", linewidth=2)
    plt.show(block=True)

    kx_s = hline.y
    ky_s = vline.x
    print(f"\n  kx_shift = {kx_s:.4f}   (horizontal line)")
    print(f"  ky_shift = {ky_s:.4f}   (vertical line)")
    return kx_s, ky_s


def interactive_grid_calibrate(E_Mon_layer, kx, ky):
    """Open interactive window for grid shift calibration in momentum space."""
    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │  Drag lines to center Γ in momentum space.  │")
    print("  │  Close the window to continue.              │")
    print("  └─────────────────────────────────────────────┘\n")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel(r"$k_y$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$k_x$ ($\AA^{-1}$)")

    im = ax.imshow(E_Mon_layer, aspect="auto", extent=[ky[0], ky[-1], kx[0], kx[-1]], cmap="plasma", origin="lower")
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

    kx_gs = hline.y
    ky_gs = vline.x
    print(f"\n  kx_grid_shift = {kx_gs:.4f}   (horizontal line)")
    print(f"  ky_grid_shift = {ky_gs:.4f}   (vertical line)")
    return kx_gs, ky_gs


# ──── Parse args ────
parser = argparse.ArgumentParser(description="RbTiBi end-to-end pipeline test.")
parser.add_argument("--config", default=CONFIG_PATH, help="Config file path")
parser.add_argument("--skip-calibrate", action="store_true", help="Skip calibration, use saved values")
parser.add_argument("--skip-grid-calibrate", action="store_true", help="Skip grid calibration")
parser.add_argument("--layer", type=int, default=10, help="Energy layer for calibration (default: 10)")
args = parser.parse_args()

cfg = load_config(args.config)
calib = cfg["calibration"]
crystal = crystallographic_data(cfg)
ar = cfg["arpes"]
dft_cfg = cfg["dft"]
pp_cfg = cfg["preprocessing"]

# ──── DFT Processing ────
print("=" * 60)
print("1. DFT Processing")
print("=" * 60, flush=True)

dft_dir = dft_cfg["path"]
fermi, vbm, cbm = read_band_gap(os.path.join(dft_dir, dft_cfg["fermi_file"]))
print(f"Fermi={fermi:.4f}, VBM={vbm}, CBM={cbm}", flush=True)

nkx, nky = dft_cfg["k_grid"]
cartesian_coords, energy_bands, ebands = read_dft_csv(
    os.path.join(dft_dir, dft_cfg["csv_file"]), fermi, nkx, nky)
print(f"ebands: {ebands.shape}", flush=True)

bz_coords, repeated_bands = expand_bz(cartesian_coords, energy_bands)
print(f"BZ: {bz_coords.shape[0]} points", flush=True)

dnx, dny = dft_cfg["output_grid"]
mapping, kx_grid, ky_grid = interpolate_to_grid(bz_coords, repeated_bands, dnx, dny)
gap_id = vbm + 1
evb = mapping[:gap_id][::-1]
ecb = mapping[gap_id:]
print(f"evb: {evb.shape}, ecb: {ecb.shape}", flush=True)

band_map_path = os.path.join(TEST_DIR, "band_map.mat")
save_band_map_mat(band_map_path, evb, ecb, kx_grid, ky_grid)
print(f"Saved band_map.mat", flush=True)

# ──── Experimental Preprocessing ────
print("=" * 60)
print("2. Experimental Preprocessing")
print("=" * 60, flush=True)

with h5py.File(ar["path"], "r") as f:
    parts = ar["dataset"].split("/")
    d = f
    for p in parts:
        d = d[p]
    bands = d[:]
print(f"bands: {bands.shape}", flush=True)

E_grid = np.linspace(ar["energy"]["start"],
                     ar["energy"]["start"] + ar["energy"]["delta"] * (ar["energy"]["npts"] - 1),
                     ar["energy"]["npts"])
if ar["energy"].get("flip", False):
    E_grid = E_grid[::-1]
kx_angle = np.linspace(ar["kx_angle"]["start"],
                       ar["kx_angle"]["start"] + ar["kx_angle"]["delta"] * (ar["kx_angle"]["npts"] - 1),
                       ar["kx_angle"]["npts"])
if ar["kx_angle"].get("flip", False):
    kx_angle = kx_angle[::-1]
ky_angle = np.linspace(ar["ky_angle"]["start"],
                       ar["ky_angle"]["start"] + ar["ky_angle"]["delta"] * (ar["ky_angle"]["npts"] - 1),
                       ar["ky_angle"]["npts"])
if ar["ky_angle"].get("flip", False):
    ky_angle = ky_angle[::-1]

if pp_cfg.get("sort_axes", False):
    if E_grid[0] > E_grid[-1]:
        bands = np.flip(bands, axis=0)
        E_grid = E_grid[::-1]
    if kx_angle[0] > kx_angle[-1]:
        bands = np.flip(bands, axis=1)
        kx_angle = kx_angle[::-1]
    if ky_angle[0] > ky_angle[-1]:
        bands = np.flip(bands, axis=2)
        ky_angle = ky_angle[::-1]
    print("  Sorted axes", flush=True)

# ──── Interactive Calibration ────
if not args.skip_calibrate:
    print(f"\n  Current calibration: kx_shift={calib['kx_shift']:.4f}, ky_shift={calib['ky_shift']:.4f}")
    kx_s, ky_s = interactive_calibrate(bands, kx_angle, ky_angle, layer=args.layer)
    calib["kx_shift"] = kx_s
    calib["ky_shift"] = ky_s
    save_cfg(cfg, args.config)

# Apply shifts
kx_angle = kx_angle - calib["kx_shift"]
ky_angle = ky_angle - calib["ky_shift"]
print(f"  Applied: kx_shift={calib['kx_shift']:.4f}, ky_shift={calib['ky_shift']:.4f}", flush=True)

# Angle to momentum
KX, KY = Angle2Mon(E_grid, kx_angle, ky_angle, work_function=ar["work_function"])
print(f"  KX shape: {KX.shape}", flush=True)

if pp_cfg.get("sign_correct", False):
    xmask = kx_angle < 0
    ymask = ky_angle < 0
    KX_abs, KY_abs = Angle2Mon(E_grid, np.abs(kx_angle), np.abs(ky_angle),
                               work_function=ar["work_function"])
    KX_abs[:, xmask, :] *= -1
    KY_abs[:, :, ymask] *= -1
    KX, KY = KX_abs, KY_abs
    print("  Applied sign correction", flush=True)

# Multi-layer expansion
n_rot = pp_cfg["n_rotations"]
bands_rep = np.repeat(bands[:, :, np.newaxis], n_rot, axis=2).reshape(bands.shape[0], bands.shape[1], -1)
KX_rot = np.zeros((KX.shape[0], KX.shape[1], KX.shape[2] * n_rot))
KY_rot = np.zeros_like(KX_rot)
KX_rot[:, :, :KX.shape[2]] = KX
KY_rot[:, :, :KY.shape[2]] = KY
for i in range(1, n_rot):
    kxr, kyr = RotateCoordinates(KX, KY, theta=60 * i)
    KX_rot[:, :, i * KX.shape[2]:(i + 1) * KX.shape[2]] = kxr
    KY_rot[:, :, i * KY.shape[2]:(i + 1) * KY.shape[2]] = kyr

if pp_cfg.get("auto_grid", False):
    n_out = np.max(KX_rot.shape)
else:
    n_out = min(np.max(KX_rot.shape), pp_cfg["output_grid"])
print(f"  Output grid: {n_out}x{n_out}", flush=True)
kx_out = np.linspace(np.min(KX_rot), np.max(KX_rot), n_out)
ky_out = np.linspace(np.min(KY_rot), np.max(KY_rot), n_out)
kxm, kym = np.meshgrid(kx_out, ky_out, indexing="ij")

# Quick single-layer for grid calibration
if not args.skip_grid_calibrate:
    E_temp = KDInterp(bands_rep[args.layer], KX_rot[args.layer], KY_rot[args.layer],
                      radius=pp_cfg["kd_radius"], kx_grid=kxm, ky_grid=kym)
    kx_gs, ky_gs = interactive_grid_calibrate(E_temp, kx_out, ky_out)
    calib["kx_grid_shift"] = kx_gs
    calib["ky_grid_shift"] = ky_gs
    save_cfg(cfg, args.config)

kx_out = kx_out - calib["kx_grid_shift"]
ky_out = ky_out - calib["ky_grid_shift"]
kxm, kym = np.meshgrid(kx_out, ky_out, indexing="ij")

# Full KD-interpolation (use stride for speed)
print(f"  KD-interp on {bands_rep.shape[0]} layers (stride=10)...", flush=True)
E_Mon = np.zeros((bands_rep.shape[0], n_out, n_out))
stride = 10
for i in range(0, bands_rep.shape[0], stride):
    if i % 50 == 0:
        print(f"    layer {i}/{bands_rep.shape[0]}", flush=True)
    E_Mon[i] = KDInterp(bands_rep[i], KX_rot[i], KY_rot[i], radius=0.05, kx_grid=kxm, ky_grid=kym)
for i in range(bands_rep.shape[0]):
    if i % stride != 0:
        lo = (i // stride) * stride
        hi = min(lo + stride, bands_rep.shape[0] - 1)
        frac = (i - lo) / (hi - lo) if hi > lo else 0
        E_Mon[i] = (1 - frac) * E_Mon[lo] + frac * E_Mon[hi]

print(f"  E_Mon: {E_Mon.shape}", flush=True)

prep_path = os.path.join(TEST_DIR, "exp_preprocessed.h5")
save_preprocessed_h5(prep_path, E_grid, kx_out, ky_out, E_Mon)
print(f"  Saved preprocessed data", flush=True)

# Switch to non-interactive backend for batch plotting
plt.switch_backend("Agg")

# ──── MRF Reconstruction ────
print("=" * 60)
print("3. MRF Reconstruction")
print("=" * 60, flush=True)

data = loadHDF(prep_path)
E = data["E"]
kx = data["kx"][1:-1]
ky = data["ky"][1:-1]
I = np.transpose(data["V"][1:-1, 1:-1, 1:-1], (1, 2, 0))
print(f"MRF: E={E.shape}, kx={kx.shape}, ky={ky.shape}, I={I.shape}", flush=True)

mrf = MrfRec(E=E, kx=kx, ky=ky, I=I, eta=0.12)
mrf.smoothenI(sigma=cfg["mrf"]["smooth_sigma"])

mat_data = sio.loadmat(band_map_path)
evb_m = mat_data["evb"][:]
ecb_m = mat_data["ecb"][:]
E_dft = np.nan_to_num(np.vstack((ecb_m[::-1], evb_m)))
E_dft = E_dft[33:]

if np.abs(np.sum(np.diff(mat_data["kxxsc"][:, 0]))) > np.abs(np.sum(np.diff(mat_data["kxxsc"][0, :]))):
    ky_dft = mat_data["kxxsc"][:, 0]
    kx_dft = mat_data["kyysc"][0, :]
else:
    ky_dft = mat_data["kxxsc"][0, :]
    kx_dft = mat_data["kyysc"][:, 0]

E_dft_exp, kx_dft_exp, ky_dft_exp = expand_dft_bands(E_dft, kx_dft, ky_dft, kx, ky)
kx_dft = kx_dft_exp
ky_dft = ky_dft_exp
E_dft = E_dft_exp

G, M, M1, K, K1, K2, K3, K4, K5 = Get_G_M_K(crystal, kx, ky)
KP_dft, MP_dft = dft_KM(kx_dft, ky_dft)
print(f"G={G}, M={M}, K={K}", flush=True)

# Align DFT
kx_dft_0 = kx[M[0]] - ((kx[M[0]] - kx_dft[np.argmin(np.abs(kx_dft))]) / (MP_dft[0] - np.argmin(np.abs(kx_dft)))) * (MP_dft[0] - 1)
kx_dft_step = (kx[M[0]] - kx_dft[np.argmin(np.abs(kx_dft))]) / (MP_dft[0] - np.argmin(np.abs(kx_dft)))
ky_dft_0 = ky[M[1]] - ((ky[M[1]] - ky_dft[np.argmin(np.abs(ky_dft))]) / (MP_dft[1] - np.argmin(np.abs(ky_dft)))) * (MP_dft[1] - 1)
ky_dft_step = (ky[M[1]] - ky_dft[np.argmin(np.abs(ky_dft))]) / (MP_dft[1] - np.argmin(np.abs(ky_dft)))

kx_dft_a = np.array(np.arange(Decimal(kx_dft_0), Decimal(kx_dft_0 + kx_dft.shape[0] * kx_dft_step), Decimal(kx_dft_step)), dtype="float64")
ky_dft_a = np.array(np.arange(Decimal(ky_dft_0), Decimal(ky_dft_0 + ky_dft.shape[0] * ky_dft_step), Decimal(ky_dft_step)), dtype="float64")

E_dft_a = np.zeros((E_dft.shape[0], kx.shape[0], ky.shape[0]))
for ind in range(E_dft.shape[0]):
    E_dft_a[ind] = theory_data_expand(ind, kx_dft_a, ky_dft_a, E_dft, kx, ky, kx.shape[0])

hyperparams = cfg["mrf"]["bands"]
n_bands = len(hyperparams)
recon = np.zeros((n_bands, kx.shape[0], ky.shape[0]))

for ind_band in range(n_bands):
    hp = hyperparams[ind_band]
    print(f"Band {ind_band}: eta={hp['eta']}", flush=True)
    mrf.eta = hp["eta"]
    Einterp = theory_data_expand(ind_band * 2, kx_dft_a, ky_dft_a, E_dft_a, kx, ky, kx.shape[0])
    E0 = np.reshape(Einterp + hp["offset"], (kx.shape[0], ky.shape[0]))
    EE, EE0 = np.meshgrid(E, E0)
    mrf.indEb = np.argmin(np.abs(EE - EE0), 1).reshape(E0.shape)
    recon[ind_band] = mrf.getEb()
    sym_band(ind_band, recon, kx, ky, mrf.lengthKx, mrf.lengthKy)

np.save(os.path.join(TEST_DIR, "recon_bands.npy"), recon)
print(f"Recon saved: {recon.shape}", flush=True)

# ──── Visualizations ────
print("=" * 60)
print("4. Visualizations")
print("=" * 60, flush=True)

colors = ["r", "y", "b", "g", "w"]

# DFT top valence
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(evb[0], extent=(kx_grid.min(), kx_grid.max(), ky_grid.min(), ky_grid.max()), origin="lower", cmap="plasma")
ax.set(title="Top Valence Band (DFT)", xlabel=r"$k_x$ ($\AA^{-1}$)", ylabel=r"$k_y$ ($\AA^{-1}$)")
plt.colorbar(im, label="E-EF (eV)")
fig.savefig(os.path.join(TEST_DIR, "dft_top_valence.png"), dpi=150, bbox_inches="tight")
plt.close()

# Exp layer
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(E_Mon[10], extent=(ky_out[0], ky_out[-1], kx_out[0], kx_out[-1]), origin="lower", cmap="plasma")
ax.set(title="Experimental Data (layer 10)", xlabel=r"$k_y$ ($\AA^{-1}$)", ylabel=r"$k_x$ ($\AA^{-1}$)")
ax.axvline(0, color="red", ls="--", lw=1)
ax.axhline(0, color="red", ls="--", lw=1)
plt.colorbar(im, label="Intensity")
fig.savefig(os.path.join(TEST_DIR, "exp_layer10.png"), dpi=150, bbox_inches="tight")
plt.close()

# Recon bands
fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 4))
if n_bands == 1:
    axes = [axes]
for i in range(n_bands):
    im = axes[i].imshow(recon[i], extent=(ky[0], ky[-1], kx[0], kx[-1]), origin="lower", cmap="viridis", aspect="auto")
    axes[i].set(title=f"Band {i}", xlabel=r"$k_y$ ($\AA^{-1}$)", ylabel=r"$k_x$ ($\AA^{-1}$)")
    plt.colorbar(im, ax=axes[i], label="E (eV)")
plt.tight_layout()
fig.savefig(os.path.join(TEST_DIR, "recon_bands.png"), dpi=150, bbox_inches="tight")
plt.close()

# Band path
nGM = int(math.sqrt((M[0] - G[0]) ** 2 + (M[1] - G[1]) ** 2))
nMK = int(math.sqrt((M[0] - K[0]) ** 2 + (M[1] - K[1]) ** 2))
nKG = int(math.sqrt((K[0] - G[0]) ** 2 + (K[1] - G[1]) ** 2))
path_points = np.asarray([G, M, K, G])
row_inds, col_inds, path_inds = points2path(path_points[:, 0], path_points[:, 1], npoints=[nGM, nMK, nKG])
pathD = bpm(np.transpose(I, (2, 0, 1)), pathr=row_inds, pathc=col_inds, eaxis=0)
prec = bpm(recon, pathr=row_inds, pathc=col_inds, eaxis=0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(pathD, cmap="plasma", extent=[0, len(row_inds), E[0], E[min(109, len(E) - 1)]], aspect="auto", origin="upper")
for ib in range(prec.shape[0]):
    ax.plot(savgol_filter(prec[ib], min(30, len(prec[ib]) - 2), 2), zorder=1, lw=2.3, color=colors[ib], label=f"Band {ib}")
ax.set(xlim=(0, len(row_inds)), ylim=(E[0], E[min(109, len(E) - 1)]))
ax.set_xticks(path_inds)
ax.set_xticklabels([r"$\overline{\Gamma}$", r"$\overline{\mathrm{M}}$", r"$\overline{\mathrm{K}}$", r"$\overline{\Gamma}$"], fontsize=15)
ax.legend()
plt.colorbar(ax.images[0], label="Intensity")
fig.savefig(os.path.join(TEST_DIR, "band_path_GMKG.png"), dpi=150, bbox_inches="tight")
plt.close()

# DFT vs Recon
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
dft_path = np.vstack((ebands[0, :, 2:], ebands[:, -1, 2:], ebands[-1, ::-1, 2:]))
for band in range(105, 130):
    ax1.plot(dft_path[:, band], color="blue", alpha=0.3, lw=0.5)
ax1.set(ylabel="E-EF (eV)", xlabel="k-path", title="DFT Bands (GMKG)", ylim=(-2, 2))
ax1.axhline(0, color="gray", ls="--", lw=0.5)
for ib in range(prec.shape[0]):
    ax2.plot(prec[ib], color=colors[ib], lw=2, label=f"Band {ib}")
ax2.set(ylabel="E (eV)", xlabel="k-path", title="Reconstructed Bands (GMKG)")
ax2.legend()
plt.tight_layout()
fig.savefig(os.path.join(TEST_DIR, "dft_vs_recon.png"), dpi=150, bbox_inches="tight")
plt.close()

# Save params
with open(os.path.join(TEST_DIR, "pipeline_params.txt"), "w") as f:
    f.write(f"Fermi Energy: {fermi:.4f} eV\nVBM={vbm}, CBM={cbm}\n")
    f.write(f"DFT grid: 20x20 -> BZ {bz_coords.shape[0]} pts -> mapping {mapping.shape}\n")
    f.write(f"Exp data: {bands.shape} -> preprocessed {E_Mon.shape}\n")
    f.write(f"MRF: {n_bands} bands, kx={kx.shape}, ky={ky.shape}\n")
    f.write(f"High-symmetry: G={G}, M={M}, K={K}\n")
    f.write(f"Calibration: kx_shift={calib['kx_shift']:.4f}, ky_shift={calib['ky_shift']:.4f}, "
            f"kx_grid_shift={calib['kx_grid_shift']:.4f}, ky_grid_shift={calib['ky_grid_shift']:.4f}\n")
    f.write("Hyperparams:\n")
    for i, hp in enumerate(hyperparams):
        f.write(f"  Band {i}: kScale={hp['k_scale']}, offset={hp['offset']}, eta={hp['eta']}\n")

print("=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
for f in sorted(os.listdir(TEST_DIR)):
    sz = os.path.getsize(os.path.join(TEST_DIR, f))
    print(f"  {f:40s} {sz:>12,d} bytes")