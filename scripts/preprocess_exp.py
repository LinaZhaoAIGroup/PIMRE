#!/usr/bin/env python3
"""Interactive experimental data preprocessing with Gamma point calibration.

Loads raw ARPES data, performs angle-to-momentum conversion,
rotational expansion, and KD-interpolation.  Two interactive
calibration stages (Qt5Agg) let you centre the Gamma point.

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pimre.config import load_config, save_config as save_cfg
from pimre.experiment.calibration import (
    Angle2Mon, KDInterp, RotateCoordinates, save_preprocessed_h5,
)
from pimre.utils.interaction import DraggableVLine, DraggableHLine

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "configs", "pimre_config.yaml")
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test")


# ── stage 1: angle-space calibration ──────────────────────────────────

class GammaCalibrator:
    """Interactive Gamma-point calibration in angle space."""

    def __init__(self, bands, kx_angle, ky_angle):
        import matplotlib
        matplotlib.use("Qt5Agg")
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        self.plt = plt
        self.Slider = Slider
        self.bands = bands
        self.kx_angle = kx_angle
        self.ky_angle = ky_angle
        self.n_layers = bands.shape[0]
        self.kx_s = (kx_angle[0] + kx_angle[-1]) / 2
        self.ky_s = (ky_angle[0] + ky_angle[-1]) / 2
        self._build_ui()

    def _build_ui(self):
        self.fig = self.plt.figure("Stage 1: Gamma Point Calibration (Angle Space)", figsize=(10, 9))
        self.ax = self.fig.add_axes([0.12, 0.22, 0.83, 0.72])
        self.ax_slider = self.fig.add_axes([0.15, 0.08, 0.70, 0.04])

        self.layer = self.n_layers // 2
        self.im = self.ax.imshow(
            self.bands[self.layer],
            aspect="auto",
            extent=[self.ky_angle[0], self.ky_angle[-1],
                    self.kx_angle[0], self.kx_angle[-1]],
            cmap="plasma", origin="lower",
        )
        self.ax.set_xlabel("ky angle (deg)")
        self.ax.set_ylabel("kx angle (deg)")
        self.ax.set_title(f"Layer {self.layer} / {self.n_layers-1}")
        self.ax.grid(True, alpha=0.3)
        self.fig.canvas.draw()  # ensure canvas initialised before interactive widgets

        self.vline = DraggableVLine(self.ax, x=self.ky_s, color="red", linestyle="--", linewidth=2)
        self.hline = DraggableHLine(self.ax, y=self.kx_s, color="green", linestyle="--", linewidth=2)

        self.slider = self.Slider(self.ax_slider, "Energy layer", 0, self.n_layers - 1,
                             valinit=self.layer, valstep=1)
        self.slider.on_changed(self._on_slider)

    def _on_slider(self, val):
        self.layer = int(val)
        self.im.set_data(self.bands[self.layer])
        self.ax.set_title(f"Layer {self.layer} / {self.n_layers - 1}")
        self.fig.canvas.draw_idle()

    def run(self):
        print("\n" + "=" * 60)
        print("STAGE 1 – Angle-space Gamma calibration")
        print("=" * 60)
        print("  • Drag the red   vertical line to Gamma's ky position")
        print("  • Drag the green  horizontal line to Gamma's kx position")
        print("  • Use the slider to find the clearest layer")
        print("  • Close the window when done")
        self.fig.canvas.manager.window.raise_()
        self.plt.show(block=True)
        self.kx_s = self.hline.y
        self.ky_s = self.vline.x
        print(f"  → kx_shift = {self.kx_s:.4f}  (horizontal line)")
        print(f"  → ky_shift = {self.ky_s:.4f}  (vertical line)")
        return self.kx_s, self.ky_s


# ── stage 2: momentum-space calibration ───────────────────────────────

class GridCalibrator:
    """Interactive Gamma-point calibration in momentum space."""

    def __init__(self, E_mon_layer, kx, ky):
        import matplotlib
        matplotlib.use("Qt5Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.E_mon_layer = E_mon_layer
        self.kx = kx
        self.ky = ky
        self.kx_gs = (kx[0] + kx[-1]) / 2
        self.ky_gs = (ky[0] + ky[-1]) / 2
        self._build_ui()

    def _build_ui(self):
        self.fig = self.plt.figure("Stage 2: Grid Shift Calibration (Momentum Space)", figsize=(9, 8))
        self.ax = self.fig.add_axes([0.12, 0.10, 0.83, 0.85])

        self.im = self.ax.imshow(
            self.E_mon_layer,
            aspect="auto",
            extent=[self.ky[0], self.ky[-1], self.kx[0], self.kx[-1]],
            cmap="plasma", origin="lower",
        )
        self.ax.set_xlabel(r"$k_y$ ($\AA^{-1}$)")
        self.ax.set_ylabel(r"$k_x$ ($\AA^{-1}$)")
        self.ax.set_title("Drag lines to centre Γ in momentum space")
        self.ax.grid(True, alpha=0.3)
        self.fig.canvas.draw()  # ensure canvas initialised before interactive widgets

        self.vline = DraggableVLine(self.ax, x=self.ky_gs, color="red", linestyle="--", linewidth=2)
        self.hline = DraggableHLine(self.ax, y=self.kx_gs, color="green", linestyle="--", linewidth=2)

    def run(self):
        print("\n" + "=" * 60)
        print("STAGE 2 – Momentum-space grid calibration")
        print("=" * 60)
        print("  • Drag the lines to centre the Gamma point in momentum space")
        print("  • Close the window when done")
        self.fig.canvas.manager.window.raise_()
        self.plt.show(block=True)
        self.kx_gs = self.hline.y
        self.ky_gs = self.vline.x
        print(f"  → kx_grid_shift = {self.kx_gs:.4f}  (horizontal line)")
        print(f"  → ky_grid_shift = {self.ky_gs:.4f}  (vertical line)")
        return self.kx_gs, self.ky_gs


# ── preprocessing pipeline ─────────────────────────────────────────────

def compute_grid(cfg):
    """Compute momentum-space grid (fast, no KD-interpolation)."""
    print("\n" + "=" * 60)
    print("BUILDING MOMENTUM GRID")
    print("=" * 60)

    ar = cfg["arpes"]
    cal = cfg["calibration"]
    pp = cfg["preprocessing"]

    with h5py.File(ar["path"], "r") as f:
        parts = ar["dataset"].split("/")
        d = f
        for p in parts:
            d = d[p]
        bands = d[:]
    print(f"  Raw data: {bands.shape}")

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

    if pp.get("sort_axes", False):
        if E_grid[0] > E_grid[-1]:
            bands = np.flip(bands, axis=0)
            E_grid = E_grid[::-1]
        if kx_angle[0] > kx_angle[-1]:
            bands = np.flip(bands, axis=1)
            kx_angle = kx_angle[::-1]
        if ky_angle[0] > ky_angle[-1]:
            bands = np.flip(bands, axis=2)
            ky_angle = ky_angle[::-1]

    kx_angle = kx_angle - cal["kx_shift"]
    ky_angle = ky_angle - cal["ky_shift"]
    print(f"  Applied shifts: kx={cal['kx_shift']:.4f}, ky={cal['ky_shift']:.4f}")

    KX, KY = Angle2Mon(E_grid, kx_angle, ky_angle,
                       work_function=ar["work_function"])
    print(f"  KX shape: {KX.shape}")

    if pp.get("sign_correct", False):
        xmask = kx_angle < 0
        ymask = ky_angle < 0
        KX_abs, KY_abs = Angle2Mon(E_grid, np.abs(kx_angle), np.abs(ky_angle),
                                   work_function=ar["work_function"])
        KX_abs[:, xmask, :] *= -1
        KY_abs[:, :, ymask] *= -1
        KX, KY = KX_abs, KY_abs

    n_rot = pp["n_rotations"]
    bands_rep = np.repeat(bands[:, :, np.newaxis], n_rot, axis=2)
    bands_rep = bands_rep.reshape(bands.shape[0], bands.shape[1], -1)

    KX_rot = np.zeros((KX.shape[0], KX.shape[1], KX.shape[2] * n_rot))
    KY_rot = np.zeros_like(KX_rot)
    KX_rot[:, :, :KX.shape[2]] = KX
    KY_rot[:, :, :KY.shape[2]] = KY
    for i in range(1, n_rot):
        kxr, kyr = RotateCoordinates(KX, KY, theta=60 * i)
        KX_rot[:, :, i * KX.shape[2]:(i + 1) * KX.shape[2]] = kxr
        KY_rot[:, :, i * KY.shape[2]:(i + 1) * KY.shape[2]] = kyr

    if pp.get("auto_grid", False):
        n_out = np.max(KX_rot.shape)
    else:
        n_out = min(np.max(KX_rot.shape), pp["output_grid"])
    kx_out = np.linspace(np.min(KX_rot), np.max(KX_rot), n_out)
    ky_out = np.linspace(np.min(KY_rot), np.max(KY_rot), n_out)
    kx_out = kx_out - cal["kx_grid_shift"]
    ky_out = ky_out - cal["ky_grid_shift"]
    print(f"  Output grid: {n_out}×{n_out}")
    return E_grid, bands, kx_angle, ky_angle, bands_rep, KX_rot, KY_rot, kx_out, ky_out


def preprocess_full(cfg, E_grid, bands_rep, KX_rot, KY_rot, kx_out, ky_out):
    """KD-interpolation on all layers (slow)."""
    pp = cfg["preprocessing"]
    print(f"  KD-interpolation on {bands_rep.shape[0]} layers (stride={pp['stride']}) ...")
    n_out = kx_out.shape[0]
    kxm, kym = np.meshgrid(kx_out, ky_out, indexing="ij")
    E_Mon = np.zeros((bands_rep.shape[0], n_out, n_out))
    stride = pp["stride"]
    for i in range(0, bands_rep.shape[0], stride):
        if i % 50 == 0:
            print(f"    layer {i}/{bands_rep.shape[0]}")
        E_Mon[i] = KDInterp(bands_rep[i], KX_rot[i], KY_rot[i],
                            radius=pp["kd_radius"], kx_grid=kxm, ky_grid=kym)
    for i in range(bands_rep.shape[0]):
        if i % stride != 0:
            lo = (i // stride) * stride
            hi = min(lo + stride, bands_rep.shape[0] - 1)
            frac = (i - lo) / (hi - lo) if hi > lo else 0
            E_Mon[i] = (1 - frac) * E_Mon[lo] + frac * E_Mon[hi]

    print(f"  E_Mon: {E_Mon.shape}")
    save_preprocessed_h5(pp["output_path"] or os.path.join(TEST_DIR, "exp_preprocessed.h5"),
                         E_grid, kx_out, ky_out, E_Mon)
    print(f"  Saved → {pp['output_path'] or 'test/exp_preprocessed.h5'}")
    return E_Mon


# ── main ────────────────────────────────────────────────────────────────

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

    # ── Stage 1: angle-space calibration ──
    if not args.skip_calib:
        print(f"  Current calibration: kx_shift={cal['kx_shift']:.4f}, "
              f"ky_shift={cal['ky_shift']:.4f}")
        gc = GammaCalibrator(bands, kx_angle, ky_angle)
        kx_s, ky_s = gc.run()
        cal["kx_shift"] = kx_s
        cal["ky_shift"] = ky_s
        save_cfg(cfg, args.config)

    if args.calib_only:
        print("\nCalibration done.  Run without --calib-only to preprocess.")
        return

    # ── Compute grid (fast) ──
    E_grid, bands, kx_angle, ky_angle, bands_rep, KX_rot, KY_rot, kx_out, ky_out = compute_grid(cfg)

    # ── Stage 2: momentum-space calibration (skip if --skip-calib) ──
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
        save_cfg(cfg, args.config)

        # Recompute grid with updated grid shifts
        E_grid, bands, kx_angle, ky_angle, bands_rep, KX_rot, KY_rot, kx_out, ky_out = compute_grid(cfg)

    # ── Full KD-interpolation (slow, only once) ──
    E_Mon = preprocess_full(cfg, E_grid, bands_rep, KX_rot, KY_rot, kx_out, ky_out)

    # ── Quick preview ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
    print(f"  Config:             {args.config}")
    print(f"  kx_shift={cal['kx_shift']:.4f}  ky_shift={cal['ky_shift']:.4f}")
    print(f"  kx_grid_shift={cal['kx_grid_shift']:.4f}  "
          f"ky_grid_shift={cal['ky_grid_shift']:.4f}")


if __name__ == "__main__":
    main()