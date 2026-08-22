"""Experimental data preprocessing pipeline.

Replicates the workflow from 2.exp_data_pre.ipynb:
1. Load raw experimental HDF5 data
2. Angle-to-momentum conversion
3a. (method="kdtree") Multi-layer rotation and KD-interpolation
3b. (method="quadrant") 1/4-BZ crop, mirror symmetrization, binning
4. Save preprocessed data as HDF5
"""

import h5py
import numpy as np

from pimre.experiment.calibration import (
    Angle2Mon,
    KDInterp,
    RotateCoordinates,
    quadrant_symmetrize,
    save_preprocessed_h5,
)


def compute_grid(cfg):
    """Compute momentum-space grid (fast, no KD-interpolation).

    Parameters
    ----------
    cfg : dict
        Full PIMRE config dict.

    Returns
    -------
    E_grid, bands, kx_angle, ky_angle, bands_rep, KX_rot, KY_rot, kx_out, ky_out
    """
    ar = cfg["arpes"]
    cal = cfg["calibration"]
    pp = cfg["preprocessing"]
    method = pp.get("method", "kdtree")
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

    if method in ("quadrant", "direct"):
        bands_rep = bands
        KX_rot = KX
        KY_rot = KY
    else:
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

    if method == "direct":
        # Use the momentum-space pixel grid at the reference energy layer
        # as the output axes (same resolution as the angular grid); each
        # energy layer keeps its own intensity on these pixels.
        ref = KX.shape[0] // 2
        kx_out = KX_rot[ref, :, KX.shape[2] // 2]
        ky_out = KY_rot[ref, KX.shape[1] // 2, :]
        kx_out = kx_out - cal["kx_grid_shift"]
        ky_out = ky_out - cal["ky_grid_shift"]
        print(f"  Output grid: {kx_out.size}×{ky_out.size} (method=direct, "
              f"reference layer {ref})")
    else:
        if pp.get("auto_grid", False):
            n_out = np.max(KX_rot.shape)
        else:
            n_out = min(np.max(KX_rot.shape), pp["output_grid"])
        if method == "quadrant":
            kx_max = float(np.max(np.abs(KX)))
            ky_max = float(np.max(np.abs(KY)))
            kx_out = np.linspace(-kx_max, kx_max, n_out)
            ky_out = np.linspace(-ky_max, ky_max, n_out)
        else:
            kx_out = np.linspace(np.min(KX_rot), np.max(KX_rot), n_out)
            ky_out = np.linspace(np.min(KY_rot), np.max(KY_rot), n_out)
        kx_out = kx_out - cal["kx_grid_shift"]
        ky_out = ky_out - cal["ky_grid_shift"]
        print(f"  Output grid: {n_out}×{n_out} (method={method})")

    return E_grid, bands, kx_angle, ky_angle, bands_rep, KX_rot, KY_rot, kx_out, ky_out


def preprocess_full(cfg, E_grid, bands_rep, KX_rot, KY_rot, kx_out, ky_out):
    """KD-interpolation on all layers (slow).

    Parameters
    ----------
    cfg : dict
        Full PIMRE config.
    E_grid : 1D array
        Energy axis.
    bands_rep : 3D array
        Repeated bands for all rotations.
    KX_rot, KY_rot : 3D array
        Rotated momentum coordinates.
    kx_out, ky_out : 1D array
        Output momentum axes.

    Returns
    -------
    E_Mon : 3D array
        Interpolated intensity data (E, kx, ky).
    """
    pp = cfg["preprocessing"]
    method = pp.get("method", "kdtree")
    if method == "quadrant":
        qcfg = pp.get("quadrant", {})
        flip_kx = bool(qcfg.get("flip_kx", True))
        flip_ky = bool(qcfg.get("flip_ky", True))
        smooth_radius = qcfg.get("smooth_radius", 0.02)
        fill_radius = qcfg.get("fill_radius", 0.03)
        print(f"  Quadrant symmetrization on {bands_rep.shape[0]} layers"
              f" (flip_kx={flip_kx}, flip_ky={flip_ky},"
              f" smooth_radius={smooth_radius}, fill_radius={fill_radius},"
              f" stride={pp['stride']}) ...")
    elif method == "direct":
        print(f"  Direct use of momentum-space pixels on {bands_rep.shape[0]} layers"
              f" (no interpolation/symmetrization)")
    else:
        print(f"  KD-interpolation on {bands_rep.shape[0]} layers (stride={pp['stride']}) ...")

    if method == "direct":
        # No interpolation or symmetrization: the angular pixel grid with
        # its momentum axes is used as-is.
        E_Mon = bands_rep.astype(float)
        print(f"  E_Mon: {E_Mon.shape}")
        save_preprocessed_h5(pp["output_path"] or "test/exp_preprocessed.h5",
                             E_grid, kx_out, ky_out, E_Mon)
        print(f"  Saved → {pp['output_path'] or 'test/exp_preprocessed.h5'}")
        return E_Mon

    n_out = kx_out.shape[0]
    kxm, kym = np.meshgrid(kx_out, ky_out, indexing="ij")
    E_Mon = np.zeros((bands_rep.shape[0], n_out, n_out))
    stride = pp["stride"]
    for i in range(0, bands_rep.shape[0], stride):
        if i % 50 == 0:
            print(f"    layer {i}/{bands_rep.shape[0]}")
        if method == "quadrant":
            E_Mon[i] = quadrant_symmetrize(
                bands_rep[i], KX_rot[i], KY_rot[i],
                flip_kx=flip_kx, flip_ky=flip_ky,
                smooth_radius=smooth_radius, fill_radius=fill_radius,
                kx_grid=kxm, ky_grid=kym)
        else:
            E_Mon[i] = KDInterp(bands_rep[i], KX_rot[i], KY_rot[i],
                                radius=pp["kd_radius"], kx_grid=kxm, ky_grid=kym)
    for i in range(bands_rep.shape[0]):
        if i % stride != 0:
            lo = (i // stride) * stride
            hi = min(lo + stride, bands_rep.shape[0] - 1)
            frac = (i - lo) / (hi - lo) if hi > lo else 0
            E_Mon[i] = (1 - frac) * E_Mon[lo] + frac * E_Mon[hi]

    print(f"  E_Mon: {E_Mon.shape}")
    save_preprocessed_h5(pp["output_path"] or "test/exp_preprocessed.h5",
                         E_grid, kx_out, ky_out, E_Mon)
    print(f"  Saved → {pp['output_path'] or 'test/exp_preprocessed.h5'}")
    return E_Mon
