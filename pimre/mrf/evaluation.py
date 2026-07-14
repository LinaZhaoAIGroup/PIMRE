"""MRF reconstruction evaluation: BSFI scoring, band path plotting, affine transform.

Extracted from 4.mrf.ipynb and mrf_bsfi_pipeline.py.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

from ..kpath.path import points2path, bandpath_map as _bandpath_map
from ..utils.io import loadHDF


def theory_data_expand(ind, kx_dft, ky_dft, E_dft, padded_kx, padded_ky, new_scale=900):
    """Expand DFT band data to match experimental grid size.

    Parameters
    ----------
    ind : int
        Band index.
    kx_dft, ky_dft : 1D array
        DFT momentum axes.
    E_dft : 3D array
        DFT band data (nbands, kx, ky).
    padded_kx, padded_ky : 1D array
        Target momentum axes.
    new_scale : int
        Target grid size.

    Returns
    -------
    Ek_scaled : 2D array
        Interpolated band data.
    """
    intFunc_th = RegularGridInterpolator(
        (kx_dft, ky_dft), E_dft[ind, :, :], bounds_error=False, fill_value=0
    )
    kxx, kyy = np.meshgrid(padded_kx, padded_ky, indexing="ij")
    Ek_scaled = intFunc_th(np.column_stack((kxx.ravel(), kyy.ravel())))
    return Ek_scaled.reshape((new_scale, new_scale))


def expand_dft_bands(E_dft, kx_dft, ky_dft, target_kx, target_ky):
    """Expand all DFT bands to match experimental grid.

    Parameters
    ----------
    E_dft : 3D array
        DFT band data.
    kx_dft, ky_dft : 1D array
        DFT momentum axes.
    target_kx, target_ky : 1D array
        Target momentum axes.

    Returns
    -------
    E_dft_expanded : 3D array
        Expanded DFT band data.
    kx_dft_new, ky_dft_new : 1D array
        New momentum axes.
    """
    from scipy.interpolate import interp1d

    interp_kx = interp1d(np.linspace(0, 1, kx_dft.shape[0]), kx_dft, kind="linear")
    interp_ky = interp1d(np.linspace(0, 1, ky_dft.shape[0]), ky_dft, kind="linear")
    padded_kx = interp_kx(np.linspace(0, 1, target_kx.shape[0]))
    padded_ky = interp_ky(np.linspace(0, 1, target_ky.shape[0]))

    E_dft_expanded = np.zeros((E_dft.shape[0], target_kx.shape[0], target_ky.shape[0]))
    for ind in range(E_dft.shape[0]):
        E_dft_expanded[ind] = theory_data_expand(
            ind, kx_dft, ky_dft, E_dft, padded_kx, padded_ky, target_kx.shape[0]
        )
    return E_dft_expanded, padded_kx, padded_ky


def align_dft_to_exp(kx, ky, kx_dft, ky_dft, M, MP_dft, G):
    """Align DFT coordinates to experimental coordinates.

    Parameters
    ----------
    kx, ky : 1D array
        Experimental momentum axes.
    kx_dft, ky_dft : 1D array
        DFT momentum axes.
    M : tuple
        (M_x, M_y) indices in experimental grid.
    MP_dft : tuple
        (MP_x, MP_y) indices in DFT grid.
    G : tuple
        (G_x, G_y) Gamma point indices in experimental grid.

    Returns
    -------
    kx_dft_aligned, ky_dft_aligned : 1D array
        Aligned DFT momentum axes.
    """
    from decimal import Decimal

    kx_dft_0 = kx[M[0]] - ((kx[M[0]] - kx_dft[np.argmin(np.abs(kx_dft))]) / (MP_dft[0] - np.argmin(np.abs(kx_dft)))) * (MP_dft[0] - 1)
    kx_dft_step = (kx[M[0]] - kx_dft[np.argmin(np.abs(kx_dft))]) / (MP_dft[0] - np.argmin(np.abs(kx_dft)))
    ky_dft_0 = ky[M[1]] - ((ky[M[1]] - ky_dft[np.argmin(np.abs(ky_dft))]) / (MP_dft[1] - np.argmin(np.abs(ky_dft)))) * (MP_dft[1] - 1)
    ky_dft_step = (ky[M[1]] - ky_dft[np.argmin(np.abs(ky_dft))]) / (MP_dft[1] - np.argmin(np.abs(ky_dft)))

    kx_dft_aligned = np.array(
        np.arange(start=Decimal(kx_dft_0), stop=Decimal(kx_dft_0 + kx_dft.shape[0] * kx_dft_step), step=Decimal(kx_dft_step)),
        dtype="float64",
    )
    ky_dft_aligned = np.array(
        np.arange(start=Decimal(ky_dft_0), stop=Decimal(ky_dft_0 + ky_dft.shape[0] * ky_dft_step), step=Decimal(ky_dft_step)),
        dtype="float64",
    )
    return kx_dft_aligned, ky_dft_aligned


def bandpath_map(bcsm, path_points, seg_points, eaxis=2):
    """Extract band path map from 3D data.

    Parameters
    ----------
    bcsm : 3D array
        Band structure data.
    path_points : ndarray
        (n_points, 2) array of (row, col) indices.
    seg_points : list
        Number of points per segment.
    eaxis : int
        Energy axis index.

    Returns
    -------
    bdi : 2D array
        Band path map.
    """
    row_inds, col_inds, path_inds = points2path(path_points[:, 0], path_points[:, 1], npoints=seg_points)
    bdi = _bandpath_map(np.moveaxis(bcsm, eaxis, 2), pathr=row_inds, pathc=col_inds, eaxis=2)
    return bdi.T


def compute_bsfi_score(pathD, preci, E):
    """Compute BSFI (Band Structure Feature Intensity) score.

    Parameters
    ----------
    pathD : 2D array
        Band path map (E, k).
    preci : 1D array
        Reconstructed band energies along the path.
    E : 1D array
        Energy axis.

    Returns
    -------
    total_score : float
        BSFI score = 0.3 * intensity_ratio + 0.6 * correlation + 0.1 * snr.
    """
    size = pathD.shape[1] if pathD.ndim > 1 else len(pathD)

    def energy_to_index(E_target, E_array):
        return np.argmin(np.abs(E_array - E_target))

    pI = np.array([pathD[energy_to_index(preci[j], E), j] for j in range(size)])

    window_length = min(21, len(pI) - 1 if len(pI) % 2 == 0 else len(pI))
    if window_length >= 3 and window_length % 2 == 0:
        window_length -= 1
    if window_length >= 3:
        pI = savgol_filter(pI, window_length, 3)

    dE_dk = np.gradient(preci)
    dI_dk = np.gradient(pI)

    intensity_ratio_peak = (pI.max() - pathD.min()) / (pathD.max() - pathD.min()) if pathD.max() > pathD.min() else 0
    snr = np.mean(pI) / np.std(pI) if np.std(pI) > 0 else 0
    correlation_sign = np.abs(np.corrcoef(dE_dk, dI_dk)[0, 1]) if len(dE_dk) > 1 else 0

    total_score = 0.3 * intensity_ratio_peak + 0.6 * correlation_sign + 0.1 * snr
    return total_score


def compute_bsfi_score_old(pathD, preci, E, band_index):
    """Compute BSFI (Band Structure Feature Intensity) score.

    Parameters
    ----------
    pathD : 2D array
        Band path map (E, k).
    preci : 1D array
        Reconstructed band energies along the path.
    E : 1D array
        Energy axis.
    band_index : int
        Band index.

    Returns
    -------
    total_score : float
        BSFI score.
    """
    size = pathD.shape[1] if pathD.ndim > 1 else len(pathD)
    E_norm = (preci - preci.mean()) / preci.std()

    def energy_to_index(E_target, E_array):
        return np.argmin(np.abs(E_array - E_target))

    pI = np.array([pathD[energy_to_index(preci[j], E), j] for j in range(size)])

    window_length = min(21, len(pI) - 1 if len(pI) % 2 == 0 else len(pI))
    if window_length >= 3 and window_length % 2 == 0:
        window_length -= 1
    if window_length >= 3:
        pI = savgol_filter(pI, window_length, 3)

    I_norm = (pI - pI.mean()) / pI.std() if pI.std() > 0 else pI

    dE_dk = np.gradient(preci)
    dI_dk = np.gradient(pI)

    intensity_ratio_peak = (pI.max() - pathD.min()) / (pathD.max() - pathD.min()) if pathD.max() > pathD.min() else 0
    snr = np.mean(pI) / np.std(pI) if np.std(pI) > 0 else 0
    correlation_sign = np.abs(np.corrcoef(dE_dk, dI_dk)[0, 1]) if len(dE_dk) > 1 else 0

    total_score = 0.3 * intensity_ratio_peak + 0.6 * correlation_sign + 0.1 * snr
    return total_score


def draw_band_path(pathD, recon_bands, row_inds, path_inds, E, choose="M", colors=None, save_path=None):
    """Plot band path diagram with reconstructed bands overlaid.

    Parameters
    ----------
    pathD : 2D array
        Band path map.
    recon_bands : 2D array
        Reconstructed bands along the path.
    row_inds : 1D array
        Row indices along the path.
    path_inds : 1D array
        High-symmetry point indices.
    E : 1D array
        Energy axis.
    choose : str
        Path type: 'M', 'K', or 'GMK'.
    colors : list or None
        Colors for each band.
    save_path : str or None
        Path to save figure.
    """
    if colors is None:
        colors = ["r", "y", "b", "g", "w"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(
        pathD,
        cmap="plasma",
        extent=[0, len(row_inds), E[0], E[min(109, len(E) - 1)]],
        aspect="auto",
        origin="upper",
    )

    for ib in range(recon_bands.shape[1]):
        ax.plot(
            savgol_filter(recon_bands[:, ib], min(30, len(recon_bands) - 2), 2),
            zorder=1,
            linewidth=2.3,
            color=colors[ib % len(colors)],
        )

    ax.set_xlim(0, len(row_inds))
    ax.set_ylim(E[0], E[min(109, len(E) - 1)])
    ax.set_xticks(path_inds)

    labels = {
        "M": [r"$\overline{\mathrm{M}}$", r"$\overline{\Gamma}$", r"$\overline{\mathrm{M}}$"],
        "K": [r"$\overline{\mathrm{K}}$", r"$\overline{\Gamma}$", r"$\overline{\mathrm{K}}$"],
        "GMK": [r"$\overline{\Gamma}$", r"$\overline{\mathrm{M}}$", r"$\overline{\mathrm{K}}$", r"$\overline{\Gamma}$"],
    }
    ax.set_xticklabels(labels.get(choose, labels["GMK"]), fontsize=15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.colorbar()
    return fig, ax


def run_mrf_reconstruction_single_band(mrf, ind_band, hyperparam, kx, ky, E_dft, kx_dft, ky_dft, kx_dft_expanded, ky_dft_expanded, G, M, M1, K, K1, recon, colors, pre, I_t):
    """Run MRF reconstruction for a single band.

    Parameters
    ----------
    mrf : MrfRec
        MRF reconstruction object.
    ind_band : int
        Band index.
    hyperparam : ndarray
        Hyperparameters for this band: [index, kScale, offset, eta].
    kx, ky : 1D array
        Experimental momentum axes.
    E_dft : 3D array
        DFT band data.
    kx_dft, ky_dft : 1D array
        DFT momentum axes.
    kx_dft_expanded, ky_dft_expanded : 1D array
        Expanded DFT momentum axes.
    G, M, M1, K, K1 : tuple
        High-symmetry point indices.
    recon : 3D array
        Reconstructed band array to fill.
    colors : list
        Colors for plotting.
    pre : dict
        Preprocessed experimental data.
    I_t : 3D array
        Intensity data for MRF.

    Returns
    -------
    recon : 3D array
        Updated reconstructed bands.
    pathD : 2D array
        Band path diagram.
    prec : 2D array
        Reconstructed bands along the path.
    """
    mrf.eta = hyperparam[3]

    Einterp = theory_data_expand(
        ind_band * 2, kx_dft_expanded, ky_dft_expanded, E_dft, kx, ky, kx.shape[0]
    )
    E0 = np.reshape(Einterp + hyperparam[2], (kx.shape[0], ky.shape[0]))

    EE, EE0 = np.meshgrid(mrf.E, E0)
    ind1d = np.argmin(np.abs(EE - EE0), 1)
    mrf.indEb = ind1d.reshape(E0.shape)

    recon[ind_band, ...] = mrf.getEb()

    from ..kpath.path import bandpath_map as bpm

    nGM = int(math.sqrt((M[0] - G[0]) ** 2 + (M[1] - G[1]) ** 2))
    nMK = int(math.sqrt((M[0] - K[0]) ** 2 + (M[1] - K[1]) ** 2))
    nKG = int(math.sqrt((K[0] - G[0]) ** 2 + (K[1] - G[1]) ** 2))

    path_points = np.asarray([G, M, K, G])
    seg_points = [nGM, nMK, nKG]
    row_inds, col_inds, path_inds = points2path(path_points[:, 0], path_points[:, 1], npoints=seg_points)

    pathD = bpm(np.transpose(I_t, (1, 2, 0)), pathr=row_inds, pathc=col_inds, eaxis=2)
    prec = bpm(recon, pathr=row_inds, pathc=col_inds, eaxis=0)

    draw_band_path(pathD, prec, row_inds, path_inds, pre["E"], choose="GMK", colors=colors)

    from .symmetry import sym_band

    sym_band(ind_band, recon, kx, ky, mrf.lengthKx, mrf.lengthKy)

    return recon, pathD, prec


def sweep_bsfi_offsets(mrf, ind_band, kx, ky, E_dft_a, kx_dft_a, ky_dft_a, E, I, G, M, K,
                       offset_range=(-0.5, 0.5, 0.1)):
    """Sweep energy offsets and find the one with maximum BSFI score.

    Parameters
    ----------
    mrf : MrfRec
        MRF reconstruction object.
    ind_band : int
        Band index.
    kx, ky : 1D array
        Experimental momentum axes.
    E_dft_a, kx_dft_a, ky_dft_a : ndarray
        Aligned DFT band data.
    E : 1D array
        Energy axis.
    I : 3D array
        Intensity data (kx, ky, E).
    G, M, K : tuple
        High-symmetry point indices.
    offset_range : tuple
        (start, stop, step) for energy offset sweep.

    Returns
    -------
    best_offset : float
        Energy offset with maximum BSFI score.
    scores : list
        BSFI scores for each offset.
    offsets : ndarray
        Tested offset values.
    """
    from ..kpath.path import bandpath_map as bpm

    offsets = np.arange(offset_range[0], offset_range[1] + offset_range[2] / 2, offset_range[2])
    scores = []

    nGM = int(math.sqrt((M[0] - G[0]) ** 2 + (M[1] - G[1]) ** 2))
    nMK = int(math.sqrt((M[0] - K[0]) ** 2 + (M[1] - K[1]) ** 2))
    nKG = int(math.sqrt((K[0] - G[0]) ** 2 + (K[1] - G[1]) ** 2))
    path_points = np.asarray([G, M, K, G])
    row_inds, col_inds, _ = points2path(path_points[:, 0], path_points[:, 1], npoints=[nGM, nMK, nKG])
    pathD = bpm(np.transpose(I, (2, 0, 1)), pathr=row_inds, pathc=col_inds, eaxis=0)

    for offset in offsets:
        Einterp = theory_data_expand(ind_band * 2, kx_dft_a, ky_dft_a, E_dft_a, kx, ky, kx.shape[0])
        E0 = np.reshape(Einterp + offset, (kx.shape[0], ky.shape[0]))
        EE, EE0 = np.meshgrid(E, E0)
        mrf.indEb = np.argmin(np.abs(EE - EE0), 1).reshape(E0.shape)
        recon_band = mrf.getEb()

        recon_cube = recon_band[np.newaxis, ...]
        prec = bpm(recon_cube, pathr=row_inds, pathc=col_inds, eaxis=0)
        score = compute_bsfi_score(pathD, prec[0], E)
        scores.append(score)

    best_idx = np.argmax(scores)
    return offsets[best_idx], scores, offsets


# ── 2D BSFI (no path extraction) ──

def compute_bsfi_2d(E0, I_t, E_arr, stride=4, w_corr=0.6, w_int=0.3, w_snr=0.1):
    """Compute BSFI directly on the 2D band map.

    BSFI = w_corr * |corr(dE/dk, dI/dk)| + w_int * intensity_ratio + w_snr * SNR

    Parameters
    ----------
    E0 : 2D array
        Band energy on (kx, ky) grid.
    I_t : 3D array
        Intensity data (E, kx, ky).
    E_arr : 1D array
        Energy axis (must be monotonically increasing for np.interp).
    stride : int
        Downsampling stride for speed.
    w_corr, w_int, w_snr : float
        BSFI weights.

    Returns
    -------
    score : float
    """
    nE = len(E_arr)

    # Ensure E_arr is increasing (np.interp requirement)
    if E_arr[0] > E_arr[-1]:
        E_arr = E_arr[::-1]
        I_t = I_t[::-1]

    E0_s = E0[::stride, ::stride]
    I_t_s = I_t[:, ::stride, ::stride]
    skx, sky = E0_s.shape

    e_idx = np.interp(E0_s.ravel(), E_arr, np.arange(nE))
    indEb = np.round(e_idx).astype(int).clip(0, nE - 1).reshape(skx, sky)

    I_band = I_t_s[indEb, np.arange(skx)[:, None], np.arange(sky)[None, :]]

    denom = I_t_s.max() - I_t_s.min()
    intensity_ratio = (I_band.mean() - I_t_s.min()) / denom if denom > 0 else 0

    snr = I_band.mean() / I_band.std() if I_band.std() > 0 else 0

    dE_dkx = np.gradient(E0_s, axis=0)
    dI_dkx = np.gradient(I_band, axis=0)
    corr_x = np.corrcoef(dE_dkx.ravel(), dI_dkx.ravel())[0, 1]
    if np.isnan(corr_x):
        corr_x = 0.0

    dE_dky = np.gradient(E0_s, axis=1)
    dI_dky = np.gradient(I_band, axis=1)
    corr_y = np.corrcoef(dE_dky.ravel(), dI_dky.ravel())[0, 1]
    if np.isnan(corr_y):
        corr_y = 0.0

    score_corr = (abs(corr_x) + abs(corr_y)) / 2

    return w_corr * score_corr + w_int * intensity_ratio + w_snr * snr


# ── Affine transform ──

def compute_affine_transform(kx, ky, G, K, M, kx_dft, ky_dft, KP_dft_raw, MP_dft_raw):
    """Compute 2×2 affine transform T that maps DFT → experiment.

    T is computed by matching the Γ→K and Γ→M vectors in both spaces.

    Parameters
    ----------
    kx, ky : 1D array
        Experimental momentum axes.
    G, K, M : tuple
        High-symmetry point indices in experimental grid.
    kx_dft, ky_dft : 1D array
        DFT momentum axes (original, unscaled).
    KP_dft_raw, MP_dft_raw : tuple
        K and M point indices in DFT grid.

    Returns
    -------
    T : 2×2 array
        Affine transform matrix.
    T_inv : 2×2 array
        Inverse transform.
    scale_x, scale_y : float
        Scale factors.
    rotation_deg : float
        Rotation angle in degrees.
    """
    g_dft_x = np.argmin(np.abs(kx_dft))
    g_dft_y = np.argmin(np.abs(ky_dft))

    K_vec_exp = np.array([kx[K[0]] - kx[G[0]], ky[K[1]] - ky[G[1]]])
    M_vec_exp = np.array([kx[M[0]] - kx[G[0]], ky[M[1]] - ky[G[1]]])

    K_vec_dft = np.array([kx_dft[KP_dft_raw[0]] - kx_dft[g_dft_x],
                           ky_dft[KP_dft_raw[1]] - ky_dft[g_dft_y]])
    M_vec_dft = np.array([kx_dft[MP_dft_raw[0]] - kx_dft[g_dft_x],
                           ky_dft[MP_dft_raw[1]] - ky_dft[g_dft_y]])

    S_dft = np.column_stack((K_vec_dft, M_vec_dft))
    S_exp = np.column_stack((K_vec_exp, M_vec_exp))
    T = S_exp @ np.linalg.inv(S_dft)
    T_inv = np.linalg.inv(T)

    scale_x = np.linalg.norm(T[:, 0])
    scale_y = np.linalg.norm(T[:, 1])
    rotation_deg = np.rad2deg(np.arctan2(T[1, 0], T[0, 0]))

    return T, T_inv, scale_x, scale_y, rotation_deg


def map_dft_bands(E_dft_orig, kx_dft_orig, ky_dft_orig, kx, ky, T_inv, band_indices):
    """Map DFT bands to experimental grid via affine transform T_inv.

    Parameters
    ----------
    E_dft_orig : 3D array
        Original DFT band data (nbands, kx_dft, ky_dft).
    kx_dft_orig, ky_dft_orig : 1D array
        Original DFT momentum axes.
    kx, ky : 1D array
        Experimental momentum axes.
    T_inv : 2×2 array
        Inverse affine transform (exp → DFT).
    band_indices : list of int
        DFT band indices to map.

    Returns
    -------
    dft_bands : list of 2D arrays
        DFT bands mapped to experimental grid.
    """
    kxx, kyy = np.meshgrid(kx, ky, indexing="ij")
    pts_exp = np.column_stack((kxx.ravel(), kyy.ravel()))
    pts_dft = (T_inv @ pts_exp.T).T

    dft_bands = []
    for ind in band_indices:
        spline = RectBivariateSpline(kx_dft_orig, ky_dft_orig, E_dft_orig[ind, :, :], kx=1, ky=1, s=0)
        band_mapped = spline.ev(pts_dft[:, 0], pts_dft[:, 1]).reshape(kxx.shape)
        dft_bands.append(band_mapped)
    return dft_bands