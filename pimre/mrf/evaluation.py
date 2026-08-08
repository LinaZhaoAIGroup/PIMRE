"""MRF reconstruction evaluation: BSFI scoring and affine DFT mapping.

Extracted from 4.mrf.ipynb and mrf_bsfi_pipeline.py.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator


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
