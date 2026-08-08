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


def ridge_alignment_score(E0, I_t, E_arr, sigma=0.1, window=0.15):
    """Band-ridge alignment score.

    For every k point the intensity profile along E within ±window of the
    band energy is taken; E_peak is the energy of the profile maximum. The
    score is the mean of exp(-((E_band - E_peak) / sigma)^2), i.e. 1 when the
    band sits exactly on the local intensity ridge and decaying as it drifts
    away (in contrast to the derivative-correlation term, this rewards being
    on the ridge rather than on its edge).

    Parameters
    ----------
    E0 : 2D array
        Band energy on (kx, ky) grid.
    I_t : 3D array
        Intensity data (E, kx, ky).
    E_arr : 1D array
        Energy axis (monotonically increasing).
    sigma : float
        Width of the ridge penalty in eV.
    window : float
        Half-width of the intensity profile window in eV.

    Returns
    -------
    score : float
        Mean ridge alignment in [0, 1].
    """
    nE = len(E_arr)
    if E_arr[0] > E_arr[-1]:
        E_arr = E_arr[::-1]
        I_t = I_t[::-1]
    dE = abs(E_arr[1] - E_arr[0])
    hw = max(1, int(round(window / dE)))

    skx, sky = E0.shape
    e_idx = np.interp(E0.ravel(), E_arr, np.arange(nE))
    ind = np.round(e_idx).astype(int).clip(0, nE - 1).reshape(skx, sky)

    rows = np.arange(skx)[:, None]
    cols = np.arange(sky)[None, :]
    prof = np.stack([
        I_t[np.clip(ind + k, 0, nE - 1), rows, cols]
        for k in range(-hw, hw + 1)
    ])  # (2hw+1, skx, sky)
    k_peak = np.argmax(prof, axis=0)
    I_peak = np.max(prof, axis=0)
    E_peak = E_arr[np.clip(ind + (k_peak - hw), 0, nE - 1)]

    dE2 = ((E0 - E_peak) / sigma) ** 2
    score = np.exp(-dE2)
    # Blank-region protection: a profile maximum that is not significantly
    # brighter than the global background cannot be a real ridge; such points
    # get zero score instead of absorbing the band into noise.
    thr = 0.05 * np.max(prof)
    score[I_peak < thr] = 0.0

    return float(np.mean(score))


def path_ridge_score(pathD, band_path, E_arr, sigma=0.1, window=0.15):
    """Band-ridge alignment along a band path.

    Same ridge logic as :func:`ridge_alignment_score` but evaluated on a
    band-path map ``pathD`` of shape (E, k): the intensity profile along E at
    each path point is used to find the local ridge energy. This is more
    focused than the full 2D evaluation because ARPES intensity typically
    only covers part of the BZ.

    Parameters
    ----------
    pathD : 2D array
        Band path intensity map (E, k).
    band_path : 1D array
        Band energy along the path (k,).
    E_arr : 1D array
        Energy axis (monotonically increasing).
    sigma : float
        Width of the ridge penalty in eV.
    window : float
        Half-width of the intensity profile window in eV.

    Returns
    -------
    score : float
        Mean ridge alignment in [0, 1].
    """
    nE = len(E_arr)
    if E_arr[0] > E_arr[-1]:
        E_arr = E_arr[::-1]
        pathD = pathD[::-1]
    dE = abs(E_arr[1] - E_arr[0])
    hw = max(1, int(round(window / dE)))
    nK = len(band_path)

    idx = np.round(np.interp(band_path, E_arr, np.arange(nE))).astype(int).clip(0, nE - 1)
    prof = np.stack([
        pathD[np.clip(idx + k, 0, nE - 1), np.arange(nK)]
        for k in range(-hw, hw + 1)
    ])  # (2hw+1, nK)
    k_peak = np.argmax(prof, axis=0)
    I_peak = np.max(prof, axis=0)
    E_peak = E_arr[np.clip(idx + (k_peak - hw), 0, nE - 1)]

    score = np.exp(-((band_path - E_peak) / sigma) ** 2)
    thr = 0.05 * np.max(prof)
    score[I_peak < thr] = 0.0
    return float(np.mean(score))


def compute_bsfi_2d(E0, I_t, E_arr, stride=4, w_corr=0.6, w_int=0.3, w_snr=0.1,
                    w_ridge=0.0, ridge_sigma=0.1):
    """Compute BSFI directly on the 2D band map.

    BSFI = Σ w_i · metric_i / Σ w_i with:
      corr:       |corr(dE/dk, dI/dk)| (derivative correlation)
      intensity:  global intensity ratio
      snr:        band intensity SNR
      ridge:      band-ridge alignment (see ridge_alignment_score)

    Setting a weight to 0 disables that component; the score is normalized
    by the sum of the active weights.

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
    w_corr, w_int, w_snr, w_ridge : float
        Component weights.
    ridge_sigma : float
        Width of the ridge penalty in eV.

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
    ridge = ridge_alignment_score(E0_s, I_t_s, E_arr, sigma=ridge_sigma) if w_ridge > 0 else 0.0

    total_w = w_corr + w_int + w_snr + w_ridge
    if total_w <= 0:
        return 0.0
    return (w_corr * score_corr + w_int * intensity_ratio
            + w_snr * snr + w_ridge * ridge) / total_w


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
