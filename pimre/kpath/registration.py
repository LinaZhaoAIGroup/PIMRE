"""BZ orientation registration via perpendicular reflection symmetry.

Strategy:
  1. From lattice parameters, compute two perpendicular directions:
     the K-K line direction and M-M line direction.
  2. Sweep over rotation angle θ ∈ [0, 60°).
  3. For each θ, reflect the ARPES data across the two perpendicular
     lines at θ and θ+90° through Γ.
  4. The best θ is where both reflections yield maximum correlation
     with the original data.
  5. Build K points at |Γ-K| along the 6 K-K directions, M points at
     |Γ-M| along the 6 M-M directions (offset by 30° from K-K).
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pimre.kpath.symmetry import (
    lattice_to_reciprocal,
)


def _reflect_across_line(data, kx, ky, G, angle_deg):
    """Reflect 2D data across a line through G at given angle.

    Parameters
    ----------
    data : 2D array
        ARPES intensity slice.
    kx, ky : 1D array
        Momentum axes.
    G : tuple (gx, gy)
        Gamma point grid index.
    angle_deg : float
        Angle of the reflection line in degrees.

    Returns
    -------
    reflected : 2D array
        Data reflected across the line, interpolated.
    """
    gx, gy = kx[G[0]], ky[G[1]]
    c2, s2 = np.cos(2 * np.deg2rad(angle_deg)), np.sin(2 * np.deg2rad(angle_deg))

    kxx, kyy = np.meshgrid(kx, ky, indexing="ij")
    dx = kxx - gx
    dy = kyy - gy

    # Reflection across line at angle θ: (x', y') = R_θ · diag(1,-1) · R_{-θ} · (x,y)
    # = (x·c2 + y·s2,  x·s2 - y·c2)
    rx = gx + dx * c2 + dy * s2
    ry = gy + dx * s2 - dy * c2

    interp = RegularGridInterpolator(
        (kx, ky), data, bounds_error=False, fill_value=0.0
    )
    pts = np.column_stack((rx.ravel(), ry.ravel()))
    reflected = interp(pts).reshape(kxx.shape)
    return reflected


def _mirror_symmetry_score(data, kx, ky, G, angle_deg):
    """Score mirror symmetry across a line at given angle.

    Returns the Pearson correlation between original and reflected data.
    """
    reflected = _reflect_across_line(data, kx, ky, G, angle_deg)
    c = np.corrcoef(data.ravel(), reflected.ravel())[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def register_bz(intensity_slice, kx, ky, crystal_data, G,
                n_angles=60, downsample=200):
    """Register BZ orientation via perpendicular mirror symmetry.

    The two perpendicular lines through Γ are the K-K and M-M lines.
    The best rotation angle θ is where both lines have maximum
    reflection symmetry.

    Parameters
    ----------
    intensity_slice : 2D array
        ARPES intensity at a single energy layer.
    kx, ky : 1D array
        Momentum axes.
    crystal_data : list
        [a, b, c, alpha, beta, gamma].
    G : tuple (gx, gy)
        Gamma point grid index.
    n_angles : int
        Number of rotation angle samples.
    downsample : int
        Target grid size for downsampled data.

    Returns
    -------
    best_theta : float
        Optimal K-K line angle in degrees.
    best_score : float
        Combined symmetry score at optimum.
    """
    # ── Downsample data ──
    kx_ds = np.linspace(kx[0], kx[-1], downsample)
    ky_ds = np.linspace(ky[0], ky[-1], downsample)
    gx, gy = kx[G[0]], ky[G[1]]

    interp_data = RegularGridInterpolator(
        (kx, ky), intensity_slice, bounds_error=False, fill_value=0.0
    )
    kxx_ds, kyy_ds = np.meshgrid(kx_ds, ky_ds, indexing="ij")
    data_ds = interp_data(np.column_stack((kxx_ds.ravel(), kyy_ds.ravel())))
    data_ds = data_ds.reshape(kxx_ds.shape)

    G_ds = (np.argmin(np.abs(kx_ds - gx)), np.argmin(np.abs(ky_ds - gy)))

    # ── Sweep ──
    angles = np.linspace(0, 60, n_angles, endpoint=False)
    best_theta, best_score = 0.0, -1.0

    for theta in angles:
        score1 = _mirror_symmetry_score(data_ds, kx_ds, ky_ds, G_ds, theta)
        score2 = _mirror_symmetry_score(data_ds, kx_ds, ky_ds, G_ds, theta + 90)
        combined = score1 + score2
        if combined > best_score:
            best_score, best_theta = combined, theta

    return float(best_theta), float(best_score)


def build_hsps_from_registration(crystal_data, kx, ky, G, theta, scale=1.0):
    """Build K and M points from registration angle and lattice parameters.

    K points are at |Γ-K| distance along the 6 K-K directions.
    M points are at |Γ-M| distance along the 6 M-M directions
    (offset by 30° from K-K).

    Parameters
    ----------
    crystal_data : list
        [a, b, c, alpha, beta, gamma].
    kx, ky : 1D array
        Momentum axes.
    G : tuple (gx, gy)
        Gamma point grid index.
    theta : float
        K-K line angle in degrees.
    scale : float
        Multiplicative scale applied to the |Γ-K| and |Γ-M| distances
        (calibrated against the experimental BZ size).

    Returns
    -------
    K_points : list of 6 tuples
        K point grid indices.
    M_points : list of 6 tuples
        M point grid indices.
    """
    k_K, k_M = lattice_to_reciprocal(*crystal_data)
    G_kx, G_ky = kx[G[0]], ky[G[1]]

    K_dist = float(np.linalg.norm(k_K[:2])) * scale
    M_dist = float(np.linalg.norm(k_M[:2])) * scale

    theta_rad = np.deg2rad(theta)
    K_points = []
    M_points = []
    for i in range(6):
        # K at every 60° starting from θ
        ak = theta_rad + i * np.pi / 3
        kx_k = G_kx + K_dist * np.cos(ak)
        ky_k = G_ky + K_dist * np.sin(ak)
        K_points.append((
            np.argmin(np.abs(kx - kx_k)),
            np.argmin(np.abs(ky - ky_k)),
        ))
        # M at every 60° starting from θ+30°
        am = theta_rad + np.pi / 6 + i * np.pi / 3
        kx_m = G_kx + M_dist * np.cos(am)
        ky_m = G_ky + M_dist * np.sin(am)
        M_points.append((
            np.argmin(np.abs(kx - kx_m)),
            np.argmin(np.abs(ky - ky_m)),
        ))

    return K_points, M_points
