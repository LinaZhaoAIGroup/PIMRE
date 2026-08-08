"""High-symmetry point finding and lattice-to-reciprocal conversion.

Extracted from ArpesBandRecons.CoordTrans and 4.mrf.ipynb.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HspsResult:
    """Result of robust high-symmetry point finding."""

    hsps: dict
    confidence: float
    source: str = "theory"
    symmetry_score: float = 0.0
    best_layer: int = -1
    cv_errors: dict = field(default_factory=dict)
    rotation_angle: float = 0.0
    scale: float = 1.0
    registration_score: float = 0.0


def lattice_to_reciprocal(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """Convert real-space lattice parameters to reciprocal-space K and M points.

    Parameters
    ----------
    a, b, c : float
        Real-space lattice constants (angstroms).
    alpha_deg, beta_deg, gamma_deg : float
        Lattice angles in degrees.

    Returns
    -------
    k_K, k_M : ndarray
        Reciprocal-space coordinates of K and M high-symmetry points.
    """
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    gamma = np.deg2rad(gamma_deg)

    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])

    term1 = np.cos(beta)
    term2 = (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    term3 = np.sqrt(1 - term1**2 - term2**2)

    a3 = np.array([c * term1, c * term2, c * term3])
    volume = np.dot(a1, np.cross(a2, a3))

    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume

    k_K = (1 / 3) * b1 + (1 / 3) * b2
    k_M = 0.5 * b1
    return k_K, k_M


def dft_KM(kx_dft, ky_dft):
    """Find K and M point indices in DFT k-grid.

    The DFT momentum axes kx_dft/ky_dft are Cartesian (X, Y) coordinates;
    the K and M points are matched component-wise.

    Parameters
    ----------
    kx_dft, ky_dft : 1D array
        DFT momentum axes.

    Returns
    -------
    (KP_x, KP_y), (MP_x, MP_y) : tuple of ints
        Grid indices of K and M points.
    """
    reciprocal_to_cartesian = np.array([[1, 0.5], [0, np.sqrt(3) / 2]])
    MP = reciprocal_to_cartesian.dot(np.array([[0], [0.5]])).T
    KP = reciprocal_to_cartesian.dot(np.array([[1 / 3], [1 / 3]])).T

    KP_x = np.argmin(np.abs(kx_dft - KP[0, 0]))
    KP_y = np.argmin(np.abs(ky_dft - KP[0, 1]))
    MP_x = np.argmin(np.abs(kx_dft - MP[0, 0]))
    MP_y = np.argmin(np.abs(ky_dft - MP[0, 1]))
    return (KP_x, KP_y), (MP_x, MP_y)


def _rotate_around_center(point, center, angle_deg):
    """Rotate a point around a center by a given angle.

    Parameters
    ----------
    point : tuple (x, y)
        Point to rotate.
    center : tuple (cx, cy)
        Rotation center.
    angle_deg : float
        Rotation angle in degrees.

    Returns
    -------
    (rx, ry) : tuple of float
        Rotated point coordinates.
    """
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    dx, dy = point[0] - center[0], point[1] - center[1]
    rx = dx * c - dy * s + center[0]
    ry = dx * s + dy * c + center[1]
    return (rx, ry)


def _c6_generate(base_point, center, n=6):
    """Generate n points by C6 rotation of a base point around a center.

    Parameters
    ----------
    base_point : tuple (x, y)
        Base point in momentum space.
    center : tuple (cx, cy)
        Rotation center.
    n : int
        Number of rotations (default 6).

    Returns
    -------
    points : list of tuple
        n rotated points in order (0°, 60°, 120°, ...).
    """
    return [_rotate_around_center(base_point, center, i * 60) for i in range(n)]


def Get_G_M_K(crystal_data, kx, ky):
    """Find all Gamma, M and K high-symmetry point indices in the k-grid.

    K points (6): vertices of the hexagonal Brillouin zone.
    M points (6): midpoints of the hexagonal edges (perpendicular projection
    of Gamma onto each edge).

    Parameters
    ----------
    crystal_data : list
        [a, b, c, alpha, beta, gamma] lattice parameters.
    kx, ky : 1D array
        Momentum axes.

    Returns
    -------
    G : tuple (gx, gy)
        Gamma point grid index.
    K_points : list of 6 tuples
        K point indices (counterclockwise from first).
    M_points : list of 6 tuples
        M point indices (counterclockwise from first).
    """
    G = (np.argmin(np.abs(kx)), np.argmin(np.abs(ky)))
    G_kx, G_ky = kx[G[0]], ky[G[1]]

    k_K, k_M = lattice_to_reciprocal(*crystal_data)

    K_pts_abs = _c6_generate((k_K[0], k_K[1]), (0, 0))
    M_pts_abs = _c6_generate((k_M[0], k_M[1]), (0, 0))

    K_points = [
        (np.argmin(np.abs(kx - (G_kx + px))), np.argmin(np.abs(ky - (G_ky + py))))
        for px, py in K_pts_abs
    ]
    M_points = [
        (np.argmin(np.abs(kx - (G_kx + px))), np.argmin(np.abs(ky - (G_ky + py))))
        for px, py in M_pts_abs
    ]

    return G, K_points, M_points


def _build_hsps_dict(G, K_points, M_points):
    """Build standardized HSPs dict from G, K_points, M_points."""
    return {
        "G": G,
        **{f"K{i}": K_points[i] for i in range(6)},
        **{f"M{i}": M_points[i] for i in range(6)},
    }


# ── Robust HSP finding ──────────────────────────────────────────────────


def _hexagonal_symmetry_score(image):
    """Score how well an image exhibits 6-fold rotational symmetry via FFT.

    Parameters
    ----------
    image : 2D array
        Intensity image (kx, ky) slice.

    Returns
    -------
    score : float
        0–1 score where higher means stronger 6-fold symmetry.
    """
    fft = np.abs(np.fft.fftshift(np.fft.fft2(image - image.mean())))
    ny, nx = fft.shape
    cy, cx = ny // 2, nx // 2
    y, x = np.ogrid[:ny, :nx]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    r_max = min(cx, cy) * 0.8
    r_min = r_max * 0.15
    mask = (r >= r_min) & (r < r_max)
    if not np.any(mask):
        return 0.0

    theta = np.arctan2(y - cy, x - cx)
    theta_idx = np.round(np.rad2deg(theta) % 360).astype(int)

    profile = np.bincount(theta_idx[mask], weights=fft[mask], minlength=360)
    profile = profile / (profile.sum() + 1e-12)

    sixfold_power = 0.0
    for k in range(1, 7):
        sixfold_power += profile[k * 60 % 360]
    total_power = profile.sum()
    return sixfold_power / (total_power + 1e-12)


def _find_best_energy_layer(intensity, n_samples=20):
    """Find the energy layer with the strongest hexagonal symmetry.

    Parameters
    ----------
    intensity : 3D array
        (E, kx, ky) intensity data.
    n_samples : int
        Number of layers to sample (evenly spaced).

    Returns
    -------
    best_layer : int
        Index of the best layer.
    best_score : float
        Symmetry score of the best layer.
    """
    nE = intensity.shape[0]
    indices = np.linspace(0, nE - 1, min(n_samples, nE), dtype=int)
    best_layer, best_score = 0, -1.0
    for i in indices:
        score = _hexagonal_symmetry_score(intensity[i])
        if score > best_score:
            best_score, best_layer = score, i
    return int(best_layer), best_score



def find_hsps_robust(intensity, kx, ky, crystal_data, E_grid=None, calibration=None):
    """Robust high-symmetry point finding.

    Supports two modes:
    - Auto: perpendicular mirror symmetry registration (default).
    - Manual: uses user-specified rotation_angle from calibration config.

    Parameters
    ----------
    intensity : 3D array
        (E, kx, ky) experimental intensity data.
    kx, ky : 1D array
        Momentum axes.
    crystal_data : list
        [a, b, c, alpha, beta, gamma] lattice parameters.
    E_grid : 1D array or None
        Energy axis (optional, for reporting).
    calibration : dict or None
        If dict with 'manual': True, uses 'rotation_angle' and 'scale'
        from the dict instead of auto-registration.

    Returns
    -------
    HspsResult
        Dataclass with hsps dict, confidence, registration info.
    """
    from ..kpath.registration import build_hsps_from_registration, register_bz

    G, _, _ = Get_G_M_K(crystal_data, kx, ky)

    best_layer, sym_score = _find_best_energy_layer(intensity)

    if calibration and calibration.get("manual", False):
        theta = calibration.get("rotation_angle", 0.0)
        reg_score = 1.0
        source = "manual"
    else:
        theta, reg_score = register_bz(
            intensity[best_layer], kx, ky, crystal_data, G,
            n_angles=60,
        )
        source = "theory" if reg_score < 0.1 else "registered"

    scale = float(calibration.get("scale", 1.0)) if calibration else 1.0
    K_points, M_points = build_hsps_from_registration(
        crystal_data, kx, ky, G, theta, scale=scale,
    )

    hsps = _build_hsps_dict(G, K_points, M_points)
    confidence = float(np.clip(reg_score, 0.0, 1.0))

    return HspsResult(
        hsps=hsps,
        confidence=confidence,
        source=source,
        symmetry_score=sym_score,
        best_layer=best_layer,
        cv_errors={},
        rotation_angle=theta,
        scale=scale,
        registration_score=reg_score,
    )
