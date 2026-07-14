"""Experimental data preprocessing: loading, calibration, angle-to-momentum conversion.

Replicates the pipeline from 2.exp_data_pre.ipynb.
"""

import re
import numpy as np
import h5py
from scipy.constants import hbar, m_e, electron_volt
from scipy.interpolate import griddata
from scipy.spatial import KDTree


# --- Parameter extraction ---


def extract_wave_params(WAVE_INFO, EXP_INFO=None):
    """Extract wave parameters from WAVE_INFO text.

    Parameters
    ----------
    WAVE_INFO : str
        Wave information text with Rows, Columns, Layers metadata.
    EXP_INFO : dict or None
        Experiment information (optional).

    Returns
    -------
    Range : dict
        Dictionary with E_range, kx_range, ky_range.
    """
    def extract_param(pattern, text):
        match = re.search(pattern, text)
        if match:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        return None

    rows = extract_param(r"Rows:\s*(\d+)\s*Start:\s*([-\d.]+)\s*Delta:\s*([-\d.]+)", WAVE_INFO)
    columns = extract_param(r"Columns:\s*(\d+)\s*Start:\s*([-\d.]+)\s*Delta:\s*([-\d.]+)", WAVE_INFO)
    layers = extract_param(r"Layers:\s*(\d+)\s*Start:\s*([-\d.]+)\s*Delta:\s*([-\d.]+)", WAVE_INFO)

    Range = {
        "E_range": (
            rows[1],
            rows[1] + rows[2] * rows[0],
        ),
        "kx_range": (
            columns[1],
            columns[1] + columns[2] * columns[0],
        ),
        "ky_range": (
            layers[1],
            layers[1] + layers[2] * layers[0],
        ),
    }
    return Range


# --- Data loading ---


def load_exp_data(filepath, wave_name):
    """Load experimental HDF5 data.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    wave_name : str
        Name of the wave group in the HDF5 file.

    Returns
    -------
    bands : ndarray
        3D array of intensity data (E, kx, ky).
    """
    with h5py.File(filepath, "r") as f:
        bands = f[wave_name][:]
    return bands


def load_exp_data_from_h5(filepath, axes_key="axes", intensity_key="intensity"):
    """Load preprocessed experimental data from HDF5.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file.
    axes_key : str
        Key for axes group.
    intensity_key : str
        Key for intensity group.

    Returns
    -------
    E_grid : ndarray
        Energy grid.
    kx : ndarray
        kx axis.
    ky : ndarray
        ky axis.
    E_Mon : ndarray
        3D intensity data.
    """
    with h5py.File(filepath, "r") as f:
        E_grid = f[axes_key]["E"][:]
        kx = f[axes_key]["kx"][:]
        ky = f[axes_key]["ky"][:]
        E_Mon = f[intensity_key]["V"][:]
    return E_grid, kx, ky, E_Mon


# --- Angle to momentum conversion ---


def Angle2Mon(E_grid, X_Angle, Y_Angle, X_Shift=0, Y_Shift=0, work_function=16.03):
    """Convert angles to momentum space coordinates (KX, KY).

    Parameters
    ----------
    E_grid : 1D array
        Energy grid (kinetic energy in eV).
    X_Angle : 1D array
        Angles in X direction (degrees), typically alpha + beta.
    Y_Angle : 1D array
        Angles in Y direction (degrees), typically theta.
    X_Shift, Y_Shift : float
        Angle shifts in X and Y directions.
    work_function : float
        Work function in eV.

    Returns
    -------
    KX, KY : ndarray
        3D arrays of momentum coordinates (E, kx_angle, ky_angle).
    """
    X_Angle = X_Angle - X_Shift
    Y_Angle = Y_Angle - Y_Shift

    E_kinetic = (work_function - E_grid) * electron_volt
    K_norm = np.sqrt(2 * m_e * E_kinetic) / (1e10 * hbar)
    K_norm = K_norm[:, np.newaxis, np.newaxis]

    X_Angle = np.deg2rad(X_Angle)
    Y_Angle = np.deg2rad(Y_Angle)

    sin_X = np.sin(X_Angle)[np.newaxis, :, np.newaxis]
    sin_Y = np.sin(Y_Angle)[np.newaxis, np.newaxis, :]
    cos_Y = np.cos(Y_Angle)[np.newaxis, np.newaxis, :]

    KX = K_norm * sin_X * cos_Y
    KY = K_norm * sin_Y
    KY = np.repeat(KY, KX.shape[1], axis=1)

    return KX, KY


def Angle2MonGrid(E_grid, alpha, beta, theta, work_function=16.03):
    """Convert angles to momentum space using alpha, beta, theta grid.

    Parameters
    ----------
    E_grid : 1D array
        Energy grid.
    alpha, beta, theta : 1D array
        Angle arrays in degrees.
    work_function : float
        Work function in eV.

    Returns
    -------
    KX, KY : ndarray
        3D momentum arrays.
    """
    E_kinetic = (work_function - E_grid) * electron_volt
    K_norm = np.sqrt(2 * m_e * E_kinetic) / (1e10 * hbar)
    K_norm = K_norm[:, np.newaxis, np.newaxis]

    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    theta = np.deg2rad(theta)

    sin_alpha = np.sin(alpha)[np.newaxis, np.newaxis, :]
    cos_alpha = np.cos(alpha)[np.newaxis, np.newaxis, :]
    sin_beta = np.sin(beta)[np.newaxis, np.newaxis, :]
    cos_beta = np.cos(beta)[np.newaxis, np.newaxis, :]
    sin_theta = np.sin(theta)[np.newaxis, :, np.newaxis]
    cos_theta = np.cos(theta)[np.newaxis, :, np.newaxis]

    KX = K_norm * (cos_beta * sin_alpha + sin_beta * cos_alpha * cos_theta)
    KY = K_norm * sin_theta * cos_alpha
    return KX, KY


# --- Single layer conversion ---


def SingleLayerConversion(bands, KX, KY, kx_dim, ky_dim, layer_index=0, method="cubic"):
    """Interpolate a single layer of band data onto a regular kx/ky grid.

    Parameters
    ----------
    bands : ndarray
        3D intensity data (E, kx_angle, ky_angle).
    KX, KY : ndarray
        3D momentum coordinate arrays.
    kx_dim, ky_dim : int
        Output grid dimensions.
    layer_index : int
        Energy layer index to process.
    method : str
        Interpolation method ('cubic', 'linear', 'nearest').

    Returns
    -------
    E_Mon : ndarray
        2D interpolated intensity for the layer.
    kx, ky : ndarray
        kx and ky grid axes.
    """
    kx = np.linspace(np.min(KX), np.max(KX), kx_dim)
    ky = np.linspace(np.min(KY), np.max(KY), ky_dim)

    E_Mon = griddata(
        np.column_stack((KX[layer_index].ravel(), KY[layer_index].ravel())),
        bands[layer_index].ravel(),
        (kx[:, None], ky[None, :]),
        method=method,
    )
    return E_Mon, kx, ky


# --- KD-Tree interpolation ---


def KDInterp(bands, KX, KY, radius=0.05, kx_grid=None, ky_grid=None):
    """Interpolate band data using KD-Tree based merging.

    Parameters
    ----------
    bands : ndarray
        2D intensity data (kx_angle, ky_angle).
    KX, KY : ndarray
        2D momentum coordinate arrays.
    radius : float
        Merge radius for KD-Tree.
    kx_grid, ky_grid : ndarray
        2D coordinate grids for interpolation.

    Returns
    -------
    E_Mon : ndarray
        2D interpolated intensity.
    """
    points_source = np.vstack((KX.ravel(), KY.ravel())).T
    values_source = bands.ravel()

    tree = KDTree(points_source)
    neighbors_indices = tree.query_ball_point(points_source, r=radius)

    new_points = []
    new_values = []
    for i in range(len(points_source)):
        indices_in_neighborhood = neighbors_indices[i]
        points_in_neighborhood = points_source[indices_in_neighborhood]
        values_in_neighborhood = values_source[indices_in_neighborhood]
        new_points.append(np.mean(points_in_neighborhood, axis=0))
        new_values.append(np.mean(values_in_neighborhood))

    cubic_map = griddata(
        np.array(new_points),
        np.array(new_values),
        (kx_grid.ravel(), ky_grid.ravel()),
        method="cubic",
    )
    result = cubic_map.reshape(kx_grid.shape)
    result = np.nan_to_num(result, nan=0.0)
    return result


# --- Coordinate rotation ---


def RotateCoordinates(KX, KY, theta=60, KX_Shift=0, KY_Shift=0):
    """Rotate 2D coordinates by a given angle around (KX_Shift, KY_Shift).

    Parameters
    ----------
    KX, KY : ndarray
        Input coordinates.
    theta : float
        Rotation angle in degrees.
    KX_Shift, KY_Shift : float
        Center of rotation.

    Returns
    -------
    KX_rotated, KY_rotated : ndarray
        Rotated coordinates, centered at the same point as input.
    """
    theta_rad = np.deg2rad(theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    KX = KX - KX_Shift
    KY = KY - KY_Shift

    KX_rotated = KX * cos_t - KY * sin_t
    KY_rotated = KX * sin_t + KY * cos_t
    return KX_rotated + KX_Shift, KY_rotated + KY_Shift


# --- Multi-layer expansion ---


def expand_all_layers(bands, KX, KY, KX_Shift=0, KY_Shift=0, n_rotations=6):
    """Expand experimental data by rotating all layers.

    Parameters
    ----------
    bands : ndarray
        3D intensity data (E, kx_angle, ky_angle).
    KX, KY : ndarray
        3D momentum coordinate arrays.
    KX_Shift, KY_Shift : float
        Center of rotation.
    n_rotations : int
        Number of rotational copies.

    Returns
    -------
    KX_Rotated, KY_Rotated : ndarray
        Concatenated rotated coordinates.
    bands_repeated : ndarray
        Repeated bands for all rotations.
    """
    bands_repeated = np.repeat(bands[:, :, np.newaxis], n_rotations, axis=2)
    bands_repeated = bands_repeated.reshape(bands.shape[0], bands.shape[1], bands.shape[2] * n_rotations)

    KX_Rotated = np.zeros((KX.shape[0], KX.shape[1], KX.shape[2] * n_rotations))
    KY_Rotated = np.zeros_like(KX_Rotated)

    KX_Rotated[:, :, : KX.shape[2]] = KX
    KY_Rotated[:, :, : KY.shape[2]] = KY

    for i in range(1, n_rotations):
        KX_rot, KY_rot = RotateCoordinates(KX, KY, theta=60 * i, KX_Shift=KX_Shift, KY_Shift=KY_Shift)
        KX_Rotated[:, :, i * KX.shape[2] : (i + 1) * KX.shape[2]] = KX_rot
        KY_Rotated[:, :, i * KY.shape[2] : (i + 1) * KY.shape[2]] = KY_rot

    return KX_Rotated, KY_Rotated, bands_repeated


# --- Save preprocessed data ---


def save_preprocessed_h5(filepath, E_grid, kx, ky, E_Mon):
    """Save preprocessed experimental data as HDF5.

    Parameters
    ----------
    filepath : str
        Output file path.
    E_grid, kx, ky : ndarray
        Axes.
    E_Mon : ndarray
        3D intensity data (E, kx, ky).
    """
    if E_grid[0] > E_grid[-1]:
        E_grid = E_grid[::-1].copy()
        E_Mon = E_Mon[::-1, :, :].copy()
    with h5py.File(filepath, "w") as f:
        f.create_group("axes")
        f.create_group("intensity")
        f["axes"]["E"] = E_grid
        f["axes"]["kx"] = kx
        f["axes"]["ky"] = ky
        f["intensity"]["V"] = E_Mon