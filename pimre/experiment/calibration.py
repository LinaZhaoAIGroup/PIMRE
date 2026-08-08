"""Experimental data preprocessing: loading, calibration, angle-to-momentum conversion.

Replicates the pipeline from 2.exp_data_pre.ipynb.
"""

import h5py
import numpy as np
from scipy.constants import electron_volt, hbar, m_e
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

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

    tree = cKDTree(points_source)
    neighbors_indices = tree.query_ball_point(points_source, r=radius, return_sorted=False)

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


# --- Quadrant symmetrization ---


def quadrant_symmetrize(bands, KX, KY, flip_kx=True, flip_ky=True,
                        kx_grid=None, ky_grid=None):
    """Reconstruct the full Brillouin zone from its 1/4 (kx>=0, ky>=0) crop.

    Scattered momentum points (KX, KY) with Gamma at the origin are cropped
    to the first quadrant, mirrored across the kx=0 / ky=0 axes (flip_kx /
    flip_ky, both on by default) and binned onto the regular output grid.
    Copies landing in the same bin are averaged.

    Parameters
    ----------
    bands : 2D array
        Intensity per (kx_angle, ky_angle) pixel of a single energy layer.
    KX, KY : 2D array
        Momentum coordinates of each pixel (scattered points).
    flip_kx : bool
        Mirror across the ky=0 axis (expand into kx<0).
    flip_ky : bool
        Mirror across the kx=0 axis (expand into ky<0).
    kx_grid, ky_grid : 2D array or None
        Target regular grid; if None, inferred from the mirrored points.

    Returns
    -------
    E_Mon : 2D array
        Symmetrized intensity on the regular grid (NaN filled with 0).
    """
    mask = (KX >= 0) & (KY >= 0)
    pts = np.column_stack((KX[mask], KY[mask]))
    vals = bands[mask]

    kx, ky, v = pts[:, 0], pts[:, 1], vals
    if flip_kx:
        kx = np.concatenate((kx, -kx))
        ky = np.concatenate((ky, ky))
        v = np.concatenate((v, v))
    if flip_ky:
        kx = np.concatenate((kx, kx))
        ky = np.concatenate((ky, -ky))
        v = np.concatenate((v, v))

    if kx_grid is None or ky_grid is None:
        n = int(np.ceil(np.max(np.abs(kx))))
        kx_out = np.linspace(-n, n, 2 * n + 1)
        ky_out = np.linspace(-n, n, 2 * n + 1)
    else:
        kx_out = kx_grid[:, 0] if kx_grid.ndim == 2 else kx_grid
        ky_out = ky_grid[0, :] if ky_grid.ndim == 2 else ky_grid

    nx, ny = kx_out.shape[0], ky_out.shape[0]
    kx_edges = (kx_out[:-1] + kx_out[1:]) / 2
    ky_edges = (ky_out[:-1] + ky_out[1:]) / 2
    ix = np.clip(np.digitize(kx, kx_edges), 0, nx - 1)
    iy = np.clip(np.digitize(ky, ky_edges), 0, ny - 1)

    sums = np.zeros((nx, ny))
    counts = np.zeros((nx, ny))
    np.add.at(sums, (ix, iy), v)
    np.add.at(counts, (ix, iy), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        E_Mon = np.divide(sums, counts)
    E_Mon = np.nan_to_num(E_Mon, nan=0.0)
    return E_Mon


# --- Multi-layer expansion ---



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
