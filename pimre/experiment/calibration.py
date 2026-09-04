"""Experimental data preprocessing: loading, calibration, angle-to-momentum conversion.

Replicates the pipeline from 2.exp_data_pre.ipynb.
"""

import h5py
import numpy as np
from scipy.constants import electron_volt, hbar, m_e
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

# --- Angle to momentum conversion ---


def _neighborhood_sums(tree, points, values, radius, chunk=2048):
    """KD-tree radius neighborhoods: per-point sums and counts.

    Querying and accumulating in chunks keeps the memory footprint bounded:
    scipy's ``query_ball_point`` returns lists of Python ints (~450 MB for
    25k dense points) and a global flat-index array plus its fancy-index
    temporaries would hold ~1 GB live per worker; with ~200 MB per chunk a
    12-process pool stays inside physical memory.  Per point, the
    contributions are still summed in query order, so the means match the
    original per-point Python loop to float rounding.

    Returns
    -------
    sum_x, sum_y, sum_v : ndarray
        Per-point sums of the neighbor coordinates and values.
    counts : ndarray of int64
        Per-point neighborhood sizes.
    """
    n = points.shape[0]
    sum_x = np.zeros(n)
    sum_y = np.zeros(n)
    sum_v = np.zeros(n)
    counts = np.zeros(n, dtype=np.int64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        nb = tree.query_ball_point(points[start:stop], r=radius,
                                   return_sorted=False)
        c = np.fromiter((len(x) for x in nb), dtype=np.int64, count=stop - start)
        counts[start:stop] = c
        f = np.concatenate(nb).astype(np.int64, copy=False)
        s = np.repeat(np.arange(stop - start, dtype=np.int64), c)
        sum_x[start:stop] = np.bincount(s, weights=points[f, 0],
                                        minlength=stop - start)
        sum_y[start:stop] = np.bincount(s, weights=points[f, 1],
                                        minlength=stop - start)
        sum_v[start:stop] = np.bincount(s, weights=values[f],
                                        minlength=stop - start)
    return sum_x, sum_y, sum_v, counts





def Angle2Mon(E_grid, X_Angle, Y_Angle, X_Shift=0, Y_Shift=0, work_function=16.03):
    """Convert angles to momentum space coordinates (KX, KY).

    Parameters
    ----------
    E_grid : 1D array
        Energy grid: binding energy in eV relative to E_F (positive below
        E_F).
    X_Angle : 1D array
        Angles in X direction (degrees), typically alpha + beta.
    Y_Angle : 1D array
        Angles in Y direction (degrees), typically theta.
    X_Shift, Y_Shift : float
        Angle shifts in X and Y directions.
    work_function : float
        NOTE: despite the name this is the kinetic energy at the Fermi
        level, i.e. hnu - Phi_analyzer in eV (e.g. He-I lamp 21.22 eV minus
        an ~5.2 eV analyzer work function gives ~16 eV).  E_kin = this
        value - binding energy.

    Returns
    -------
    KX, KY : ndarray
        3D arrays of momentum coordinates (E, kx_angle, ky_angle).  KY is
        approximated without the cos(X) factor; the error is negligible for
        |angles| < ~10 deg and reaches a few percent at ±20 deg.
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

    Notes
    -----
    The neighborhood means and the cubic interpolation are vectorized, but
    the per-layer cost is dominated by the single-threaded cubic
    ``griddata``; parallelize over energy layers with
    ``preprocessing.workers`` in the pipeline instead.

    Returns
    -------
    E_Mon : ndarray
        2D interpolated intensity.
    """
    points_source = np.vstack((KX.ravel(), KY.ravel())).T
    values_source = bands.ravel()

    tree = cKDTree(points_source)
    sum_x, sum_y, sum_v, counts = _neighborhood_sums(
        tree, points_source, values_source, radius)

    # Every point is its own neighbor, so each neighborhood holds at least
    # one element and the division below is safe.
    new_points = np.column_stack((sum_x / counts, sum_y / counts))
    new_values = sum_v / counts

    cubic_map = griddata(
        new_points,
        new_values,
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


def _flip_join(arr, flip, include_axis):
    """Mirror a 2D array across its leading axis and join the halves.

    Parameters
    ----------
    arr : 2D array
        Upper/right half including the symmetry axis (first row/column
        lies on kx=0 or ky=0 if include_axis is True).
    flip : bool
        Whether to mirror at all.
    include_axis : bool
        True if the first row/column of arr is the symmetry axis (it is
        shared by both halves and must not be duplicated).

    Returns
    -------
    joined : 2D array
        Full array symmetric about the axis.
    """
    if not flip:
        return arr
    if include_axis:
        return np.concatenate((arr[1:][::-1], arr), axis=0)
    return np.concatenate((arr[::-1], arr), axis=0)


def quadrant_symmetrize(bands, KX, KY, flip_kx=True, flip_ky=True,
                        kx_grid=None, ky_grid=None, interp_method="cubic",
                        smooth_radius=0.02, fill_radius=0.03):
    """Reconstruct the full Brillouin zone from its 1/4 (kx>=0, ky>=0) crop.

    The scattered momentum points (KX, KY) with Gamma at the origin are
    cropped to the first quadrant, interpolated onto a regular grid covering
    only that quadrant (1/4 of the interpolation work of the full-plane
    KD-tree path), and the full BZ is then obtained by pure array
    mirror/flip operations (flip_kx / flip_ky, both on by default). The
    symmetry axis is shared between the mirrored halves, so the result is
    exactly symmetric.

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
        Target regular grid of the full BZ; if None, inferred from the
        mirrored points.
    interp_method : str
        Griddata interpolation method ('cubic' or 'linear').
    smooth_radius : float
        Radius (1/Angstrom) of a light neighborhood average applied to the
        scattered 1/4-BZ pixels before interpolation, to reduce shot noise
        while keeping most band details (0 or None disables smoothing).
        For reference the KD-tree path uses radius=0.05 which visibly
        smooths away fine band structure.
    fill_radius : float
        Maximum distance (1/Angstrom) from the scattered pixels at which
        interpolation holes (Gamma corner, hull boundary) are filled with
        the nearest pixel value.  Farther holes stay zero so that
        genuinely uncovered regions (e.g. K points outside the measured
        window) remain distinguishable from measured data.  None fills
        every hole.

    Returns
    -------
    E_Mon : 2D array
        Symmetrized intensity on the regular grid (NaN filled with 0).
    """
    mask = (KX >= 0) & (KY >= 0)
    pts = np.column_stack((KX[mask], KY[mask]))
    vals = bands[mask]

    if smooth_radius:
        tree = cKDTree(pts)
        _, _, sum_v, counts = _neighborhood_sums(tree, pts, vals,
                                                 float(smooth_radius))
        vals = sum_v / counts

    if kx_grid is None or ky_grid is None:
        n = int(np.ceil(np.max(np.abs(pts[:, 0]))))
        kx_out = np.linspace(-n, n, 2 * n + 1)
        ky_out = np.linspace(-n, n, 2 * n + 1)
    else:
        kx_out = kx_grid[:, 0] if kx_grid.ndim == 2 else kx_grid
        ky_out = ky_grid[0, :] if ky_grid.ndim == 2 else ky_grid

    # 1/4-BZ grid: right-upper half of the full grid (kx>=0, ky>=0).
    ix0 = int(np.searchsorted(kx_out, 0.0))
    iy0 = int(np.searchsorted(ky_out, 0.0))
    kx_h = kx_out[ix0:]
    ky_h = ky_out[iy0:]
    kx_axis = bool(np.isclose(kx_h[0], 0.0))
    ky_axis = bool(np.isclose(ky_h[0], 0.0))

    # Interpolate the scattered 1/4-BZ pixels onto the 1/4 regular grid.
    half = griddata(pts, vals, (kx_h[:, None], ky_h[None, :]),
                    method=interp_method)
    if np.isnan(half).any():
        # Fill interpolation holes (Gamma corner, hull boundary) with the
        # nearest pixel value, but only close to the measured pixels so
        # genuinely uncovered regions stay zero.
        half_nearest = griddata(pts, vals, (kx_h[:, None], ky_h[None, :]),
                                method="nearest")
        if fill_radius:
            tree = cKDTree(pts)
            pts_grid = np.column_stack(
                (kx_h[:, None].repeat(ky_h.size), np.tile(ky_h, kx_h.size)))
            dist, _ = tree.query(pts_grid)
            dist = dist.reshape(half.shape)
            half_nearest = np.where(dist <= fill_radius, half_nearest, 0.0)
        half = np.where(np.isnan(half), half_nearest, half)
    half = np.nan_to_num(half, nan=0.0)

    # Pure array mirror operations to the full BZ.
    full = _flip_join(half, flip_kx, kx_axis)
    full = _flip_join(full.T, flip_ky, ky_axis).T

    # Align to the full-grid shape when a flip is disabled (the empty
    # half/quadrant stays zero-filled).
    nx, ny = kx_out.shape[0], ky_out.shape[0]
    if full.shape != (nx, ny):
        padded = np.zeros((nx, ny))
        kx0 = 0 if flip_kx else ix0
        ky0 = 0 if flip_ky else iy0
        padded[kx0:kx0 + full.shape[0], ky0:ky0 + full.shape[1]] = full
        full = padded
    return full


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
