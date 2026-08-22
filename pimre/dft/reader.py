"""DFT calculation data processing: reading, coordinate transformation, interpolation.

Replicates the pipeline from 1.Calu_Data_Processing.ipynb.
"""

import re

import numpy as np
import scipy.io as sio
from scipy.interpolate import griddata

# --- Coordinate transforms ---


def rotate(theta):
    """Return a 2x2 rotation matrix."""
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def reflect():
    """Return a 2x2 reflection matrix (flips y)."""
    return np.array([[1, 0], [0, -1]])


def reciprocal_to_cartesian_matrix():
    """Return the 2x2 matrix for reciprocal-to-Cartesian coordinate transform."""
    return np.array([[1, 0.5], [0, np.sqrt(3) / 2]])


# --- File readers ---


def read_fermi_energy(filepath):
    """Read Fermi energy from FERMI_ENERGY file.

    Parameters
    ----------
    filepath : str
        Path to the FERMI_ENERGY file.

    Returns
    -------
    float
        Fermi energy value.
    """
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                match = re.search(r"[-+]?\d*\.\d+|\d+", line)
                if match:
                    return float(match.group())
    return None


def read_band_gap(filepath):
    """Read Fermi energy, VBM and CBM band indices from BAND_GAP file.

    VBM/CBM indices are returned 0-based (BAND_GAP files written by VASPKIT
    use 1-based indices consistent with EIGENVAL, so 1 is subtracted here).

    Parameters
    ----------
    filepath : str
        Path to the BAND_GAP file.

    Returns
    -------
    fermi_energy : float
        Fermi energy in eV.
    vbm_index : int or None
        Valence band maximum band index (0-based).
    cbm_index : int or None
        Conduction band minimum band index (0-based).
    """
    with open(filepath) as f:
        text = f.read()

    fermi_energy = None
    match = re.search(r"Fermi Energy \(eV\)\s*[:=]?\s*([-+]?\d*\.?\d+)", text)
    if match:
        fermi_energy = float(match.group(1))

    vbm_index = None
    cbm_index = None
    match = re.search(r"Band Indexes of VBM & CBM\s*[:=]?\s*(\d+)\s+(\d+)", text)
    if match:
        vbm_index = int(match.group(1)) - 1
        cbm_index = int(match.group(2)) - 1

    return fermi_energy, vbm_index, cbm_index


def read_dft_csv(filepath, fermi_energy, nkx=20, nky=20):
    """Read DFT band structure data from CSV.

    The CSV is expected to contain k-point coordinates (kx, ky, kz) followed
    by band energies per row, either comma-separated or with mixed
    whitespace/comma delimiters. The kz column and any all-NaN columns are
    dropped; the first row (usually the Gamma point) is kept.

    If the number of rows matches a regular nkx × nky grid the bands are
    reshaped to (nkx, nky, nbands); otherwise (scattered k-points) the bands
    are returned as-is.

    Parameters
    ----------
    filepath : str
        Path to the CSV file with k-points and band energies.
    fermi_energy : float
        Fermi energy to subtract from band energies.
    nkx, nky : int
        Number of k-points along kx and ky (used only for grid detection).

    Returns
    -------
    cartesian_coords : ndarray
        Cartesian coordinates of k-points (nk, 2).
    energy_bands : ndarray
        Energy bands (nk, nbands), Fermi-shifted.
    ebands : ndarray
        Energy bands reshaped to (nkx, nky, nbands) for regular grids,
        otherwise the same array as energy_bands.
    """
    df = _read_csv_mixed(filepath)
    reciprocal_coords = df.iloc[:, :3].astype(float)
    M = reciprocal_to_cartesian_matrix()
    cartesian_coords = M.dot(reciprocal_coords.values[:, :2].T).T

    energy_bands = df.iloc[:, 3:].values - float(fermi_energy)
    energy_bands = energy_bands[:, ~np.all(np.isnan(energy_bands), axis=0)]

    nk = cartesian_coords.shape[0]
    neb = energy_bands.shape[1]
    if nk == nkx * nky:
        ebands = np.moveaxis(energy_bands.reshape((nkx, nky, neb)), 0, 1)
    else:
        ebands = energy_bands.copy()
    return cartesian_coords, energy_bands, ebands


def _read_csv_mixed(filepath):
    """Read a DFT CSV supporting comma and whitespace delimiters."""
    import pandas as pd

    try:
        df = pd.read_csv(filepath, header=None)
        if not _is_numeric(df.iloc[:, 0]):
            raise ValueError
        return df
    except (ValueError, TypeError, pd.errors.ParserError):
        df = pd.read_csv(filepath, header=None, sep=r"[\s,]+", engine="python")
        return df


def _is_numeric(series):
    """Check whether a pandas Series can be converted to float."""
    try:
        series.astype(float)
        return True
    except (ValueError, TypeError):
        return False


# --- Brillouin zone expansion ---


def expand_bz(cartesian_coords, energy_bands, n_rotations=6):
    """Expand Brillouin zone by 6-fold rotation and reflection.

    Parameters
    ----------
    cartesian_coords : ndarray
        Cartesian coordinates of k-points in the irreducible BZ.
    energy_bands : ndarray
        Energy bands in the irreducible BZ.
    n_rotations : int
        Number of rotational symmetries.

    Returns
    -------
    bz_coords : ndarray
        Cartesian coordinates of the full BZ.
    repeated_bands : ndarray
        Energy bands repeated for the full BZ.
    """
    M = reflect()
    bz_coords = np.vstack((M.dot(cartesian_coords.T).T, cartesian_coords))
    bz_copy = bz_coords.copy()

    for i in range(1, n_rotations):
        theta = i * np.pi / 3
        R = rotate(theta)
        rotated = R.dot(bz_copy.T).T
        bz_coords = np.vstack((bz_coords, rotated))

    repeated_bands = np.tile(energy_bands, (2 * n_rotations, 1))
    return bz_coords, repeated_bands


# --- Grid interpolation ---


def interpolate_to_grid(coords, bands, nx=101, ny=101, method="cubic"):
    """Interpolate scattered band data to a uniform grid.

    Parameters
    ----------
    coords : ndarray
        (n_points, 2) array of (kx, ky) coordinates.
    bands : ndarray
        (n_points, n_bands) array of band energies.
    nx, ny : int
        Grid dimensions.
    method : str
        Interpolation method ('cubic', 'linear', 'nearest').

    Returns
    -------
    mapping : ndarray
        (n_bands, nx, ny) interpolated band map.
    kx_grid : ndarray
        2D array of kx grid values.
    ky_grid : ndarray
        2D array of ky grid values.
    """
    kx = np.linspace(coords[:, 0].min(), coords[:, 0].max(), nx)
    ky = np.linspace(coords[:, 1].min(), coords[:, 1].max(), ny)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")

    mapping = np.full((bands.shape[1], nx, ny), np.nan)
    for i in range(bands.shape[1]):
        mapping[i] = griddata(
            (coords[:, 0], coords[:, 1]), bands[:, i], (kx_grid, ky_grid), method=method
        )

    return mapping, kx_grid, ky_grid


def build_band_map_3d(bz_coords, energy_bands, n_x=70, n_y=70, n_plus=3, scaling_factor=2):
    """Build 3D band map by averaging energy bands in grid cells.

    Parameters
    ----------
    bz_coords : ndarray
        (n_points, 2) Cartesian coordinates in the full BZ.
    energy_bands : ndarray
        (n_points, n_bands) energy values.
    n_x, n_y : int
        Number of grid cells along x and y.
    n_plus : int
        Extra cells on each side.
    scaling_factor : float
        Cell size scaling factor.

    Returns
    -------
    bandmap : ndarray
        (n_bands, n_x + 2*n_plus, n_y + 2*n_plus) band map; uncovered
        cells are NaN.
    carcoo : ndarray
        (2, n_x + 2*n_plus, n_y + 2*n_plus) coordinate grid.
    """
    max_x = np.max(bz_coords[:, 0])
    min_x = np.min(bz_coords[:, 0])
    max_y = np.max(bz_coords[:, 1])
    min_y = np.min(bz_coords[:, 1])

    step_x = (max_x - min_x) / n_x
    step_y = (max_y - min_y) / n_y

    # Integer-counted cell origins: np.arange with float steps can drift by
    # one cell and break the reshape below.
    nxt = n_x + 2 * n_plus
    nyt = n_y + 2 * n_plus

    BANDMAP = []
    CARCOO = []

    for i in range(nxt):
        x_lo = min_x + (i - n_plus) * step_x
        x_end = x_lo + scaling_factor * step_x
        x_coord = (x_lo + x_end) / 2
        for j in range(nyt):
            y_lo = min_y + (j - n_plus) * step_y
            y_end = y_lo + scaling_factor * step_y
            y_coord = (y_lo + y_end) / 2
            mask = (
                (bz_coords[:, 0] >= x_lo)
                & (bz_coords[:, 0] < x_end)
                & (bz_coords[:, 1] >= y_lo)
                & (bz_coords[:, 1] < y_end)
            )
            if np.any(mask):
                indices = np.where(mask)[0]
                avg_band = np.nanmean(energy_bands[indices], axis=0)
            else:
                # Uncovered cells stay NaN so that downstream stages can
                # distinguish "no DFT data" from a real band at 0 eV (E_F).
                avg_band = np.full(energy_bands.shape[1], np.nan)
            BANDMAP.append(avg_band)
            CARCOO.append([x_coord, y_coord])

    BANDMAP = np.array(BANDMAP).reshape((nxt, nyt, energy_bands.shape[1]))
    CARCOO = np.array(CARCOO).reshape((nxt, nyt, 2))
    BANDMAP = np.moveaxis(BANDMAP, -1, 0)
    CARCOO = np.moveaxis(CARCOO, -1, 0)
    return BANDMAP, CARCOO



# --- Save utilities ---


def save_band_map_mat(filepath, evb, ecb, kx_grid, ky_grid):
    """Save band map as MATLAB .mat file.

    Parameters
    ----------
    filepath : str
        Output file path.
    evb : ndarray
        Valence band data.
    ecb : ndarray
        Conduction band data.
    kx_grid : ndarray
        kx grid.
    ky_grid : ndarray
        ky grid.
    """
    sio.savemat(
        filepath,
        {"evb": evb, "ecb": ecb, "kxxsc": kx_grid, "kyysc": ky_grid},
    )


def save_band_map_h5(filepath, evb, ecb, kx_grid, ky_grid):
    """Save band map as HDF5 file with 1D kx/ky axes.

    Parameters
    ----------
    filepath : str
        Output file path.
    evb, ecb : ndarray
        Valence and conduction band data (n_bands, nkx, nky).
    kx_grid, ky_grid : ndarray
        2D kx and ky grid arrays. First row/column is extracted as 1D axes.
    """
    import h5py

    if kx_grid.ndim == 2:
        kx = kx_grid[:, 0] if np.abs(np.sum(np.diff(kx_grid[:, 0]))) > 0 else kx_grid[0, :]
    else:
        kx = kx_grid
    if ky_grid.ndim == 2:
        ky = ky_grid[0, :] if np.abs(np.sum(np.diff(ky_grid[0, :]))) > 0 else ky_grid[:, 0]
    else:
        ky = ky_grid

    with h5py.File(filepath, "w") as f:
        axes_group = f.create_group("axes")
        axes_group.create_dataset("kx", data=kx)
        axes_group.create_dataset("ky", data=ky)
        bands_group = f.create_group("bands")
        bands_group.create_dataset("evb", data=evb)
        bands_group.create_dataset("ecb", data=ecb)


def load_band_map_h5(filepath, drop_top_bands=None):
    """Load band map from HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to band_map.h5.
    drop_top_bands : int or None
        Number of highest-energy conduction bands to drop from the stacked
        band structure. None (default) keeps all bands.

    Returns
    -------
    E_dft : ndarray
        Stacked band structure (n_bands, nkx, nky) in descending
        Gamma-point energy order. Cells without DFT coverage are NaN
        (NOT zero — 0 eV would masquerade as a flat band at E_F).
    evb, ecb : ndarray
        Valence and conduction band data.
    kx, ky : 1D array
        kx and ky axes.
    """
    import h5py

    with h5py.File(filepath, "r") as f:
        evb = f["bands/evb"][:]
        ecb = f["bands/ecb"][:]
        kx = f["axes/kx"][:]
        ky = f["axes/ky"][:]

    E_dft = np.vstack((ecb[::-1], evb))
    if drop_top_bands:
        E_dft = E_dft[drop_top_bands:]
    return E_dft, evb, ecb, kx, ky
