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

    Parameters
    ----------
    filepath : str
        Path to the BAND_GAP file.

    Returns
    -------
    fermi_energy : float
        Fermi energy in eV.
    vbm_index : int
        Valence band maximum band index.
    cbm_index : int
        Conduction band minimum band index.
    """
    with open(filepath) as f:
        text = f.read()

    fermi_energy = None
    fermi_index = text.find("Fermi Energy (eV)")
    if fermi_index != -1:
        start_index = fermi_index + len("Fermi Energy (eV):")
        end_index = start_index
        while text[end_index].isspace() or text[end_index].isdigit() or text[end_index] == "." or text[end_index] == "-":
            end_index += 1
        fermi_energy = float(text[start_index:end_index].strip())

    vbm_index = None
    cbm_index = None
    vbm_cbm_indexes_index = text.find("Band Indexes of VBM & CBM:")
    if vbm_cbm_indexes_index != -1:
        start_index = vbm_cbm_indexes_index + len("Band Indexes of VBM & CBM:")
        numbers = []
        while True:
            number = ""
            while text[start_index].isspace():
                start_index += 1
            while text[start_index].isdigit():
                number += text[start_index]
                start_index += 1
            if number:
                numbers.append(int(number))
            if text[start_index] != " ":
                break
        vbm_index, cbm_index = numbers

    return fermi_energy, vbm_index, cbm_index


def read_dft_csv(filepath, fermi_energy, nkx=20, nky=20):
    """Read DFT band structure data from CSV and reshape to (nkx, nky, nbands).

    Parameters
    ----------
    filepath : str
        Path to the CSV file with k-points and band energies.
    fermi_energy : float
        Fermi energy to subtract from band energies.
    nkx, nky : int
        Number of k-points along kx and ky.

    Returns
    -------
    cartesian_coords : ndarray
        Cartesian coordinates of k-points (nk, 2).
    ebands : ndarray
        Energy bands reshaped to (nkx, nky, nbands).
    """
    import pandas as pd

    df = pd.read_csv(filepath, header=None)
    reciprocal_coords = df.iloc[:, :3].astype(float).drop(columns=[2])
    M = reciprocal_to_cartesian_matrix()
    cartesian_coords = M.dot(reciprocal_coords.T).T

    energy_bands = df.iloc[:, 1:].values - float(fermi_energy)
    energy_bands = energy_bands[:, ~np.all(np.isnan(energy_bands), axis=0)]

    combined = np.concatenate((cartesian_coords, energy_bands), axis=1)
    combined = combined[1:]  # Skip first row (gamma point duplicate)
    nk = nkx * nky
    neb = combined.shape[1]
    ebands = np.moveaxis(combined.reshape((nkx, nky, neb)), 0, 1)[:, :, :-1]
    return cartesian_coords, energy_bands, ebands


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
        (n_bands, n_x, n_y) band map.
    carcoo : ndarray
        (2, n_x, n_y) coordinate grid.
    """
    max_x = np.max(bz_coords[:, 0])
    min_x = np.min(bz_coords[:, 0])
    max_y = np.max(bz_coords[:, 1])
    min_y = np.min(bz_coords[:, 1])

    step_x = (max_x - min_x) / n_x
    step_y = (max_y - min_y) / n_y

    BANDMAP = []
    CARCOO = []

    for i in np.arange(min_x - n_plus * step_x, max_x + n_plus * step_x, step_x):
        x_end = i + scaling_factor * step_x
        x_coord = x_end / 2
        for j in np.arange(min_y - n_plus * step_y, max_y + n_plus * step_y, step_y):
            y_end = j + scaling_factor * step_y
            y_coord = y_end / 2
            mask = (
                (bz_coords[:, 0] >= i)
                & (bz_coords[:, 0] < x_end)
                & (bz_coords[:, 1] >= j)
                & (bz_coords[:, 1] < y_end)
            )
            if np.any(mask):
                indices = np.where(mask)[0]
                indices = indices % energy_bands.shape[0]
                avg_band = np.nanmean(energy_bands[indices], axis=0)
            else:
                avg_band = np.zeros(energy_bands.shape[1])
            BANDMAP.append(avg_band)
            CARCOO.append([x_coord, y_coord])

    length = n_x + 2 * n_plus
    width = len(BANDMAP) // length
    BANDMAP = np.array(BANDMAP).reshape((length, width, energy_bands.shape[1]))
    CARCOO = np.array(CARCOO).reshape((length, width, 2))
    BANDMAP = np.moveaxis(BANDMAP, -1, 0)
    CARCOO = np.moveaxis(CARCOO, -1, 0)
    return BANDMAP, CARCOO


# --- High-symmetry path extraction ---


def extract_high_symmetry_path(ebands):
    """Extract energy bands along the Gamma-M-K-Gamma path.

    Parameters
    ----------
    ebands : ndarray
        (nkx, nky, nbands) energy band data.

    Returns
    -------
    high_symmetry_points : ndarray
        (n_points, n_bands) energy along the path.
    gamma1, m, k, gamma2 : int
        Indices of high-symmetry points along the path.
    """
    high_symmetry_points = np.vstack((ebands[0, :, 2:], ebands[:, -1, 2:], ebands[-1, ::-1, 2:]))
    gamma1 = 0
    m = ebands[0, :, 2:].shape[0]
    k = m + ebands[:, -1, 2:].shape[0]
    gamma2 = high_symmetry_points.shape[0] - 1
    return high_symmetry_points, gamma1, m, k, gamma2


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
    """Save band map as HDF5 file.

    Parameters
    ----------
    filepath : str
        Output file path.
    evb, ecb : ndarray
        Valence and conduction band data.
    kx_grid, ky_grid : ndarray
        kx and ky grid arrays.
    """
    import h5py

    with h5py.File(filepath, "w") as f:
        axes_group = f.create_group("axes")
        axes_group.create_dataset("kxxsc", data=kx_grid)
        axes_group.create_dataset("kyysc", data=ky_grid)
        bands_group = f.create_group("bands")
        bands_group.create_dataset("evb", data=evb)
        bands_group.create_dataset("ecb", data=ecb)