"""High-symmetry point finding and lattice-to-reciprocal conversion.

Extracted from ArpesBandRecons.CoordTrans and 4.mrf.ipynb.
"""

import math
import numpy as np


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

    KP_x = np.argmin(np.abs(kx_dft - KP[0, 1]))
    KP_y = np.argmin(np.abs(ky_dft - KP[0, 0]))
    MP_x = np.argmin(np.abs(kx_dft - MP[0, 1]))
    MP_y = np.argmin(np.abs(ky_dft - MP[0, 0]))
    return (KP_x, KP_y), (MP_x, MP_y)


def Get_G_M_K(Crygra_data, kx, ky):
    """Find Gamma, M, K high-symmetry point indices in experimental k-grid.

    Parameters
    ----------
    Crygra_data : list
        [a, b, c, alpha, beta, gamma] lattice parameters.
    kx, ky : 1D array
        Experimental momentum axes.

    Returns
    -------
    G, M, M1, K, K1, K2, K3, K4, K5 : tuple of ints
        Grid indices of high-symmetry points.
    """
    G = (np.argmin(np.abs(kx)), np.argmin(np.abs(ky)))
    k_K, k_M = lattice_to_reciprocal(*Crygra_data)

    k_K1 = (2 * kx[G[0]] - k_K[0], k_K[1])
    k_K2 = kx[G[0]] - math.sqrt((kx[G[0]] - k_K[0]) ** 2 + (ky[G[1]] - k_K[1]) ** 2)
    k_K5 = kx[G[0]] + math.sqrt((kx[G[0]] - k_K[0]) ** 2 + (ky[G[1]] - k_K[1]) ** 2)

    k_M1 = (2 * kx[G[0]] - k_M[0], k_M[1])

    K = (np.argmin(np.abs(kx - k_K[0])), np.argmin(np.abs(ky - k_K[1])))
    K1 = (np.argmin(np.abs(kx - k_K1[0])), np.argmin(np.abs(ky - k_K1[1])))
    K2 = (np.argmin(np.abs(kx - k_K2)), G[1])
    K3 = (K1[0], 2 * G[1] - K1[1])
    K4 = (K[0], K3[1])
    K5 = (np.argmin(np.abs(kx - k_K5)), G[1])

    M = (np.argmin(np.abs(kx - k_M[0])), np.argmin(np.abs(ky - k_M[1])))
    M1 = (np.argmin(np.abs(kx - k_M1[0])), np.argmin(np.abs(ky - k_M1[1])))

    return G, M, M1, K, K1, K2, K3, K4, K5