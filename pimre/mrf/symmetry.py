"""Symmetrization utilities for MRF band reconstruction.

Extracted from fuller.generator with symmetrize dependency replaced by inline implementations.
"""

import numpy as np


def sym_band(ind_band, recon, kx, ky, lengthKx, lengthKy):
    """Symmetrize a reconstructed band with respect to kx and ky axes.

    The mirror pivot is taken as the first strictly-positive axis value, so
    the pairing (indXRef-d, indXRef+d) reproduces true mirror partners only
    when the axis is symmetric about k=0 on the grid (the case for the
    linspace output grids of the preprocessing stage).

    Parameters
    ----------
    ind_band : int
        Band index.
    recon : 3D array
        Reconstructed bands (nbands, kx, ky).
    kx, ky : 1D array
        Momentum axes.
    lengthKx, lengthKy : int
        Length of kx and ky axes.
    """
    indXRef = np.min(np.where(kx > 0.0)[0])
    lIndX = np.min([indXRef, lengthKx - indXRef])
    indX = np.arange(indXRef - lIndX, indXRef + lIndX)
    recon[ind_band, indX, :] = (recon[ind_band, indX, :] + recon[ind_band, np.flip(indX, axis=0), :]) / 2

    indYRef = np.min(np.where(ky > 0.0)[0])
    lIndY = np.min([indYRef, lengthKy - indYRef])
    indY = np.arange(indYRef - lIndY, indYRef + lIndY)
    recon[ind_band, :, indY] = (recon[ind_band, :, indY] + recon[ind_band, :, np.flip(indY, axis=0)]) / 2
