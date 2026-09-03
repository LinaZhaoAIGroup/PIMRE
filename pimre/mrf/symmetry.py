"""Symmetrization utilities for MRF band reconstruction.

Extracted from fuller.generator with symmetrize dependency replaced by inline implementations.
"""

import numpy as np


def sym_band(ind_band, recon, kx, ky, lengthKx, lengthKy):
    """Symmetrize a reconstructed band with respect to kx and ky axes.

    The mirror pairing assumes the axis is symmetric about k=0 on the grid
    (the case for the linspace output grids of the preprocessing stage).
    An axis with no strictly-positive value (e.g. pushed entirely off zero
    by a large grid shift) has no mirror partner in the window and that
    axis is left unchanged.
    """
    pos_x = np.where(kx > 0.0)[0]
    if pos_x.size == 0:
        print("  sym_band: kx axis has no positive values, skipping kx mirror")
    else:
        indXRef = np.min(pos_x)
        lIndX = np.min([indXRef, lengthKx - indXRef])
        indX = np.arange(indXRef - lIndX, indXRef + lIndX)
        recon[ind_band, indX, :] = (recon[ind_band, indX, :] + recon[ind_band, np.flip(indX, axis=0), :]) / 2

    pos_y = np.where(ky > 0.0)[0]
    if pos_y.size == 0:
        print("  sym_band: ky axis has no positive values, skipping ky mirror")
    else:
        indYRef = np.min(pos_y)
        lIndY = np.min([indYRef, lengthKy - indYRef])
        indY = np.arange(indYRef - lIndY, indYRef + lIndY)
        recon[ind_band, :, indY] = (recon[ind_band, :, indY] + recon[ind_band, :, np.flip(indY, axis=0)]) / 2
