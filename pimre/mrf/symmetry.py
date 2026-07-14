"""Symmetrization utilities for MRF band reconstruction.

Extracted from fuller.generator with symmetrize dependency replaced by inline implementations.
"""

import numpy as np
import scipy.ndimage as ndi


def to_masked(arr, val=0):
    """Convert to masked array by setting val to NaN."""
    arrm = arr.copy()
    arrm[arrm == val] = np.nan
    return arrm


def coordinate_matrix_2D(image, coordtype="homogeneous", stackaxis=0):
    """Generate pixel coordinate matrix for a 2D image."""
    nr, nc = image.shape
    xgrid, ygrid = np.meshgrid(range(0, nc), range(0, nr), indexing="xy")
    if coordtype == "cartesian":
        return np.stack((xgrid, ygrid), axis=stackaxis)
    elif coordtype == "homogeneous":
        zgrid = np.ones(xgrid.shape)
        return np.stack((xgrid, ygrid, zgrid), axis=stackaxis)


def rotation2D(angle, center=(0, 0), to_rad=True):
    """Rotation matrix in 2D homogeneous coordinates."""
    y, x = center
    if to_rad:
        angle = np.radians(angle)
    sina, cosa = np.sin(angle), np.cos(angle)
    rtx, rty = (1 - cosa) * x - sina * y, sina * x + (1 - cosa) * y
    return np.array([[cosa, sina, rtx], [-sina, cosa, rty], [0, 0, 1]])


def translation2D(xtrans, ytrans):
    """Translation matrix in 2D homogeneous coordinates."""
    return np.array([[0, 0, xtrans], [0, 0, ytrans], [0, 0, 0]]) + np.eye(3)


def scaling2D(xscale, yscale):
    """Biaxial scaling matrix in 2D homogeneous coordinates."""
    return np.array([[1 / xscale, 0, 0], [0, 1 / yscale, 0], [0, 0, 1]])


def compose_deform_field(coordmat, mat_transform, stackaxis=0, ret="deformation", ret_indexing="rc"):
    """Compose deformation/displacement field from coordinate and transform matrices."""
    if (stackaxis != 0) and (stackaxis != -1):
        stackaxis = 0
    coordmat_shape = coordmat.shape
    coord_dim = coordmat_shape[stackaxis]
    ncoords = np.prod(coordmat_shape) // coord_dim

    if stackaxis == 0:
        field = np.dot(mat_transform, coordmat.reshape((coord_dim, ncoords))).reshape(coordmat_shape)
        if ret == "displacement":
            field -= coordmat
        xfield, yfield = field[0, ...], field[1, ...]
    elif stackaxis == -1:
        field = np.dot(mat_transform, coordmat.reshape((ncoords, coord_dim)).T).T.reshape(coordmat_shape)
        if ret == "displacement":
            field -= coordmat
        xfield, yfield = field[..., 0], field[..., 1]

    if ret_indexing == "xy":
        return xfield, yfield
    elif ret_indexing == "rc":
        return yfield, xfield


def rotationDF(coordmat, stackaxis=0, angle=0, center=(0, 0), to_rad=True, **kwds):
    """Deformation field of 2D rotation in image coordinates."""
    rotation_matrix = rotation2D(angle, center, to_rad)
    return compose_deform_field(coordmat, mat_transform=rotation_matrix, stackaxis=stackaxis, **kwds)


def translationDF(coordmat, stackaxis=0, xtrans=0, ytrans=0, **kwds):
    """Deformation field of 2D translation."""
    translation_matrix = translation2D(xtrans=-xtrans, ytrans=-ytrans)
    return compose_deform_field(coordmat, mat_transform=translation_matrix, stackaxis=stackaxis, **kwds)


def rotodeform(imbase, angle, center, interp_order=1, **kwargs):
    """Image rotation using deformation field."""
    coordmat = coordinate_matrix_2D(imbase, coordtype="homogeneous", stackaxis=0)
    rdisp, cdisp = rotationDF(coordmat, stackaxis=0, ret="displacement", center=center, angle=angle)
    rdeform, cdeform = coordmat[1, ...] + rdisp, coordmat[0, ...] + cdisp
    return ndi.map_coordinates(imbase, [rdeform, cdeform], order=interp_order, **kwargs)


def rotosymmetrize(image, center, rotsym=None, angles=None, outside="nan", **kwargs):
    """Symmetrize the pattern according to rotational symmetry.

    Parameters
    ----------
    image : 2D array
        Image to symmetrize.
    center : list/tuple
        Image center pixel position (row, column).
    rotsym : int or None
        Order of rotation symmetry.
    angles : array or None
        Angles of rotation.
    outside : str or numeric
        Value outside the masked boundary.

    Returns
    -------
    rotoavg : 2D array
        Symmetrized image.
    angles : array
        Rotation angles used.
    """
    image = np.nan_to_num(image)
    if rotsym is not None:
        rotsym = int(rotsym)
        angles = np.linspace(0, 360, rotsym, endpoint=False)

    rotoeqs = []
    for angle in angles:
        rotoeqs.append(rotodeform(imbase=image, angle=angle, center=center, **kwargs))
    rotoeqs = np.asarray(rotoeqs)
    rotoavg = rotoeqs.mean(axis=0)

    if outside == "nan":
        rotoavg = to_masked(rotoavg, val=0)
        return rotoavg, angles
    elif outside == 0:
        return rotoavg, angles


def transdeform(imbase, xtrans=0, ytrans=0, interp_order=1, **kwargs):
    """Image translation using deformation field."""
    coordmat = coordinate_matrix_2D(imbase, coordtype="homogeneous", stackaxis=0)
    rdisp, cdisp = translationDF(coordmat, stackaxis=0, ret="displacement", xtrans=xtrans, ytrans=ytrans)
    rdeform, cdeform = coordmat[1, ...] + rdisp, coordmat[0, ...] + cdisp
    return ndi.map_coordinates(imbase, [rdeform, cdeform], order=interp_order, **kwargs)


def sym_band(ind_band, recon, kx, ky, lengthKx, lengthKy):
    """Symmetrize a reconstructed band with respect to kx and ky axes.

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