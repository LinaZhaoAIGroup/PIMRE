"""High-symmetry k-path utilities: symmetry points, band path maps.

Extracted from mpes.analysis, ArpesBandRecons.CoordTrans, and 4.mrf.ipynb.
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator


def line_generator(A, B, npoints, endpoint=True, ret="separated"):
    """Generate intermediate points in a line segment AB given endpoints.

    Parameters
    ----------
    A, B : tuple/list
        Pixel coordinates of the endpoints.
    npoints : int
        Number of points in the line segment.
    endpoint : bool
        Option to include the endpoint (B) in the line coordinates.
    ret : str
        'separated' or 'joined'.

    Returns
    -------
    Point coordinates.
    """
    ndim = len(A)
    npoints = int(npoints)
    points = []
    for i in range(ndim):
        points.append(np.linspace(A[i], B[i], npoints, endpoint=endpoint))
    point_coords = np.asarray(points).T
    if ret == "separated":
        return np.split(point_coords, ndim, axis=1)
    elif ret == "joined":
        return point_coords


def image_interpolator(image, iptype="RGI", interp_method="linear"):
    """Construct an image interpolator."""
    dims = image.shape
    dimaxes = [list(range(d)) for d in dims]
    if iptype == "RGI":
        return RegularGridInterpolator(dimaxes, image, method=interp_method)
    raise NotImplementedError


def interp_slice(data, pathr=None, pathc=None, path_coords=None, iptype="RGI",
                 interp_method="linear"):
    """Slice 2D/3D data through interpolation along a pixel path.

    The interpolation is performed per layer with
    ``scipy.ndimage.map_coordinates`` (spline orders: nearest=0,
    linear=1, cubic=3) and ``prefilter=False`` (local spline kernel).
    This matches the linear regular-grid interpolation of the previous
    implementation, supports 'cubic' even when a dimension has fewer
    than 4 points (e.g. a single reconstructed band), is much faster
    than the 3D RGI cubic, and keeps NaN localized instead of letting
    the spline prefilter spread them over the whole array (which would
    erase the reconstructed curves where the band leaves the measured
    energy window).
    """
    order = {"nearest": 0, "linear": 1, "cubic": 3}.get(interp_method, 1)
    if path_coords is not None:
        pathr = path_coords[:, 0]
        pathc = path_coords[:, 1]
    pathr = np.ravel(pathr)
    pathc = np.ravel(pathc)

    if data.ndim == 2:
        return ndimage.map_coordinates(data, [pathr, pathc], order=order,
                                       mode="nearest", prefilter=False)
    if data.ndim == 3:
        nstack = data.shape[-1]
        parts = [
            ndimage.map_coordinates(data[..., i], [pathr, pathc],
                                    order=order, mode="nearest",
                                    prefilter=False)
            for i in range(nstack)
        ]
        return np.concatenate(parts)
    raise ValueError(f"interp_slice supports 2D/3D data, got ndim={data.ndim}")


def points2path(pointsr, pointsc, method="analog", npoints=None, ret="separated"):
    """Calculate ordered pixel coordinates along a path defined by intermediate points.

    Parameters
    ----------
    pointsr, pointsc : list/tuple/array
        Row and column pixel coordinates of special points along the path.
    method : str
        'analog' or 'digital'.
    npoints : list/tuple or None
        Number of points along each segment.
    ret : str
        'separated' or 'combined'.

    Returns
    -------
    polyr, polyc : 1D array
        Pixel coordinates along the path.
    pid : 1D array
        Pointwise indices of the special points.
    """
    pointsr = np.round(pointsr).astype("int")
    pointsc = np.round(pointsc).astype("int")
    npts = len(pointsr)

    polyr, polyc, pid = [], [], np.zeros((npts,), dtype="int")

    for i in range(npts - 1):
        if method == "digital":
            from skimage.draw import line as skline

            lsegr, lsegc = skline(pointsr[i], pointsc[i], pointsr[i + 1], pointsc[i + 1])
        elif method == "analog":
            lsegr, lsegc = line_generator(
                [pointsr[i], pointsc[i]],
                [pointsr[i + 1], pointsc[i + 1]],
                npoints=npoints[i],
                endpoint=True,
                ret="separated",
            )
        if i < npts - 2:
            lsegr, lsegc = lsegr[:-1], lsegc[:-1]
        polyr.append(lsegr)
        polyc.append(lsegc)
        pid[i + 1] = len(lsegr) + pid.max()

    polyr, polyc = map(np.concatenate, (polyr, polyc))
    if ret == "combined":
        return np.stack((polyr, polyc), axis=1), pid
    elif ret == "separated":
        return polyr, polyc, pid


def bandpath_map(bsvol, pathr=None, pathc=None, path_coords=None, eaxis=2,
                 method="analog", interp_method="linear"):
    """Extract band diagram map from 2D/3D data.

    Parameters
    ----------
    bsvol : 2D/3D array
        Volumetric band structure data.
    pathr, pathc : 1D array or None
        Row and column pixel coordinates along the band path.
    path_coords : 2D array or None
        Combined row and column coordinates.
    eaxis : int
        Energy axis index.
    method : str
        'analog' or 'digital'.
    interp_method : str
        Interpolation order for the analog path ('linear', 'cubic',
        'nearest'); passed to RegularGridInterpolator.

    Returns
    -------
    bpm : 2D array
        Band path map.
    """
    edim = bsvol.shape[eaxis]
    bsvol = np.moveaxis(bsvol, eaxis, 2)

    if method == "digital":
        if path_coords is not None:
            axid = np.where(np.array(path_coords.shape) == 2)[0][0]
            pathr, pathc = np.split(path_coords, 2, axis=axid)
            pathr, pathc = map(np.ravel, [pathr, pathc])
        bpm = bsvol[pathr, pathc, :]
    elif method == "analog":
        bpm = interp_slice(bsvol, pathr=pathr, pathc=pathc,
                           path_coords=path_coords, interp_method=interp_method)
        bpm = bpm.reshape((edim, bpm.size // edim))
    return bpm
