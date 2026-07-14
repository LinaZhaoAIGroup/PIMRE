"""Momentum distortion correction for finding high-symmetry points.

Simplified MomentumCorrector extracted from mpes.analysis, with minimal
embedded pointops functions (maxlist method only, no astropy/photutils).
"""

import numpy as np
from numpy.linalg import norm
from skimage.feature import peak_local_max


# --- Minimal pointops (embedded to avoid symmetrize dependency) ---


def _peakdetect2d(img, method="maxlist", **kwds):
    """Peak detection in 2D image. Only 'maxlist' method supported."""
    mindist = kwds.pop("mindist", 10)
    numpeaks = kwds.pop("numpeaks", 7)
    return peak_local_max(img, min_distance=mindist, num_peaks=numpeaks, **kwds)


def _pointset_center(pset, method="centroidnn", ret="cnc"):
    """Determine the center position of a point set."""
    pmean = np.mean(pset, axis=0)
    if method == "centroidnn":
        dist = norm(pset - pmean, axis=1)
        minid = np.argmin(dist)
        pscenter = pset[minid, :]
        prest = np.delete(pset, minid, axis=0)
    elif method == "centroid":
        pscenter = pmean
        prest = pset
    else:
        raise NotImplementedError
    if ret == "cnc":
        return pscenter, prest
    elif ret == "all":
        return pscenter, prest, pmean


def _pointset_order(pset, center=None, direction="ccw"):
    """Order a point set around a center."""
    dirdict = {"cw": 1, "ccw": -1}
    if center is None:
        pmean = np.mean(pset, axis=0)
        pshifted = pset - pmean
    else:
        pshifted = pset - center
    pangle = np.arctan2(pshifted[:, 1], pshifted[:, 0]) * 180 / np.pi
    order = np.argsort(pangle)[:: dirdict[direction]]
    return pset[order]


def _cvdist(verts, center):
    """Calculate center-vertex distances."""
    return norm(verts - center, axis=1)


def _vvdist(verts, neighbor=1):
    """Calculate neighboring vertex-vertex distances."""
    if neighbor == 1:
        return norm(verts - np.roll(verts, shift=-1, axis=0), axis=1)
    return None


def _polyarea(coords, coord_order="rc"):
    """Calculate polygon area using surveyor's formula."""
    if coord_order in ("rc", "yx"):
        y, x = zip(*coords)
    elif coord_order in ("cr", "xy"):
        x, y = zip(*coords)
    A = abs(sum(i * j for i, j in zip(x, y[1:] + y[:1])) - sum(i * j for i, j in zip(x[1:] + x[:1], y))) / 2
    return A


def _reorder(points, itemid, axis=0):
    """Reorder a point set along an axis."""
    return np.roll(points, shift=itemid - 1, axis=axis)


def _rotmat(theta, to_rad=True, coordsys="cartesian"):
    """Rotation matrix in 2D."""
    if to_rad:
        theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    if coordsys == "cartesian":
        return np.array([[c, -s], [s, c]])
    elif coordsys == "homogen":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _csm(pcent, pvert, rotsym=None, symtype="rotation"):
    """Continuous (a)symmetry measure for a set of polygon vertices."""
    if symtype != "rotation":
        return 0.0
    npts = len(pvert)
    cvd = _cvdist(pvert, pcent)
    maxind = np.argmax(cvd)
    maxlen = cvd[maxind]
    cvdnorm = cvd / maxlen
    pts_reord = _reorder(pvert, maxind, axis=0)
    mcv = cvdnorm.mean()
    rotangles = 360 * (np.linspace(1, rotsym, rotsym) - 1) / rotsym
    xvec = pts_reord[0, :] - pcent
    xvec /= norm(xvec)
    devangles = [0.0]
    for p, rota in zip(pts_reord[1:], rotangles[1:]):
        R = _rotmat(rota, to_rad=True)
        rotv = np.dot(R, (p - pcent).T)
        devangles.append(np.arccos(np.sum(rotv * xvec) / norm(rotv)))
    devangles = np.array(devangles)
    mang = devangles.mean()
    dpq = mcv**2 + cvdnorm**2 - 2 * mcv * cvdnorm * np.cos(devangles - mang)
    s = dpq.sum() / npts
    return s


def _coordinate_matrix_2D(image, coordtype="cartesian", stackaxis=0):
    """Generate pixel coordinate matrix for a 2D image."""
    nr, nc = image.shape
    xgrid, ygrid = np.meshgrid(range(0, nc), range(0, nr), indexing="xy")
    if coordtype == "cartesian":
        return np.stack((xgrid, ygrid), axis=stackaxis)
    elif coordtype == "homogeneous":
        zgrid = np.ones(xgrid.shape)
        return np.stack((xgrid, ygrid, zgrid), axis=stackaxis)


def _applyWarping(imgstack, axis, hgmat):
    """Apply warping transform for a stack of images along an axis."""
    try:
        import cv2
    except ImportError:
        raise ImportError("cv2 (opencv-python) is required for applyWarping.")
    imgstack = np.moveaxis(imgstack, axis, 0)
    imgstack_transformed = np.zeros_like(imgstack)
    nimg = imgstack.shape[0]
    for i in range(nimg):
        imgstack_transformed[i, ...] = cv2.warpPerspective(imgstack[i, ...], hgmat, imgstack[i, ...].shape)
    imgstack_transformed = np.moveaxis(imgstack_transformed, 0, axis)
    return imgstack_transformed


# --- Core class ---


class MomentumCorrector:
    """Momentum distortion correction and symmetry-based point finding.

    Simplified version of mpes.analysis.MomentumCorrector.
    Uses 'maxlist' peak detection only (skimage), no astropy/photutils.
    """

    def __init__(self, image, rotsym=6):
        self.image = np.squeeze(image)
        self.imgndim = image.ndim
        if self.imgndim > 3 or self.imgndim < 2:
            raise ValueError("The input image dimension need to be 2 or 3!")
        if self.imgndim == 2:
            self.slice = self.image
        self.rotsym = int(rotsym)
        self.pouter_ord = None
        self.pcent = None

    def selectSlice2D(self, selector, axis=0):
        """Select a 2D slice from a 3D volume."""
        if self.imgndim > 2:
            im = np.moveaxis(self.image, axis, 0)
            try:
                self.slice = im[selector, ...].sum(axis=0)
            except Exception:
                self.slice = im[selector, ...]
        elif self.imgndim == 2:
            raise ValueError("Input image dimension is already 2!")

    def featureExtract(self, image, direction="ccw", symscores=False, **kwds):
        """Extract point features from a 2D slice.

        Parameters
        ----------
        image : 2D array
            Image slice to extract features from.
        direction : str
            'ccw' or 'cw' ordering direction.
        symscores : bool
            Whether to calculate symmetry scores.
        **kwds : passed to peak_local_max (mindist, numpeaks, etc.)
        """
        self.resetDeformation(image=image, coordtype="cartesian")

        method = kwds.pop("method", "maxlist")
        center_det = kwds.pop("center_det", "centroidnn")

        self.peaks = _peakdetect2d(image, method=method, **kwds)
        if center_det is None:
            self.pouter = self.peaks
            self.pcent = None
        else:
            self.pcent, self.pouter = _pointset_center(self.peaks, method=center_det, ret="cnc")
            self.pcent = tuple(self.pcent)

        self.pouter_ord = _pointset_order(self.pouter, direction=direction)

        try:
            self.area_old = _polyarea(coords=self.pouter_ord, coord_order="rc")
        except Exception:
            pass

        self.calcGeometricDistances()

        if symscores:
            self.csm_original = self.calcSymmetryScores(symtype="rotation")

        if self.rotsym == 6:
            self.mdist = (self.mcvdist + self.mvvdist) / 2
            self.mcvdist = self.mdist
            self.mvvdist = self.mdist

    def calcGeometricDistances(self):
        """Calculate geometric distances involving center and vertices."""
        self.cvdist = _cvdist(self.pouter_ord, self.pcent)
        self.mcvdist = self.cvdist.mean()
        self.vvdist = _vvdist(self.pouter_ord)
        self.mvvdist = self.vvdist.mean()

    def calcSymmetryScores(self, symtype="rotation"):
        """Calculate symmetry scores."""
        return _csm(self.pcent, self.pouter_ord, rotsym=self.rotsym, symtype=symtype)

    def resetDeformation(self, **kwds):
        """Reset the deformation field."""
        image = kwds.pop("image", self.slice)
        coordtype = kwds.pop("coordtype", "cartesian")
        coordmat = _coordinate_matrix_2D(image, coordtype=coordtype, stackaxis=0).astype("float64")
        self.rdeform_field = coordmat[1, ...]
        self.cdeform_field = coordmat[0, ...]


def find_MM(ddata):
    """Find M points from experimental data using MomentumCorrector.

    Parameters
    ----------
    ddata : 3D array
        Experimental intensity data (E, kx, ky).

    Returns
    -------
    pouter_ord : 2D array
        Ordered coordinates of detected M points.
    """
    mc = MomentumCorrector(ddata, rotsym=6)
    mc.selectSlice2D(selector=slice(90, 100), axis=0)
    mc.featureExtract(image=mc.slice, method="maxlist", symscores=False, mindist=80)
    return mc.pouter_ord