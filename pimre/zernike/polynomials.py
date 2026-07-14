"""Zernike and Hexike orthonormal polynomials.

Extracted from poppy.zernike, adapted to work with pure numpy (no astropy, no pyFFTW, no OpenCL).
"""

import warnings
import numpy as np
from math import factorial


def _is_odd(integer):
    """Test if an integer is odd by bitwise & with 1."""
    return integer & 1


def zern_name(i):
    """Return a human-readable text name for Zernike term j."""
    names = [
        "Null", "Piston", "Tilt X", "Tilt Y",
        "Focus", "Astigmatism 45", "Astigmatism 0",
        "Coma Y", "Coma X",
        "Trefoil Y", "Trefoil X",
        "Spherical", "2nd Astig 0", "2nd Astig 45",
        "Tetrafoil 0", "Tetrafoil 22.5",
        "2nd coma X", "2nd coma Y", "3rd Astig X", "3rd Astig Y",
        "Pentafoil X", "Pentafoil Y", "5th order spherical",
    ]
    if i < len(names):
        return names[i]
    return "Z%d" % i


def str_zernike(n, m):
    """Return analytic expression for a given Zernike in LaTeX syntax."""
    signed_m = int(m)
    m = int(np.abs(m))
    n = int(np.abs(n))

    terms = []
    for k in range(int((n - m) / 2) + 1):
        coef = ((-1) ** k * factorial(n - k) /
                (factorial(k) * factorial(int((n + m) / 2) - k) * factorial(int((n - m) / 2) - k)))
        if coef != 0:
            formatcode = "{0:d}" if k == 0 else "{0:+d}"
            terms.append((formatcode + " r^{1:d} ").format(int(coef), n - 2 * k))

    outstr = " ".join(terms)
    if m == 0:
        if n == 0:
            return "1"
        return "sqrt(%d)* ( %s ) " % (n + 1, outstr)
    elif signed_m > 0:
        return "\\sqrt{%d}* ( %s ) * \\cos(%d \\theta)" % (2 * (n + 1), outstr, m)
    else:
        return "\\sqrt{%d}* ( %s ) * \\sin(%d \\theta)" % (2 * (n + 1), outstr, m)


def noll_indices(j):
    """Convert from 1-D to 2-D indexing for Zernikes or Hexikes.

    Parameters
    ----------
    j : int
        Zernike function ordinate, following the convention of Noll et al. JOSA 1976.
        Starts at 1.

    Returns
    -------
    n, m : int
        Zernike polynomial degree and azimuthal order.
    """
    if j < 1:
        raise ValueError("Zernike index j must be a positive integer.")

    n = int(np.ceil((-1 + np.sqrt(1 + 8 * j)) / 2) - 1)
    if n == 0:
        m = 0
    else:
        nprev = (n + 1) * (n + 2) / 2
        resid = int(j - nprev - 1)

        if _is_odd(j):
            sign = -1
        else:
            sign = 1

        if _is_odd(n):
            row_m = [1, 1]
        else:
            row_m = [0]

        for i in range(int(np.floor(n / 2.0))):
            row_m.append(row_m[-1] + 2)
            row_m.append(row_m[-1])

        m = row_m[resid] * sign

    return n, m


def R(n, m, rho):
    """Compute R[n, m], the Zernike radial polynomial.

    Parameters
    ----------
    n, m : int
        Zernike function degree.
    rho : array
        Image plane radial coordinates. rho should be 1 at the edge of the unit circle.

    Returns
    -------
    output : array
        Radial polynomial values.
    """
    m = int(np.abs(m))
    n = int(np.abs(n))
    output = np.zeros(rho.shape)
    if _is_odd(n - m):
        return 0
    for k in range(int((n - m) / 2) + 1):
        coef = ((-1) ** k * factorial(n - k) /
                (factorial(k) * factorial(int((n + m) / 2) - k) * factorial(int((n - m) / 2) - k)))
        output += coef * rho ** (n - 2 * k)
    return output


def zernike(n, m, npix=100, rho=None, theta=None, outside=np.nan, noll_normalize=True):
    """Return the Zernike polynomial Z[m,n] for a given pupil.

    Parameters
    ----------
    n, m : int
        Zernike function degree.
    npix : int
        Desired diameter for circular pupil. Only used if rho and theta are not provided.
    rho, theta : array_like
        Image plane coordinates. rho should be 0 at the origin and 1.0 at the edge.
        theta should be the angle in radians.
    outside : float
        Value for pixels outside the circular aperture (rho > 1). Default is np.nan.
    noll_normalize : bool
        If True, normalize such that the integral of Z[n,m]*Z[n,m] over the unit disk is pi.

    Returns
    -------
    zern : 2D numpy array
        Z(m,n) evaluated at each (rho, theta).
    """
    if not n >= m:
        raise ValueError("Zernike index m must be >= index n")
    if (n - m) % 2 != 0:
        warnings.warn(
            "Radial polynomial is zero for these inputs: m={}, n={} (are you sure you wanted this Zernike?)".format(m, n)
        )

    if theta is None and rho is None:
        x = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.0) / ((npix - 1) / 2.0)
        y = x
        xx, yy = np.meshgrid(x, y)
        rho = np.sqrt(xx**2 + yy**2)
        theta = np.arctan2(yy, xx)
    elif (theta is None and rho is not None) or (theta is not None and rho is None):
        raise ValueError("If you provide either the `theta` or `rho` input array, you must provide both of them.")

    if rho.shape != theta.shape:
        raise ValueError("The rho and theta arrays do not have consistent shape.")

    aperture = (rho <= 1)

    if m == 0:
        if n == 0:
            zernike_result = aperture.astype(float)
        else:
            norm_coeff = np.sqrt(n + 1) if noll_normalize else 1
            zernike_result = norm_coeff * R(n, m, rho) * aperture
    elif m > 0:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        zernike_result = norm_coeff * R(n, m, rho) * np.cos(np.abs(m) * theta) * aperture
    else:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        zernike_result = norm_coeff * R(n, m, rho) * np.sin(np.abs(m) * theta) * aperture

    zernike_result[rho > 1] = outside
    return zernike_result


def zernike1(j, **kwargs):
    """Return the Zernike polynomial Z_j for pupil points.

    Parameters
    ----------
    j : int
        Zernike function ordinate, following the convention of Noll et al. JOSA 1976.

    Other parameters are passed through to `zernike`.

    Returns
    -------
    zern : 2D numpy array
        Z_j evaluated at each (rho, theta).
    """
    n, m = noll_indices(j)
    return zernike(n, m, **kwargs)


def zernike_basis(nterms=15, npix=512, rho=None, theta=None, outside=np.nan, **kwargs):
    """Return a cube of Zernike terms from 1 to N each as a 2D array.

    Parameters
    ----------
    nterms : int
        Number of Zernike terms to return, starting from piston. Default is 15.
    npix : int
        Desired pixel diameter for circular pupil.
    rho, theta : array_like, optional
        Image plane coordinates.
    outside : float
        Value for pixels outside the circular aperture.
    """
    if rho is not None and theta is not None:
        shape = rho.shape
        use_polar = True
    elif (theta is None and rho is not None) or (theta is not None and rho is None):
        raise ValueError("If you provide either the `theta` or `rho` input array, you must provide both of them.")
    else:
        shape = (npix, npix)
        use_polar = False

    zern_output = np.zeros((nterms,) + shape)
    if use_polar:
        for j in range(nterms):
            zern_output[j] = zernike1(j + 1, rho=rho, theta=theta, outside=outside, **kwargs)
    else:
        for j in range(nterms):
            zern_output[j] = zernike1(j + 1, npix=npix, outside=outside, **kwargs)
    return zern_output


def hex_aperture(npix=1024, rho=None, theta=None, vertical=False, outside=0):
    """Return an aperture function for a hexagon.

    The flat sides are aligned with the X direction by default.

    Parameters
    ----------
    npix : int
        Size in pixels of the aperture array.
    rho, theta : 2D numpy arrays, optional
        Polar coordinates. The hexagon is defined such that it can be circumscribed
        in a rho=1 circle.
    vertical : bool
        Make flat sides parallel to the Y axis instead of the default X.
    outside : float
        Value for pixels outside the hexagonal aperture.
    """
    if rho is not None and theta is not None:
        if rho is None or theta is None:
            raise ValueError("If you provide either the `theta` or `rho` input array, you must provide both of them.")
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
    else:
        x_ = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.0) / (npix / 2.0)
        x, y = np.meshgrid(x_, x_)

    absy = np.abs(y)
    aperture = np.full(x.shape, outside)
    w_rect = (np.abs(x) <= 0.5) & (absy <= np.sqrt(3) / 2)
    w_left_tri = (x <= -0.5) & (x >= -1) & (absy <= (x + 1) * np.sqrt(3))
    w_right_tri = (x >= 0.5) & (x <= 1) & (absy <= (1 - x) * np.sqrt(3))
    aperture[w_rect] = 1.0
    aperture[w_left_tri] = 1.0
    aperture[w_right_tri] = 1.0

    if vertical:
        return aperture.transpose()
    return aperture


def hexike_basis(nterms=15, npix=512, rho=None, theta=None, aperture=None, vertical=False, outside=np.nan):
    """Return a list of hexike polynomials 1-N following Mahajan and Dai 2006.

    Constructed via Gram-Schmidt orthonormalization starting from Zernike polynomials.

    Parameters
    ----------
    nterms : int
        Number of hexike terms to compute, starting from piston. Default is 15.
    npix : int
        Size in pixels of the aperture array.
    rho, theta : 2D numpy arrays, optional
        Polar coordinates.
    aperture : 2D numpy array, optional
        Aperture mask. If not set, inferred from hex_aperture.
    vertical : bool
        Make flat sides parallel to the Y axis. Default is False.
    outside : float
        Value for pixels outside the hexagonal aperture. Default is np.nan.

    Returns
    -------
    basis : 3D numpy array
        Array of shape (nterms, npix, npix) with hexike polynomials.
    """
    if rho is not None:
        shape = rho.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("Only square rho and theta arrays supported")
    else:
        shape = (npix, npix)

    if aperture is None:
        aperture = hex_aperture(npix=npix, rho=rho, theta=theta, vertical=vertical, outside=0)

    apmask = np.isfinite(aperture) & (aperture > 0)
    apmask_float = np.asarray(apmask, float)
    A = apmask.sum()

    Z = np.full((nterms + 1,) + shape, outside, dtype=float)
    Z[1:] = zernike_basis(nterms=nterms, npix=npix, rho=rho, theta=theta, outside=0.0)

    G = [np.zeros(shape), np.ones(shape)]
    H = [np.zeros(shape), apmask_float.copy()]

    for j in np.arange(nterms - 1) + 1:
        nextG = Z[j + 1] * apmask_float
        for k in np.arange(j) + 1:
            coef = -1 / A * (Z[j + 1] * H[k] * apmask_float).sum()
            if coef != 0:
                nextG += coef * H[k]

        nextH = nextG / np.sqrt((nextG**2).sum() / A)
        G.append(nextG)
        H.append(nextH)

    basis = np.asarray(H[1:])
    basis[:, ~apmask] = outside
    return basis


def arbitrary_basis(aperture, nterms=15, rho=None, theta=None, outside=np.nan):
    """Orthonormal basis on arbitrary aperture, via Gram-Schmidt from Zernikes.

    Parameters
    ----------
    aperture : 2D array_like
        Aperture mask. Nonzero/finite values are inside the aperture.
    nterms : int
        Number of terms. Default is 15.
    rho, theta : array_like, optional
        Polar coordinates.
    outside : float
        Value for pixels outside the aperture. Default is np.nan.

    Returns
    -------
    basis : 3D numpy array
    """
    shape = aperture.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Only square aperture arrays are supported")

    apmask = np.isfinite(np.asarray(aperture)) & (np.asarray(aperture) > 0)
    apmask_float = np.asarray(apmask, float)
    A = apmask.sum()

    if theta is None and rho is None:
        yind, xind = np.where(apmask)
        distance = np.sqrt((yind - (shape[0] - 1) / 2.0) ** 2 + (xind - (shape[1] - 1) / 2.0) ** 2)
        max_extent = distance.max()

        ceil = lambda x: int(np.ceil(x)) if x > 0 else 0
        padding = (ceil(max_extent - (shape[0] - 1) / 2.0), ceil(max_extent - (shape[1] - 1) / 2.0))
        padded_shape = (shape[0] + padding[0] * 2, shape[1] + padding[1] * 2)
        npix = padded_shape[0]

        Z = np.zeros((nterms + 1,) + padded_shape)
        Z[1:] = zernike_basis(nterms=nterms, npix=npix, rho=rho, theta=theta, outside=0.0)
        Z = Z[:, padding[0] : padded_shape[0] - padding[0], padding[1] : padded_shape[1] - padding[1]]
    else:
        Z = np.zeros((nterms + 1,) + shape)
        Z[1:] = zernike_basis(nterms=nterms, rho=rho, theta=theta, outside=0.0)

    G = [np.zeros(shape), np.ones(shape)]
    H = [np.zeros(shape), apmask_float.copy()]

    for j in np.arange(nterms - 1) + 1:
        nextG = Z[j + 1] * apmask_float
        for k in np.arange(j) + 1:
            coef = -1 / A * (Z[j + 1] * H[k] * apmask_float).sum()
            if coef != 0:
                nextG += coef * H[k]

        nextH = nextG / np.sqrt((nextG**2).sum() / A)
        G.append(nextG)
        H.append(nextH)

    basis = np.asarray(H[1:])
    basis[:, ~apmask] = outside
    return basis