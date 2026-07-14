"""HDF5 and file I/O utilities extracted from fuller.utils and mpes.fprocessing."""

import glob as g
import numpy as np
import scipy.io as sio
from h5py import File
from scipy.interpolate import RegularGridInterpolator as RGI
from tqdm import tqdm as tqdm_classic
from tqdm import tqdm_notebook

try:
    import natsort as nts
except ImportError:
    nts = None


def tqdmenv(env):
    """Choose tqdm progress bar executing environment.

    Parameters
    ----------
    env : str
        Name of the environment, 'classic' for ordinary environment,
        'notebook' for Jupyter notebook.
    """
    if env == "classic":
        return tqdm_classic
    elif env == "notebook":
        return tqdm_notebook
    return tqdm_classic


def to_masked(arr, val=0):
    """Convert to masked array by setting val to NaN."""
    arrm = arr.copy()
    arrm[arrm == val] = np.nan
    return arrm


def valrange(arr):
    """Output the value range of an array."""
    return arr.min(), arr.max()


def nzbound(arr):
    """Find index bounds of the nonzero elements of a 1D array."""
    arr = np.asarray(arr)
    axis_nz_index = np.argwhere(arr != 0).ravel()
    return axis_nz_index[0], axis_nz_index[-1]


def nonneg_sum_decomposition(absum, a=None, b=None):
    """Nonnegative decomposition of a sum.

    Parameters
    ----------
    ab : numeric
        Sum of the values.
    a, b : numeric or None
        Two numerics for decomposition.

    Returns
    -------
    a, b : numeric
        Nonnegative values of a and b from the decomposition.
    """
    if a is not None:
        a = min(a, absum)
        b = absum - a
        return a, b
    elif b is not None:
        b = min(b, absum)
        a = absum - b
        return a, b
    elif (a is None) and (b is None):
        raise ValueError("At least one of the components should be a numeric.")


def cut_margins(image, margins, offsetx=0, offsety=0):
    """Trim a 2D image by the given margins."""
    offsetx, offsety = int(offsetx), int(offsety)
    yim, xim = image.shape
    t, b, l, r = margins
    if offsetx != 0:
        l, r = l - offsetx, r - offsetx
    if offsety != 0:
        t, b = t - offsety, b - offsety
    return image[t : yim - b, l : xim - r]


def trim_2d_edge(arr, edges, axes=(0, 1)):
    """Trim 2D edges in the first two dimensions of an nD array."""
    edges = np.array(edges)
    trimmed = np.moveaxis(arr, axes, (0, 1))
    if edges.size == 1:
        eg = edges.item()
        trimmed = trimmed[eg:-eg, eg:-eg, ...]
    elif edges.size == 4:
        top, bot, left, rite = edges
        trimmed = trimmed[top:-bot, left:-rite, ...]
    trimmed = np.moveaxis(trimmed, (0, 1), axes)
    return trimmed


def segmod(indices):
    """Add 1 to the intermediate indices."""
    alt_indices = indices + 1
    alt_indices[0] -= 1
    alt_indices[-1] -= 1
    return alt_indices


def fexp(ke, length):
    """Exponential function."""
    return np.exp(-ke * np.arange(0, length, 1))


def pick_operator(fstring, package="numpy"):
    """Return an operator function from the specified package."""
    try:
        exec("import " + package)
        return eval(package + "." + fstring)
    except Exception:
        return fstring


def coeffgen(size, amp=1, distribution="uniform", mask=None, modulation=None, seed=None, **kwargs):
    """Generate random sequence from a distribution modulated by an envelope function and a mask."""
    op_package = kwargs.pop("package", "numpy.random")
    if seed is not None:
        np.random.seed(seed)

    if modulation is not None:
        if modulation == "exp":
            ke = kwargs.pop("ke", 2e-2)
            length = kwargs.pop("length", size[1])
            cfmod = fexp(ke, length)[None, :]
        elif isinstance(modulation, np.ndarray):
            cfmod = modulation
    else:
        cfmod = np.ones(size)

    if mask is not None:
        if mask.ndim == 1:
            cfmask = mask[None, :]
        elif isinstance(mask, np.ndarray):
            cfmask = mask
    else:
        cfmask = np.ones(size)

    opr = pick_operator(distribution, package=op_package)
    cfout = opr(size=size, **kwargs)
    cfout *= amp * cfmask * cfmod
    return cfout


def binarize(cfs, threshold, vals=(0, 1), absolute=True, eq="geq"):
    """Binarize an array by a threshold."""
    arr = np.array(cfs)
    if absolute:
        arr = np.abs(arr)
    if eq == "leq":
        arr[arr <= threshold] = vals[0]
        arr[arr > threshold] = vals[1]
    elif eq == "geq":
        arr[arr < threshold] = vals[0]
        arr[arr >= threshold] = vals[1]
    elif eq is None:
        arr[arr < threshold] = vals[0]
        arr[arr > threshold] = vals[1]
    return arr


def interpolate2d(oldx, oldy, vals, nx=None, ny=None, ret="interpolant", **kwargs):
    """Interpolate values in a newer and/or finer grid.

    Parameters
    ----------
    oldx, oldy : 1D array
        Values of the old x and y axes.
    vals : 2D array
        Image pixel values associated with the old x and y axes.
    nx, ny : int or None
        Number of elements in the interpolated axes.
    ret : str
        'interpolant' returns (vals_interp, interpolator),
        'all' returns (vals_interp, interpolator, meshgrid).

    Returns
    -------
    Depending on ret.
    """
    newx = kwargs.pop("newx", np.linspace(oldx.min(), oldx.max(), nx, endpoint=True))
    newy = kwargs.pop("newy", np.linspace(oldy.min(), oldy.max(), ny, endpoint=True))
    newxymesh = np.meshgrid(newx, newy, indexing="ij")
    newxy = np.stack(newxymesh, axis=-1).reshape((nx * ny, 2))
    vip = RGI((oldx, oldy), vals)
    vals_interp = vip(newxy).reshape((nx, ny))
    if ret == "interpolant":
        return vals_interp, vip
    elif ret == "all":
        return vals_interp, vip, newxymesh


def findFiles(fdir, fstring="", ftype="h5", **kwds):
    """Retrieve files named in a similar way from a folder."""
    if nts is None:
        raise ImportError("natsort is required for findFiles. Install with: pip install natsort")
    files = nts.natsorted(g.glob(fdir + fstring + "." + ftype), **kwds)
    return files


# --- HDF5 I/O ---


def saveHDF(*groups, save_addr="./file.h5", track_order=True, **kwds):
    """Combine dictionaries and save into a hierarchical HDF5 structure.

    Parameters
    ----------
    groups : list of [group_name, group_dict]
        Each group specified as ['folder_name', folder_dict].
    save_addr : str
        File path for saving.
    """
    try:
        hdf = File(save_addr, "w")
        for g in groups:
            grp = hdf.create_group(g[0], track_order=track_order)
            for gk, gv in g[1].items():
                grp.create_dataset(gk, data=gv, **kwds)
    finally:
        hdf.close()


def loadHDF(load_addr, hierarchy="flat", groups="all", track_order=True, dtyp="float", **kwds):
    """Load contents in an HDF5 file.

    Parameters
    ----------
    load_addr : str
        Address of the file to load.
    hierarchy : str
        'flat' to flatten all groups into a single dict.
    groups : list or 'all'
        Name of the groups to load.
    dtyp : str
        Data type to be loaded into.

    Returns
    -------
    outdict : dict
        Dictionary containing the hierarchical contents of the file.
    """
    outdict = {}
    if hierarchy == "flat":
        with File(load_addr, track_order=track_order, **kwds) as f:
            if groups == "all":
                groups = list(f)
            for g in groups:
                for gk, gv in f[g].items():
                    outdict[gk] = np.asarray(gv, dtype=dtyp)
    else:
        outdict = {}
        with File(load_addr, track_order=track_order, **kwds) as f:
            if groups == "all":
                groups = list(f)
            for g in groups:
                outdict[g] = {}
                for gk, gv in f[g].items():
                    outdict[g][gk] = np.asarray(gv, dtype=dtyp)
    return outdict


def loadH5Parts(filename, content, outtype="dict", alias=None):
    """Load specified content from a single complex HDF5 file."""
    with File(filename) as f:
        if alias is None:
            outdict = {k: np.array(f[k]) for k in content}
        else:
            if len(content) != len(alias):
                raise ValueError("Not every content entry is assigned an alias!")
            outdict = {ka: np.array(f[k]) for k in content for ka in alias}
    if outtype == "dict":
        return outdict
    elif outtype == "list":
        return list(outdict.items())
    elif outtype == "vals":
        return list(outdict.values())


def load_bandstruct(path, form, varnames=None):
    """Load band structure information from file.

    Parameters
    ----------
    path : str
        File path to load from.
    form : str
        Format of the file ('mat', 'h5', 'hdf5').
    varnames : list or None
        Names of the variables to load. Defaults to ['bands', 'kxx', 'kyy'].

    Returns
    -------
    list of arrays
    """
    if varnames is None or len(varnames) == 0:
        varnames = ["bands", "kxx", "kyy"]
    if form == "mat":
        mat = sio.loadmat(path)
        return [mat[vn] for vn in varnames]
    elif form in ("h5", "hdf5"):
        dct = loadHDF(path, hierarchy="flat", groups=varnames)
        return [dct[vn] for vn in varnames]


def load_multiple_bands(folder, ename="", kname="", form="h5", dtyp="float", **kwargs):
    """Custom loader for multiple reconstructed bands."""
    if nts is None:
        raise ImportError("natsort is required. Install with: pip install natsort")
    if form in ("h5", "hdf5"):
        files = nts.natsorted(g.glob(f"{folder}/*.h5"))
    else:
        files = nts.natsorted(g.glob(f"{folder}/*.{form}"))

    econtents = []
    for f in files:
        f_inst = File(f, **kwargs)
        econtent = np.array(f_inst[ename], dtype=dtyp)
        econtents.append(econtent)
    econtents = np.asarray(econtents)

    kcontents = []
    with File(files[0], **kwargs) as f_instance:
        kgroups = list(f_instance[kname])
        for kg in kgroups:
            kcontents.append(np.asarray(f_instance[kname][kg], dtype=dtyp))
    return econtents, kcontents


def load_calculation(path, nkx=120, nky=55, delim=" ", drop_pos=2, drop_axis=1, baxis=None, maxid=None):
    """Read and reshape energy band calculation results.

    Parameters
    ----------
    path : str
        File path where the calculation output file is located.
    nkx, nky : int
        Number of k points sampled along the kx and ky directions.
    delim : str
        Delimiter used for reading the calculation output file.
    drop_pos, drop_axis : int
        The position and axis along which to drop the elements.
    baxis : int or None
        Axis of the energy band index.
    maxid : int or None
        Maximum limiting index of the read array.

    Returns
    -------
    ebands : 3D array
        Collection of energy bands indexed by their energies.
    """
    nkx, nky = int(nkx), int(nky)
    nk = nkx * nky
    arr = np.fromfile(path, sep=delim)
    neb = int(arr.size / nk)
    if maxid is None:
        ebands = arr[: nk * neb].reshape((nk, neb))
    else:
        maxid = int(maxid)
        ebands = arr[:maxid].reshape((nk, neb))
    if drop_axis is not None:
        ebands = np.delete(ebands, drop_pos, axis=drop_axis).reshape((nky, nkx, neb - 1))
    if baxis is not None:
        baxis = int(baxis)
        ebands = np.moveaxis(ebands, 2, baxis)
    return ebands


def readBinnedhdf5(fpath, combined=True, typ="float32"):
    """Read binned hdf5 file (3D/4D data) into a dictionary.

    Extracted from mpes.fprocessing.

    Parameters
    ----------
    fpath : str
        File path.
    combined : bool
        If True and multiple binned slices, combine into a single 'V' array.
    typ : str
        Data type of the numerical values.

    Returns
    -------
    out : dict
        Dictionary with keys being the axes and the volume (slices).
    """
    f = File(fpath, "r")
    out = {}
    for ax, axval in f["axes"].items():
        out[ax] = axval[...]
    group = f["binned"]
    itemkeys = list(group.keys())
    nbinned = len(itemkeys)
    if (nbinned == 1) or (combined is False):
        for ik in itemkeys:
            out[ik] = np.asarray(group[ik], dtype=typ)
    elif (nbinned > 1) or combined:
        val = []
        if nts is not None:
            itemkeys_sorted = nts.natsorted(itemkeys)
        else:
            itemkeys_sorted = sorted(itemkeys)
        for ik in itemkeys_sorted:
            val.append(group[ik])
        out["V"] = np.asarray(val, dtype=typ)
    return out