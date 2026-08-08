"""HDF5 loading utilities extracted from fuller.utils and mpes.fprocessing."""

import numpy as np
from h5py import File


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
