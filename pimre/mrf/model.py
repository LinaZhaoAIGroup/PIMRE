"""MRF band reconstruction: MrfRec model with PyTorch backend.

Extracted from fuller.mrfRec, with TensorFlow replaced by PyTorch.
"""

import contextlib
import warnings as wn

import numpy as np
import torch
from scipy import interpolate
from scipy import io
from scipy import ndimage
from tqdm import tqdm


class MrfRec:
    """Markov Random Field band structure reconstruction.

    Parameters
    ----------
    E : 1D array
        Energy axis as numpy array.
    kx : 1D array or None
        Momentum along x axis.
    ky : 1D array or None
        Momentum along y axis.
    I : 3D array or None
        Measured intensity (kx, ky, E).
    E0 : numeric or None
        Initial guess for band structure energy values.
    eta : float
        Standard deviation of neighbor interaction term.
    includeCurv : bool
        If True, curvature term is included during optimization.
    etaCurv : float
        Standard deviation of curvature term.
    """

    def __init__(self, E, kx=None, ky=None, I=None, E0=None, eta=0.1, includeCurv=False, etaCurv=0.1):
        if kx is None and ky is None:
            raise Exception("Either kx or ky need to be specified!")
        elif kx is None:
            kx = np.array([0.0])
        elif ky is None:
            ky = np.array([0.0])

        self.kx = kx.copy()
        self.ky = ky.copy()
        self.E = E.copy()
        self.lengthKx = kx.size
        self.lengthKy = ky.size
        self.lengthE = E.size
        self.I = np.nan_to_num(I, nan=0.0)
        self.I -= np.min(self.I)
        if np.any(self.I > 0):
            self.I += np.min(self.I[self.I > 0])
        self.eta = eta
        self.includeCurv = includeCurv
        self.etaCurv = etaCurv

        if I is None:
            self.generateI()

        if E0 is None:
            self.indEb = np.ones((self.lengthKx, self.lengthKy), int) * int(self.lengthE / 2)
        else:
            EE, EE0 = np.meshgrid(E, E0)
            ind1d = np.argmin(np.abs(EE - EE0), 1)
            self.indEb = ind1d.reshape(E0.shape)
        self.indE0 = self.indEb.copy()

        self.logP = np.array([self.getLogP()])
        self.epochsDone = 0
        self.I_normalized = False

    # --- Initialization ---

    def initializeBand(self, kx, ky, Eb, offset=0.0, flipKAxes=False, kScale=1.0, interp_method="linear"):
        """Set E0 according to reference band (e.g. DFT calculation).

        Parameters
        ----------
        kx, ky : 1D array
            Momentum values for reference data.
        Eb : 2D array
            Energy values for band mapping data.
        offset : float
            Offset to be added to reference energy values.
        flipKAxes : bool
            If True, interchange momentum axes.
        kScale : float
            Scaling factor applied to k axes.
        interp_method : str
            Interpolation method ('linear' or 'nearest').
        """
        kx_in = kx.copy()
        ky_in = ky.copy()
        Eb_in = Eb.copy()

        if flipKAxes:
            kx_in, ky_in = (ky_in, kx_in)
            Eb_in = np.transpose(Eb_in)

        self.kscale = kScale
        kx_in *= self.kscale
        ky_in *= self.kscale

        intFunc = interpolate.RegularGridInterpolator(
            (kx_in, ky_in), Eb_in, method=interp_method, bounds_error=False, fill_value=None
        )
        kxx, kyy = np.meshgrid(self.kx, self.ky, indexing="ij")
        kxx = np.reshape(kxx, (self.lengthKx * self.lengthKy,))
        kyy = np.reshape(kyy, (self.lengthKx * self.lengthKy,))
        Einterp = intFunc(np.column_stack((kxx, kyy)))

        self.offset = offset
        self.E0 = np.reshape(Einterp + self.offset, (self.lengthKx, self.lengthKy))

        EE, EE0 = np.meshgrid(self.E, self.E0)
        ind1d = np.argmin(np.abs(EE - EE0), 1)
        self.indEb = ind1d.reshape(self.E0.shape)
        self.indE0 = self.indEb.copy()
        self.delHist()

    # --- Preprocessing ---

    def smoothenI(self, sigma=(1.0, 1.0, 1.0)):
        """Apply multidimensional Gaussian filter to intensity data."""
        self.I = ndimage.gaussian_filter(self.I, sigma=sigma)
        self.delHist()

    def symmetrizeI(self, mirror=True, rotational=True, rotational_order=6):
        """Symmetrize I with respect to mirror and rotational symmetry."""
        if mirror:
            indXRef = np.min(np.where(self.kx > 0.0)[0])
            lIndX = np.min([indXRef, self.lengthKx - indXRef])
            indX = np.arange(indXRef - lIndX, indXRef + lIndX)
            self.I[indX, :, :] = (self.I[indX, :, :] + self.I[np.flip(indX, axis=0), :, :]) / 2

            indYRef = np.min(np.where(self.ky > 0.0)[0])
            lIndY = np.min([indYRef, self.lengthKy - indYRef])
            indY = np.arange(indYRef - lIndY, indYRef + lIndY)
            self.I[:, indY, :] = (self.I[:, indY, :] + self.I[:, np.flip(indY, axis=0), :]) / 2

        if rotational:
            from ..mrf.symmetry import rotosymmetrize

            center = (np.argmin(np.abs(self.kx)), np.argmin(np.abs(self.ky)))
            for i in range(self.I.shape[2]):
                self.I[:, :, i], _ = rotosymmetrize(self.I[:, :, i], center, rotsym=rotational_order)

        self.delHist()

    def generateI(self):
        pass

    # --- Sequential optimization (numpy) ---

    def iter_seq(self, num_epoch=1, updateLogP=False, disable_tqdm=False):
        """Iterate band structure reconstruction sequentially.

        Parameters
        ----------
        num_epoch : int
            Number of iterations.
        updateLogP : bool
            If True, logP is updated every half epoch.
        disable_tqdm : bool
            If True, no progress bar is shown.
        """
        logI = np.log(self.I)
        ENN = self.E / (np.sqrt(2) * self.eta)
        if self.includeCurv:
            ECurv = self.E / (np.sqrt(2) * self.etaCurv)

        indList = np.random.choice(self.lengthKx * self.lengthKy, self.lengthKx * self.lengthKy * num_epoch)
        for i, ind in enumerate(tqdm(indList, disable=disable_tqdm)):
            indx = ind // self.lengthKy
            indy = ind % self.lengthKy
            logP = np.zeros(self.lengthE)
            if indx > 0:
                logP -= (ENN - ENN[self.indEb[indx - 1, indy]]) ** 2
            if indx < (self.lengthKx - 1):
                logP -= (ENN - ENN[self.indEb[indx + 1, indy]]) ** 2
            if indy > 0:
                logP -= (ENN - ENN[self.indEb[indx, indy - 1]]) ** 2
            if indy < (self.lengthKy - 1):
                logP -= (ENN - ENN[self.indEb[indx, indy + 1]]) ** 2
            logP += logI[indx, indy, :]
            self.indEb[indx, indy] = np.argmax(logP)
            if updateLogP and (
                ((i + 1) % (self.lengthKx * self.lengthKy)) == 0
                or ((i + 1) % (self.lengthKx * self.lengthKy)) == (self.lengthKx * self.lengthKy // 2)
            ):
                self.logP = np.append(self.logP, self.getLogP())

        self.epochsDone += num_epoch

    # --- Parallel optimization (PyTorch) ---

    def iter_para(self, num_epoch=1, updateLogP=False, disable_tqdm=False):
        """Iterate band structure reconstruction in parallel using PyTorch.

        Uses checkerboard (white/black) node update pattern for fast convergence.

        Parameters
        ----------
        num_epoch : int
            Number of iteration epochs.
        updateLogP : bool
            If True, logP is updated every epoch.
        disable_tqdm : bool
            If True, no progress bar is shown.
        """
        if updateLogP:
            self.logP = np.append(self.logP, np.zeros(2 * num_epoch))

        lengthKx = 2 * (self.lengthKx // 2)
        lengthKy = 2 * (self.lengthKy // 2)
        indX, indY = np.meshgrid(np.arange(lengthKx, step=2), np.arange(lengthKy, step=2), indexing="ij")

        E1d = torch.tensor(self.E / (np.sqrt(2) * self.eta), dtype=torch.float32)
        E3d = E1d.reshape(1, 1, -1)

        logI = []
        indEb = []
        for i in range(2):
            logI_row = []
            indEb_row = []
            for j in range(2):
                logI_row.append(torch.tensor(np.log(self.I[indX + i, indY + j, :]), dtype=torch.float32))
                indEb_row.append(torch.tensor(
                    np.expand_dims(self.indEb[indX + i, indY + j], 2), dtype=torch.long
                ))
            logI.append(logI_row)
            indEb.append(indEb_row)

        for epoch in tqdm(range(num_epoch), disable=disable_tqdm):
            # White nodes
            logP = self._compute_logP_pt(E1d, E3d, logI, indEb, lengthKx)
            updateW = self._compute_update_pt(logP)
            for i in range(2):
                indEb[i][i] = updateW[i].unsqueeze(2)
            if updateLogP:
                self.logP[2 * epoch + 1] = self._compute_logPTot_pt(logP, logI, indEb).item()

            # Black nodes
            logP = self._compute_logP_pt(E1d, E3d, logI, indEb, lengthKx)
            updateB = self._compute_update_pt(logP)
            for i in range(2):
                indEb[i][1 - i] = updateB[i].unsqueeze(2)
            if updateLogP:
                self.logP[2 * epoch + 2] = self._compute_logPTot_pt(logP, logI, indEb).item()

        # Extract results
        for i in range(2):
            for j in range(2):
                self.indEb[indX + i, indY + j] = indEb[i][j].squeeze(2).numpy()

        self.epochsDone += num_epoch

    def _compute_logP_pt(self, E1d, E3d, logI, indEb, lengthKx):
        """Compute logP for checkerboard pattern using PyTorch."""
        logP = []
        for i in range(2):
            logP_row = []
            for j in range(2):
                gathered = torch.gather(E1d, 0, indEb[i][j].flatten()).reshape(indEb[i][j].shape)
                squDiff = (gathered - E3d) ** 2

                lp = logI[i][j].clone()
                if i == 0:
                    # left neighbor
                    lp = lp - torch.nn.functional.pad(
                        squDiff[1:, :, :], (0, 0, 0, 0, 0, 1)
                    )
                if i == 1:
                    lp = lp - torch.nn.functional.pad(
                        squDiff[:-1, :, :], (0, 0, 0, 0, 1, 0)
                    )
                if j == 0:
                    lp = lp - torch.nn.functional.pad(
                        squDiff[:, 1:, :], (0, 0, 0, 1, 0, 0)
                    )
                if j == 1:
                    lp = lp - torch.nn.functional.pad(
                        squDiff[:, :-1, :], (0, 0, 1, 0, 0, 0)
                    )
                logP_row.append(lp)
            logP.append(logP_row)
        return logP

    def _compute_update_pt(self, logP):
        """Compute update for white/black nodes."""
        return [logP[i][i].argmax(dim=2) for i in range(2)]

    def _compute_logPTot_pt(self, logP, logI, indEb):
        """Compute total logP."""
        total = torch.tensor(0.0)
        for i in range(2):
            for j in range(2):
                gathered = torch.gather(logP[i][j], 2, indEb[i][j])
                total = total + gathered.sum()
        return total

    # --- Output ---

    def getEb(self):
        """Retrieve the energy values of the reconstructed band."""
        return self.E[self.indEb].copy()

    def getLogP(self):
        """Retrieve the log likelihood of the electronic band structure."""
        indKx, indKy = np.meshgrid(np.arange(self.lengthKx), np.arange(self.lengthKy), indexing="ij")
        logP = np.sum(np.log(self.I[indKx, indKy, self.indEb]))
        Eb = self.getEb()
        if self.lengthKx > 1:
            logP -= np.sum((Eb[0 : (self.lengthKx - 1), :] - Eb[1 : self.lengthKx, :]) ** 2) / (2 * self.eta**2)
        if self.lengthKy > 1:
            logP -= np.sum((Eb[:, 0 : (self.lengthKy - 1)] - Eb[:, 1 : self.lengthKy]) ** 2) / (2 * self.eta**2)
        return logP

    # --- Utilities ---

    def delHist(self):
        """Delete training history by resetting delta log(p)."""
        self.logP = np.array([self.getLogP()])
        self.epochsDone = 0

    def saveBand(self, fileName, hyperparams=True, index=None):
        """Save reconstructed band to HDF5 file."""
        import h5py

        with h5py.File(fileName, "w") as f:
            f.create_dataset("/axes/kx", data=self.kx)
            f.create_dataset("/axes/ky", data=self.ky)
            f.create_dataset("/bands/Einit", data=self.E0)
            f.create_dataset("/bands/Eb", data=self.getEb())
            if hyperparams:
                band_index = index if index is not None else getattr(self, "band_index", 0)
                f.create_dataset("/hyper/band_index", data=band_index)
                f.create_dataset("/hyper/k_scale", data=self.kscale)
                f.create_dataset("/hyper/E_offset", data=self.offset)
                f.create_dataset("/hyper/nn_eta", data=self.eta)

    def loadBand(self, Eb=None, fileName=None, use_as_init=True):
        """Load bands into reconstruction object."""
        if fileName is not None:
            import h5py

            with h5py.File(fileName, "r") as f:
                if self.lengthKx == f["/axes/kx"].shape[0] and self.lengthKy == f["/axes/ky"].shape[0]:
                    Eb = np.asarray(f["/bands/Eb"])
        if Eb is not None:
            EE, EEb = np.meshgrid(self.E, Eb)
            ind1d = np.argmin(np.abs(EE - EEb), 1)
            self.indEb = ind1d.reshape(Eb.shape)
        if use_as_init:
            self.indE0 = self.indEb.copy()
            self.delHist()