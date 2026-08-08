"""MRF band reconstruction: MrfRec model with PyTorch backend.

Extracted from fuller.mrfRec, with TensorFlow replaced by PyTorch.
"""


import numpy as np
import torch
from scipy import interpolate, ndimage
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
    max_shift : int or None
        Maximum number of energy-grid steps a node may move away from its
        initial value (E0). None disables the constraint. Prevents the band
        from being pulled across to a neighbouring band during iteration.
    """

    def __init__(self, E, kx=None, ky=None, I=None, E0=None, eta=0.1, max_shift=None):
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
        self.max_shift = int(max_shift) if max_shift is not None else None

        if E0 is None:
            self.indEb = np.ones((self.lengthKx, self.lengthKy), int) * int(self.lengthE / 2)
        else:
            EE, EE0 = np.meshgrid(E, E0)
            ind1d = np.argmin(np.abs(EE - EE0), 1)
            self.indEb = ind1d.reshape(E0.shape)
        self.indE0 = self.indEb.copy()

        self.logP = np.array([self.getLogP()])
        self.epochsDone = 0

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

    # --- Sequential optimization (numpy) ---

    def iter_seq(self, num_epoch=1, updateLogP=False, disable_tqdm=False):
        """Iterate band structure reconstruction sequentially.

        One epoch performs one Gibbs-style sweep: every node is visited
        exactly once per epoch in a random (permutation) order.

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
        lengthKx = self.lengthKx
        lengthKy = self.lengthKy
        total = lengthKx * lengthKy

        for epoch in range(num_epoch):
            indList = np.random.permutation(total)
            for i, ind in enumerate(tqdm(indList, disable=disable_tqdm)):
                indx = ind // lengthKy
                indy = ind % lengthKy
                self._update_node_seq(indx, indy, logI)
                if updateLogP and (
                    (i + 1) == total // 2 or (i + 1) == total
                ):
                    self.logP = np.append(self.logP, self.getLogP())

        self.epochsDone += num_epoch

    def _update_node_seq(self, indx, indy, logI):
        """Update a single node by maximizing its local log-probability."""
        ENN = self.E / (np.sqrt(2) * self.eta)
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
        if self.max_shift is not None:
            lo = max(0, self.indE0[indx, indy] - self.max_shift)
            hi = min(self.lengthE - 1, self.indE0[indx, indy] + self.max_shift)
            logP[:lo] = -np.inf
            logP[hi + 1:] = -np.inf
        self.indEb[indx, indy] = np.argmax(logP)

    # --- Parallel optimization (PyTorch) ---

    def iter_para(self, num_epoch=1, updateLogP=False, disable_tqdm=False):
        """Iterate band structure reconstruction in parallel using PyTorch.

        Uses a checkerboard (white/black) node update pattern: all white
        nodes are updated simultaneously using the frozen black nodes as
        neighbors, then the black nodes using the freshly updated white
        nodes. Trailing odd rows/columns (not covered by the checkerboard)
        are updated sequentially.

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

        nx_full, ny_full = self.lengthKx, self.lengthKy
        lengthKx = 2 * (nx_full // 2)
        lengthKy = 2 * (ny_full // 2)
        indX, indY = np.meshgrid(np.arange(lengthKx, step=2), np.arange(lengthKy, step=2), indexing="ij")

        E1d = torch.tensor(self.E / (np.sqrt(2) * self.eta), dtype=torch.float32)
        E3d = E1d.reshape(1, 1, -1)

        logI = []
        indEb = []
        indEb0 = []
        for i in range(2):
            logI_row = []
            indEb_row = []
            indEb0_row = []
            for j in range(2):
                logI_row.append(torch.tensor(np.log(self.I[indX + i, indY + j, :]), dtype=torch.float32))
                indEb_row.append(torch.tensor(
                    np.expand_dims(self.indEb[indX + i, indY + j], 2), dtype=torch.long
                ))
                indEb0_row.append(torch.tensor(
                    np.expand_dims(self.indE0[indX + i, indY + j], 2), dtype=torch.long
                ))
            logI.append(logI_row)
            indEb.append(indEb_row)
            indEb0.append(indEb0_row)

        for epoch in tqdm(range(num_epoch), disable=disable_tqdm):
            # White nodes
            logP = self._compute_logP_pt(E1d, E3d, logI, indEb)
            updateW = self._compute_update_pt(logP, indEb0, self.max_shift)
            for i in range(2):
                indEb[i][i] = updateW[i].unsqueeze(2)
            if updateLogP:
                self.logP[2 * epoch + 1] = self._compute_logPTot_pt(logP, logI, indEb).item()

            # Black nodes
            logP = self._compute_logP_pt(E1d, E3d, logI, indEb)
            updateB = self._compute_update_pt(logP, indEb0, self.max_shift, black=True)
            for i in range(2):
                indEb[i][1 - i] = updateB[i].unsqueeze(2)
            if updateLogP:
                self.logP[2 * epoch + 2] = self._compute_logPTot_pt(logP, logI, indEb).item()

        # Extract results
        for i in range(2):
            for j in range(2):
                self.indEb[indX + i, indY + j] = indEb[i][j].squeeze(2).numpy()

        # Sequential update of trailing odd rows/columns not covered above
        logI_np = np.log(self.I)
        if lengthKx < nx_full or lengthKy < ny_full:
            for indx in range(nx_full):
                for indy in range(ny_full):
                    if indx >= lengthKx or indy >= lengthKy:
                        self._update_node_seq(indx, indy, logI_np)

        self.epochsDone += num_epoch

    @staticmethod
    def _gather_energies(E1d, indEb_cell):
        """Energy values of the nodes of one checkerboard cell."""
        return torch.gather(E1d, 0, indEb_cell.flatten()).reshape(indEb_cell.shape[:2])

    @staticmethod
    def _row_neighbor_energies(gE_row, i_par):
        """Energies/masks of the row neighbors (±1 row) for row parity i_par.

        Returns (E_up, E_dn, m_up, m_dn): for cell node r, E_up/E_dn hold the
        energy of the node one row above/below (from the opposite-parity
        cell) and m_up/m_dn are 0 for boundary nodes without a neighbor.
        """
        if i_par == 0:
            # cell node r (global row 2r): upper neighbor = cell row r-1
            # (none for r=0), lower neighbor = cell row r (none for r=R-1)
            E_up = torch.nn.functional.pad(gE_row[:-1, :], (0, 0, 1, 0))
            m_up = torch.nn.functional.pad(torch.ones_like(gE_row[:-1, :]), (0, 0, 1, 0))
            E_dn = torch.nn.functional.pad(gE_row, (0, 0, 0, 1))[:-1, :]
            m_dn = torch.nn.functional.pad(torch.ones_like(gE_row), (0, 0, 0, 1))[:-1, :]
        else:
            # cell node r (global row 2r+1): upper neighbor = cell row r
            # (always exists), lower neighbor = cell row r+1 (none for r=R-1)
            E_up = gE_row.clone()
            m_up = torch.ones_like(gE_row)
            E_dn = torch.nn.functional.pad(gE_row[1:, :], (0, 0, 0, 1))
            m_dn = torch.nn.functional.pad(torch.ones_like(gE_row[1:, :]), (0, 0, 0, 1))
        return E_up, E_dn, m_up, m_dn

    @staticmethod
    def _col_neighbor_energies(gE_col, j_par):
        """Energies/masks of the column neighbors (±1 column) for parity j_par."""
        if j_par == 0:
            # cell node c (global col 2c): left neighbor = cell col c-1
            # (none for c=0), right neighbor = cell col c (none for c=C-1)
            E_lf = torch.nn.functional.pad(gE_col[:, :-1], (1, 0, 0, 0))
            m_lf = torch.nn.functional.pad(torch.ones_like(gE_col[:, :-1]), (1, 0, 0, 0))
            E_rt = torch.nn.functional.pad(gE_col, (0, 1, 0, 0))[:, :-1]
            m_rt = torch.nn.functional.pad(torch.ones_like(gE_col), (0, 1, 0, 0))[:, :-1]
        else:
            # cell node c (global col 2c+1): left neighbor = cell col c
            # (always exists), right neighbor = cell col c+1 (none for c=C-1)
            E_lf = gE_col.clone()
            m_lf = torch.ones_like(gE_col)
            E_rt = torch.nn.functional.pad(gE_col[:, 1:], (0, 1, 0, 0))
            m_rt = torch.nn.functional.pad(torch.ones_like(gE_col[:, 1:]), (0, 1, 0, 0))
        return E_lf, E_rt, m_lf, m_rt

    def _compute_logP_pt(self, E1d, E3d, logI, indEb):
        """Compute logP for the checkerboard pattern using PyTorch.

        For a cell (i, j) the row neighbors (±1 row) are frozen nodes of the
        opposite-parity row cell (1-i, j) and the column neighbors (±1
        column) are frozen nodes of the opposite-parity column cell (i, 1-j).
        """
        logP = []
        for i in range(2):
            logP_row = []
            for j in range(2):
                lp = logI[i][j].clone()

                gE_row = self._gather_energies(E1d, indEb[1 - i][j])
                E_up, E_dn, m_up, m_dn = self._row_neighbor_energies(gE_row, i)

                gE_col = self._gather_energies(E1d, indEb[i][1 - j])
                E_lf, E_rt, m_lf, m_rt = self._col_neighbor_energies(gE_col, j)

                for E_nbr, m_nbr in ((E_up, m_up), (E_dn, m_dn), (E_lf, m_lf), (E_rt, m_rt)):
                    lp = lp - (E_nbr[:, :, None] - E3d) ** 2 * m_nbr[:, :, None]
                logP_row.append(lp)
            logP.append(logP_row)
        return logP

    def _compute_update_pt(self, logP, indEb0=None, max_shift=None, black=False):
        """Compute update for white/black nodes.

        In the white phase cell (i, i) is updated using logP[i][i]; in the
        black phase cell (i, 1-i) is updated using logP[i][1-i].

        If ``max_shift`` is given, candidates further than ``max_shift``
        energy-grid steps from the initial value (indE0) are forbidden,
        preventing the band from jumping to a neighbouring band.
        """
        updates = []
        for i in range(2):
            j = 1 - i if black else i
            lp = logP[i][j]
            if max_shift is not None and indEb0 is not None:
                lo = indEb0[i][j] - max_shift
                hi = indEb0[i][j] + max_shift
                idx = torch.arange(lp.shape[2], device=lp.device)[None, None, :]
                mask = (idx >= lo) & (idx <= hi)
                lp = torch.where(mask, lp, torch.full_like(lp, -float("inf")))
            updates.append(lp.argmax(dim=2))
        return updates

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
