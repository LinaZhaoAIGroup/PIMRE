#!/usr/bin/env python3
"""PIMRE Configuration Setup GUI (PyQt5).

A PyQt5 GUI with embedded matplotlib for configuring sample parameters,
ARPES data, DFT processing, calibration, and MRF hyperparameters.

Usage:
    uv run python scripts/setup_config.py
    uv run python scripts/setup_config.py --config configs/pimre_config.yaml
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Qt5Agg")
import h5py
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pimre.config import (
    load_config,
    parse_kpoints,
    parse_outcar,
    save_config,
)
from pimre.utils.interaction import DraggableHLine, DraggableVLine

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "configs", "pimre_config.yaml"
)


# ── helpers ────────────────────────────────────────────────────────────

def _build_axis(ax_cfg, flip=False):
    s, d, n = ax_cfg["start"], ax_cfg["delta"], ax_cfg["npts"]
    arr = np.linspace(s, s + d * (n - 1), n)
    if flip:
        arr = arr[::-1]
    return arr


# ── ARPES preview canvas ───────────────────────────────────────────────

class ArpesPreviewCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumSize(400, 300)

    def show_bands(self, bands, kx_angle, ky_angle, layer=10):
        self.ax.clear()
        if bands is None:
            self.draw()
            return
        layer = min(layer, bands.shape[0] - 1)
        self.ax.imshow(
            bands[layer], aspect="auto", cmap="plasma", origin="lower",
            extent=[ky_angle[0], ky_angle[-1], kx_angle[0], kx_angle[-1]],
        )
        self.ax.set_xlabel("ky angle (deg)")
        self.ax.set_ylabel("kx angle (deg)")
        self.ax.set_title(f"ARPES layer {layer} / {bands.shape[0] - 1}")
        self.fig.tight_layout()
        self.draw()


# ── Gamma calibration dialog ───────────────────────────────────────────

class GammaCalibDialog(QtWidgets.QDialog):
    """Modal dialog with embedded matplotlib figure for Gamma calibration."""

    def __init__(self, bands, kx_angle, ky_angle, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gamma Point Calibration")
        self.resize(750, 750)
        self.kx_shift = (kx_angle[0] + kx_angle[-1]) / 2
        self.ky_shift = (ky_angle[0] + ky_angle[-1]) / 2

        layout = QtWidgets.QVBoxLayout(self)

        self.fig = Figure(figsize=(7, 7))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.canvas)

        self.ax.imshow(bands[10], aspect="auto", cmap="plasma", origin="lower",
                       extent=[ky_angle[0], ky_angle[-1], kx_angle[0], kx_angle[-1]])
        self.ax.set_xlabel("ky angle (deg)")
        self.ax.set_ylabel("kx angle (deg)")
        self.ax.set_title("Drag lines to Γ point, then click OK")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

        cx = (ky_angle[0] + ky_angle[-1]) / 2
        cy = (kx_angle[0] + kx_angle[-1]) / 2
        self.vline = DraggableVLine(self.ax, x=cx, color="red", linestyle="--", linewidth=2)
        self.hline = DraggableHLine(self.ax, y=cy, color="green", linestyle="--", linewidth=2)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.clicked.connect(self._on_ok)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

    def _on_ok(self):
        self.kx_shift = self.hline.y
        self.ky_shift = self.vline.x
        self.accept()


# ── Collapsible group box ──────────────────────────────────────────────

class CollapsibleGroup(QtWidgets.QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._btn = QtWidgets.QPushButton(title)
        self._btn.setCheckable(True)
        self._btn.setChecked(True)
        self._btn.setStyleSheet("QPushButton { text-align: left; font-weight: bold; }")
        self._btn.clicked.connect(self._toggle)
        self._layout.addWidget(self._btn)
        self._content = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(10, 5, 10, 5)
        self._layout.addWidget(self._content)

    def _toggle(self):
        self._content.setVisible(self._btn.isChecked())

    def addWidget(self, w):
        self._content_layout.addWidget(w)

    def addLayout(self, lay):
        self._content_layout.addLayout(lay)


# ── helpers ────────────────────────────────────────────────────────────

def _labeled_edit(label, default="", width=120):
    w = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(0, 2, 0, 2)
    lay.addWidget(QtWidgets.QLabel(label))
    edit = QtWidgets.QLineEdit(str(default))
    edit.setFixedWidth(width)
    lay.addWidget(edit)
    lay.addStretch()
    return w, edit

def _labeled_check(label, checked=False):
    w = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(0, 2, 0, 2)
    lay.addWidget(QtWidgets.QLabel(label))
    cb = QtWidgets.QCheckBox()
    cb.setChecked(checked)
    lay.addWidget(cb)
    lay.addStretch()
    return w, cb

def _hsep():
    w = QtWidgets.QWidget()
    w.setFixedHeight(1)
    w.setStyleSheet("background-color: #ccc;")
    return w


# ── main window ────────────────────────────────────────────────────────

class ConfigSetupWindow(QtWidgets.QMainWindow):
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self._preview_bands = None
        self._preview_kx = None
        self._preview_ky = None
        self._build_ui()
        self._load_to_ui()
        self.setWindowTitle("PIMRE Configuration Setup")
        self.resize(950, 750)

    # ── build UI ────────────────────────────────────────────────────

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # Top bar
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Config:"))
        self._path_edit = QtWidgets.QLineEdit(self.config_path)
        top.addWidget(self._path_edit)
        for label, slot in [("Load", self._on_load), ("Save", self._on_save),
                            ("Save As…", self._on_save_as)]:
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(slot)
            top.addWidget(btn)
        main_layout.addLayout(top)

        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tabs)

        self._build_arpes_tab()
        self._build_dft_tab()
        self._build_calib_tab()
        self._build_mrf_tab()
        self._build_output_tab()

    def _make_tab(self, title):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        scroll.setWidget(widget)
        self.tabs.addTab(scroll, title)
        return layout

    # ── ARPES tab ───────────────────────────────────────────────────

    def _build_arpes_tab(self):
        lay = self._make_tab("ARPES Data")

        g = CollapsibleGroup("Sample")
        w, self._e_name = _labeled_edit("Name:", width=200)
        g.addWidget(w)
        lay.addWidget(g)

        lay.addWidget(_hsep())

        g = CollapsibleGroup("HDF5 File")
        w = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(w)
        hl.setContentsMargins(0, 2, 0, 2)
        self._e_hdf5 = QtWidgets.QLineEdit()
        hl.addWidget(self._e_hdf5)
        btn = QtWidgets.QPushButton("Browse…")
        btn.clicked.connect(self._browse_hdf5)
        hl.addWidget(btn)
        g.addWidget(w)
        w2, self._e_dataset = _labeled_edit("Dataset:", width=200)
        g.addWidget(w2)
        lay.addWidget(g)

        lay.addWidget(_hsep())

        for axis_name, label in [("energy", "Energy Axis"), ("kx_angle", "kx Angle Axis"), ("ky_angle", "ky Angle Axis")]:
            g = CollapsibleGroup(label)
            row = QtWidgets.QHBoxLayout()
            w_s, self.__dict__[f"_e_{axis_name}_start"] = _labeled_edit("Start:", width=100)
            w_d, self.__dict__[f"_e_{axis_name}_delta"] = _labeled_edit("Delta:", width=100)
            w_n, self.__dict__[f"_e_{axis_name}_npts"] = _labeled_edit("Npts:", width=80)
            w_f, self.__dict__[f"_cb_{axis_name}_flip"] = _labeled_check("Flip:")
            row.addWidget(w_s)
            row.addWidget(w_d)
            row.addWidget(w_n)
            row.addWidget(w_f)
            g.addLayout(row)
            lay.addWidget(g)

        lay.addWidget(_hsep())

        # Swap kx/ky button
        w = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(w)
        hl.setContentsMargins(0, 2, 0, 2)
        swap_btn = QtWidgets.QPushButton("Swap kx/ky Angles")
        swap_btn.clicked.connect(self._swap_kx_ky)
        hl.addWidget(swap_btn)
        hl.addStretch()
        lay.addWidget(w)

        w, self._e_wf = _labeled_edit("Work Function (eV):", width=100)
        lay.addWidget(w)

        btn = QtWidgets.QPushButton("Load & Preview ARPES Data")
        btn.clicked.connect(self._preview_arpes)
        lay.addWidget(btn)

        self._preview = ArpesPreviewCanvas()
        lay.addWidget(self._preview)

    def _swap_kx_ky(self):
        """Swap kx_angle and ky_angle axis values in the UI."""
        for key in ["_start", "_delta", "_npts"]:
            kx_v = self.__dict__["_e_kx_angle" + key].text()
            ky_v = self.__dict__["_e_ky_angle" + key].text()
            self.__dict__["_e_kx_angle" + key].setText(ky_v)
            self.__dict__["_e_ky_angle" + key].setText(kx_v)
        kx_f = self._cb_kx_angle_flip.isChecked()
        ky_f = self._cb_ky_angle_flip.isChecked()
        self._cb_kx_angle_flip.setChecked(ky_f)
        self._cb_ky_angle_flip.setChecked(kx_f)

    def _browse_hdf5(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select ARPES HDF5", "",
                                                      "HDF5 (*.h5 *.hdf5);;All (*)")
        if p:
            self._e_hdf5.setText(p)

    def _preview_arpes(self):
        self._sync_arpes_from_ui()
        ar = self.cfg["arpes"]
        path = ar["path"]
        ds = ar["dataset"]
        if not path or not ds:
            QtWidgets.QMessageBox.warning(self, "Missing", "Please set HDF5 path and dataset first.")
            return
        try:
            with h5py.File(path, "r") as f:
                parts = ds.split("/")
                d = f
                for p in parts:
                    d = d[p]
                bands = d[:]
            kx = _build_axis(ar["kx_angle"], ar["kx_angle"].get("flip", False))
            ky = _build_axis(ar["ky_angle"], ar["ky_angle"].get("flip", False))
            self._preview.show_bands(bands, kx, ky, layer=10)
            self._preview_bands = bands
            self._preview_kx = kx
            self._preview_ky = ky
            e_axis = _build_axis(ar["energy"], ar["energy"].get("flip", False))
            msg = f"Loaded {bands.shape}\nE=[{e_axis[0]:.4f} … {e_axis[-1]:.4f}]"
            QtWidgets.QMessageBox.information(self, "ARPES Loaded", msg)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    # ── DFT tab ────────────────────────────────────────────────────

    def _build_dft_tab(self):
        lay = self._make_tab("DFT Data")

        g = CollapsibleGroup("DFT Raw Data Directory")
        w = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(w)
        hl.setContentsMargins(0, 2, 0, 2)
        self._e_dft_dir = QtWidgets.QLineEdit()
        hl.addWidget(self._e_dft_dir)
        btn = QtWidgets.QPushButton("Browse…")
        btn.clicked.connect(self._browse_dft_dir)
        hl.addWidget(btn)
        g.addWidget(w)
        btn2 = QtWidgets.QPushButton("Read OUTCAR & KPOINTS")
        btn2.clicked.connect(self._read_outcar)
        g.addWidget(btn2)
        lay.addWidget(g)

        lay.addWidget(_hsep())

        g = CollapsibleGroup("Lattice Parameters")
        grid = QtWidgets.QGridLayout()
        self._e_a = QtWidgets.QLineEdit()
        self._e_b = QtWidgets.QLineEdit()
        self._e_c = QtWidgets.QLineEdit()
        self._e_alpha = QtWidgets.QLineEdit()
        self._e_beta = QtWidgets.QLineEdit()
        self._e_gamma = QtWidgets.QLineEdit()
        for col, (label, edit) in enumerate([
            ("a (Å):", self._e_a), ("b (Å):", self._e_b), ("c (Å):", self._e_c),
            ("α (deg):", self._e_alpha), ("β (deg):", self._e_beta), ("γ (deg):", self._e_gamma),
        ]):
            grid.addWidget(QtWidgets.QLabel(label), col // 3, (col % 3) * 2)
            edit.setFixedWidth(80)
            grid.addWidget(edit, col // 3, (col % 3) * 2 + 1)
        g.addLayout(grid)
        lay.addWidget(g)

        lay.addWidget(_hsep())

        g = CollapsibleGroup("K-grid")
        row = QtWidgets.QHBoxLayout()
        w1, self._e_nkx = _labeled_edit("nkx:", width=60)
        w2, self._e_nky = _labeled_edit("nky:", width=60)
        row.addWidget(w1)
        row.addWidget(w2)
        row.addStretch()
        g.addLayout(row)
        lay.addWidget(g)

        g = CollapsibleGroup("Output Grid")
        row = QtWidgets.QHBoxLayout()
        w1, self._e_dft_nx = _labeled_edit("nx:", width=60)
        w2, self._e_dft_ny = _labeled_edit("ny:", width=60)
        row.addWidget(w1)
        row.addWidget(w2)
        row.addStretch()
        g.addLayout(row)
        lay.addWidget(g)

        lay.addStretch()

    def _browse_dft_dir(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select DFT Raw_Data directory")
        if p:
            self._e_dft_dir.setText(p)

    def _read_outcar(self):
        dft_dir = self._e_dft_dir.text()
        if not dft_dir:
            QtWidgets.QMessageBox.warning(self, "Missing", "Please set DFT directory first.")
            return
        msgs = []
        outcar = os.path.join(dft_dir, "OUTCAR")
        lat = parse_outcar(outcar)
        if lat:
            self._e_a.setText(str(lat[0]))
            self._e_b.setText(str(lat[1]))
            self._e_c.setText(str(lat[2]))
            self._e_alpha.setText(str(lat[3]))
            self._e_beta.setText(str(lat[4]))
            self._e_gamma.setText(str(lat[5]))
            msgs.append(f"Lattice: a={lat[0]:.4f} b={lat[1]:.4f} c={lat[2]:.4f} "
                        f"α={lat[3]:.1f} β={lat[4]:.1f} γ={lat[5]:.1f}")
        else:
            msgs.append("OUTCAR not found or could not parse lattice.")
        kgrid = parse_kpoints(os.path.join(dft_dir, "KPOINTS"))
        if kgrid:
            self._e_nkx.setText(str(kgrid[0]))
            self._e_nky.setText(str(kgrid[1]))
            msgs.append(f"K-grid: {kgrid[0]} × {kgrid[1]}")
        else:
            msgs.append("KPOINTS not found.")
        QtWidgets.QMessageBox.information(self, "OUTCAR / KPOINTS", "\n".join(msgs))

    # ── Calibration tab ────────────────────────────────────────────

    def _build_calib_tab(self):
        lay = self._make_tab("Calibration")

        g = CollapsibleGroup("Angle-Space Gamma Calibration")
        row = QtWidgets.QHBoxLayout()
        w1, self._e_kx_shift = _labeled_edit("kx_shift (deg):", width=100)
        w2, self._e_ky_shift = _labeled_edit("ky_shift (deg):", width=100)
        row.addWidget(w1)
        row.addWidget(w2)
        row.addStretch()
        g.addLayout(row)
        btn = QtWidgets.QPushButton("Launch Gamma Calibration")
        btn.clicked.connect(self._launch_gamma_calib)
        g.addWidget(btn)
        lay.addWidget(g)

        lay.addWidget(_hsep())

        g = CollapsibleGroup("Momentum-Space Grid Shift")
        row = QtWidgets.QHBoxLayout()
        w1, self._e_kx_gs = _labeled_edit("kx_grid_shift (Å⁻¹):", width=120)
        w2, self._e_ky_gs = _labeled_edit("ky_grid_shift (Å⁻¹):", width=120)
        row.addWidget(w1)
        row.addWidget(w2)
        row.addStretch()
        g.addLayout(row)
        lay.addWidget(g)

        lay.addStretch()

    def _launch_gamma_calib(self):
        self._sync_arpes_from_ui()
        ar = self.cfg["arpes"]
        path = ar["path"]
        ds = ar["dataset"]
        if not path or not ds:
            QtWidgets.QMessageBox.warning(self, "Missing", "Please set ARPES path and dataset first.")
            return
        try:
            with h5py.File(path, "r") as f:
                parts = ds.split("/")
                d = f
                for p in parts:
                    d = d[p]
                bands = d[:]
            kx = _build_axis(ar["kx_angle"], ar["kx_angle"].get("flip", False))
            ky = _build_axis(ar["ky_angle"], ar["ky_angle"].get("flip", False))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        dlg = GammaCalibDialog(bands, kx, ky, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self._e_kx_shift.setText(f"{dlg.kx_shift:.4f}")
            self._e_ky_shift.setText(f"{dlg.ky_shift:.4f}")
            QtWidgets.QMessageBox.information(self, "Calibration",
                                              f"kx_shift={dlg.kx_shift:.4f}  ky_shift={dlg.ky_shift:.4f}")

    # ── MRF tab ────────────────────────────────────────────────────

    def _build_mrf_tab(self):
        lay = self._make_tab("MRF")

        g = CollapsibleGroup("Band Hyperparameters")
        self._band_container = QtWidgets.QWidget()
        self._band_layout = QtWidgets.QVBoxLayout(self._band_container)
        self._band_layout.setContentsMargins(0, 0, 0, 0)

        # Header row
        header = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(header)
        hl.setContentsMargins(0, 2, 0, 2)
        hl.addWidget(QtWidgets.QLabel(""))
        for c in ["DFT idx", "eta"]:
            hl.addWidget(QtWidgets.QLabel(f"<b>{c}</b>"))
        hl.addStretch()
        self._band_layout.addWidget(header)

        self._band_rows = []
        self._band_widgets = []
        g.addWidget(self._band_container)

        # +/- buttons
        btn_row = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("+ Add Band")
        add_btn.clicked.connect(self._add_band_row)
        btn_row.addWidget(add_btn)
        btn_row.addStretch()
        g.addLayout(btn_row)

        lay.addWidget(g)

        # Populate from config
        for ib in range(len(self.cfg["mrf"]["bands"])):
            self._add_band_row()

        lay.addWidget(_hsep())

        g = CollapsibleGroup("BSFI Parameters")
        row = QtWidgets.QHBoxLayout()
        w1, self._e_bsfi_range = _labeled_edit("Offset range (eV):", width=100)
        w2, self._e_bsfi_step = _labeled_edit("Step:", width=80)
        w3, self._e_bsfi_fine = _labeled_edit("Fine-tune:", width=80)
        row.addWidget(w1)
        row.addWidget(w2)
        row.addWidget(w3)
        row.addStretch()
        g.addLayout(row)
        row2 = QtWidgets.QHBoxLayout()
        w4, self._e_bsfi_wcorr = _labeled_edit("w_corr:", width=80)
        w5, self._e_bsfi_wint = _labeled_edit("w_int:", width=80)
        w6, self._e_bsfi_wsnr = _labeled_edit("w_snr:", width=80)
        w7, self._e_bsfi_wridge = _labeled_edit("w_ridge:", width=80)
        row2.addWidget(w4)
        row2.addWidget(w5)
        row2.addWidget(w6)
        row2.addWidget(w7)
        row2.addStretch()
        g.addLayout(row2)
        row2b = QtWidgets.QHBoxLayout()
        w8, self._e_bsfi_wpath = _labeled_edit("w_path_ridge:", width=100)
        w9, self._e_bsfi_rsigma = _labeled_edit("ridge_sigma:", width=100)
        row2b.addWidget(w8)
        row2b.addWidget(w9)
        row2b.addStretch()
        g.addLayout(row2b)
        lay.addWidget(g)

        lay.addWidget(_hsep())

        row = QtWidgets.QHBoxLayout()
        w1, self._e_smooth = _labeled_edit("Smooth sigma (kx,ky,E):", width=180)
        w3, self._e_max_shift = _labeled_edit("max_shift:", width=80)
        row.addWidget(w1)
        row.addWidget(w3)
        row.addStretch()
        lay.addLayout(row)

        row1a = QtWidgets.QHBoxLayout()
        self._cb_device = QtWidgets.QComboBox()
        self._cb_device.addItems(["auto", "cpu", "cuda"])
        w_dev = QtWidgets.QLabel("Device:")
        w_dev.setToolTip("Torch device for the MRF checkerboard update. "
                         "auto: CUDA GPU if available, else CPU (≈10x faster on GPU, "
                         "identical results). cpu/cuda force a specific device.")
        row1a.addWidget(w_dev)
        row1a.addWidget(self._cb_device)
        row1a.addStretch()
        lay.addLayout(row1a)

        row1b = QtWidgets.QHBoxLayout()
        self._cb_alignment = QtWidgets.QComboBox()
        self._cb_alignment.addItems(["hsp", "gamma"])
        w_al = QtWidgets.QLabel("Alignment:")
        w_al.setToolTip("hsp: stretch/rotate the DFT grid so its K/M points land on the "
                        "experimental ones (recommended). gamma: identity, only when both "
                        "momentum axes already share the same absolute calibration.")
        self._cb_offset_mode = QtWidgets.QComboBox()
        self._cb_offset_mode.addItems(["per_band", "shared", "hierarchical"])
        w_om = QtWidgets.QLabel("Offset mode:")
        w_om.setToolTip("per_band: each band takes its own BSFI optimum "
                        "(recommended for metallic systems). shared: one offset for all bands. "
                        "hierarchical: shared coarse search + per-band fine-tune within "
                        "±bsfi.fine_tune_range (reference behaviour).")
        self._cb_occ = QtWidgets.QComboBox()
        self._cb_occ.addItems(["true", "false"])
        w_occ = QtWidgets.QLabel("Occupied only:")
        w_occ.setToolTip("true: restrict bands to occupied states (E0>=0 and E_dft<=0). "
                         "false: align full bands including empty-state segments "
                         "(reference behaviour for near-E_F metallic bands).")
        row1b.addWidget(w_al)
        row1b.addWidget(self._cb_alignment)
        row1b.addWidget(w_om)
        row1b.addWidget(self._cb_offset_mode)
        row1b.addWidget(w_occ)
        row1b.addWidget(self._cb_occ)
        row1b.addStretch()
        lay.addLayout(row1b)

        row2 = QtWidgets.QHBoxLayout()
        self._cb_path_interp = QtWidgets.QComboBox()
        self._cb_path_interp.addItems(["cubic", "linear", "nearest"])
        w4 = QtWidgets.QLabel("Path interp:")
        w5, self._e_path_step = _labeled_edit("Path sample (Å⁻¹/px):", width=100)
        row2.addWidget(w4)
        row2.addWidget(self._cb_path_interp)
        row2.addWidget(w5)
        row2.addStretch()
        lay.addLayout(row2)

        lay.addStretch()

    # ── Output tab ─────────────────────────────────────────────────

    def _build_output_tab(self):
        lay = self._make_tab("Output")

        g = CollapsibleGroup("Preprocessing Output")
        w = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(w)
        hl.setContentsMargins(0, 2, 0, 2)
        self._e_out_path = QtWidgets.QLineEdit()
        hl.addWidget(self._e_out_path)
        btn = QtWidgets.QPushButton("Browse…")
        btn.clicked.connect(self._browse_out)
        hl.addWidget(btn)
        g.addWidget(w)
        row = QtWidgets.QHBoxLayout()
        w1, self._e_out_grid = _labeled_edit("Output grid:", width=80)
        w2, self._e_kd_radius = _labeled_edit("KD radius:", width=80)
        w3, self._e_stride = _labeled_edit("Stride:", width=80)
        w4, self._e_nrot = _labeled_edit("N rotations:", width=80)
        row.addWidget(w1)
        row.addWidget(w2)
        row.addWidget(w3)
        row.addWidget(w4)
        row.addStretch()
        g.addLayout(row)
        row2 = QtWidgets.QHBoxLayout()
        w_s, self._cb_sort_axes = _labeled_check("Sort axes", False)
        w_a, self._cb_auto_grid = _labeled_check("Auto grid", False)
        w_n, self._cb_normalize = _labeled_check("Normalize", True)
        w_n.setToolTip("Divide the raw counts by the global maximum so the "
                       "intensity lies in [0, 1] (same convention as the "
                       "reference HPES preprocessed data). Applied to raw "
                       "input before rotation/interpolation.")
        w_w, self._e_workers = _labeled_edit("Workers (0=auto):", width=90)
        w_w.setToolTip("Number of parallel processes for the per-layer KD/quadrant "
                       "interpolation. 0 = use all CPU cores, 1 = serial.")
        row2.addWidget(w_s)
        row2.addWidget(w_a)
        row2.addWidget(w_n)
        row2.addWidget(w_w)
        row2.addStretch()
        g.addLayout(row2)
        row3 = QtWidgets.QHBoxLayout()
        row3.addWidget(QtWidgets.QLabel("Method:"))
        self._cb_method = QtWidgets.QComboBox()
        self._cb_method.addItems(["kdtree", "quadrant", "direct"])
        row3.addWidget(self._cb_method)
        w_fx, self._cb_flip_kx = _labeled_check("Flip kx", True)
        w_fy, self._cb_flip_ky = _labeled_check("Flip ky", True)
        row3.addWidget(w_fx)
        row3.addWidget(w_fy)
        w_sr, self._e_smooth_radius = _labeled_edit("Quad smooth radius:", width=80)
        row3.addWidget(w_sr)
        row3.addStretch()
        g.addLayout(row3)
        lay.addWidget(g)

        lay.addStretch()

    def _browse_out(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Preprocessed output file", "",
                                                      "HDF5 (*.h5);;All (*)")
        if p:
            self._e_out_path.setText(p)

    def _add_band_row(self):
        """Add a band row to the MRF band table."""
        ib = len(self._band_rows)
        w = QtWidgets.QWidget()
        row_lay = QtWidgets.QHBoxLayout(w)
        row_lay.setContentsMargins(0, 2, 0, 2)
        row_lay.addWidget(QtWidgets.QLabel(f"Band {ib}"))
        row = {}
        for key in ["index", "eta"]:
            e = QtWidgets.QLineEdit()
            e.setFixedWidth(80)
            row_lay.addWidget(e)
            row[key] = e
        row["index"].setToolTip(
            "DFT band selector: an integer is the absolute position in the "
            "stacked band array (after drop_top_bands); 'vbm:N' selects the "
            "band N steps below the valence-band maximum (0 = VBM), "
            "independent of drop_top_bands.")
        row["eta"].setToolTip("MRF smoothness parameter eta for this band.")
        del_btn = QtWidgets.QPushButton("✕")
        del_btn.setFixedWidth(30)
        del_btn.clicked.connect(lambda: self._remove_band_row(w))
        row_lay.addWidget(del_btn)
        row_lay.addStretch()
        self._band_layout.addWidget(w)
        self._band_rows.append(row)
        self._band_widgets.append(w)

    def _remove_band_row(self, widget):
        """Remove a band row from the MRF band table."""
        idx = self._band_widgets.index(widget)
        self._band_layout.removeWidget(widget)
        widget.deleteLater()
        del self._band_rows[idx]
        del self._band_widgets[idx]
        # Renumber labels
        for i, w in enumerate(self._band_widgets):
            lbl = w.layout().itemAt(0).widget()
            lbl.setText(f"Band {i}")

    # ── sync UI ↔ config ───────────────────────────────────────────

    def _sync_arpes_from_ui(self):
        ar = self.cfg["arpes"]
        ar["path"] = self._e_hdf5.text()
        ar["dataset"] = self._e_dataset.text()
        ar["work_function"] = float(self._e_wf.text() or 16.03)
        for axis in ["energy", "kx_angle", "ky_angle"]:
            try:
                s = float(self.__dict__[f"_e_{axis}_start"].text())
                d = float(self.__dict__[f"_e_{axis}_delta"].text())
                n = int(self.__dict__[f"_e_{axis}_npts"].text())
            except ValueError:
                s = ar[axis]["start"]
                d = ar[axis]["delta"]
                n = ar[axis]["npts"]
            ar[axis]["start"] = s
            ar[axis]["delta"] = d
            ar[axis]["npts"] = n
            ar[axis]["flip"] = self.__dict__[f"_cb_{axis}_flip"].isChecked()

    def _sync_all_from_ui(self):
        self._sync_arpes_from_ui()
        self.cfg["sample"]["name"] = self._e_name.text()
        self.cfg["dft"]["path"] = self._e_dft_dir.text()
        self.cfg["dft"]["k_grid"] = [int(self._e_nkx.text() or 20), int(self._e_nky.text() or 20)]
        self.cfg["dft"]["output_grid"] = [int(self._e_dft_nx.text() or 101), int(self._e_dft_ny.text() or 101)]

        lat = self.cfg["lattice"]
        lat["a"] = float(self._e_a.text() or 5.8077)
        lat["b"] = float(self._e_b.text() or 5.8077)
        lat["c"] = float(self._e_c.text() or 9.1297)
        lat["alpha"] = float(self._e_alpha.text() or 90)
        lat["beta"] = float(self._e_beta.text() or 90)
        lat["gamma"] = float(self._e_gamma.text() or 120)

        cal = self.cfg["calibration"]
        cal["kx_shift"] = float(self._e_kx_shift.text() or 0)
        cal["ky_shift"] = float(self._e_ky_shift.text() or 0)
        cal["kx_grid_shift"] = float(self._e_kx_gs.text() or 0)
        cal["ky_grid_shift"] = float(self._e_ky_gs.text() or 0)

        bands = []
        for ib, row in enumerate(self._band_rows):
            raw = row["index"].text().strip().lower()
            eta = float(row["eta"].text() or 1e-6)
            if raw.startswith("vbm"):
                bands.append({"from_vbm": int(raw.split(":")[-1] or 0), "eta": eta})
            else:
                bands.append({"index": int(raw or ib), "eta": eta})
        self.cfg["mrf"]["bands"] = bands

        bsfi = self.cfg["mrf"]["bsfi"]
        bsfi["offset_range"] = float(self._e_bsfi_range.text() or 1.0)
        bsfi["offset_step"] = float(self._e_bsfi_step.text() or 0.1)
        bsfi["fine_tune_range"] = float(self._e_bsfi_fine.text() or 0.05)
        bsfi["weights"]["correlation"] = float(self._e_bsfi_wcorr.text() or 0.6)
        bsfi["weights"]["intensity"] = float(self._e_bsfi_wint.text() or 0.3)
        bsfi["weights"]["snr"] = float(self._e_bsfi_wsnr.text() or 0.1)
        bsfi["weights"]["ridge"] = float(self._e_bsfi_wridge.text() or 0.5)
        bsfi["weights"]["path_ridge"] = float(self._e_bsfi_wpath.text() or 0.8)
        bsfi["ridge_sigma"] = float(self._e_bsfi_rsigma.text() or 0.1)

        smooth_str = self._e_smooth.text()
        if smooth_str:
            self.cfg["mrf"]["smooth_sigma"] = [float(x) for x in smooth_str.replace(",", " ").split()]
        if self._e_max_shift.text():
            self.cfg["mrf"]["max_shift"] = int(self._e_max_shift.text())
        self.cfg["mrf"]["alignment"] = self._cb_alignment.currentText()
        self.cfg["mrf"]["offset_mode"] = self._cb_offset_mode.currentText()
        self.cfg["mrf"]["occupied_only"] = self._cb_occ.currentText() == "true"
        self.cfg["mrf"]["device"] = self._cb_device.currentText()
        self.cfg["mrf"]["path_interp_method"] = self._cb_path_interp.currentText()
        self.cfg["mrf"]["path_sample_step"] = float(self._e_path_step.text() or 0.005)

        pp = self.cfg["preprocessing"]
        pp["output_path"] = self._e_out_path.text()
        pp["output_grid"] = int(self._e_out_grid.text() or 200)
        pp["method"] = self._cb_method.currentText()
        pp["kd_radius"] = float(self._e_kd_radius.text() or 0.05)
        pp["stride"] = int(self._e_stride.text() or 10)
        pp["n_rotations"] = int(self._e_nrot.text() or 6)
        pp["sort_axes"] = self._cb_sort_axes.isChecked()
        pp["auto_grid"] = self._cb_auto_grid.isChecked()
        pp["normalize"] = self._cb_normalize.isChecked()
        pp["workers"] = int(self._e_workers.text() or 0)
        pp.setdefault("quadrant", {})
        pp["quadrant"]["flip_kx"] = self._cb_flip_kx.isChecked()
        pp["quadrant"]["flip_ky"] = self._cb_flip_ky.isChecked()
        pp["quadrant"]["smooth_radius"] = float(self._e_smooth_radius.text() or 0.02)

    def _load_to_ui(self):
        self._e_name.setText(self.cfg["sample"]["name"])
        ar = self.cfg["arpes"]
        self._e_hdf5.setText(ar["path"])
        self._e_dataset.setText(ar["dataset"])
        self._e_wf.setText(str(ar["work_function"]))
        for axis in ["energy", "kx_angle", "ky_angle"]:
            self.__dict__[f"_e_{axis}_start"].setText(str(ar[axis]["start"]))
            self.__dict__[f"_e_{axis}_delta"].setText(str(ar[axis]["delta"]))
            self.__dict__[f"_e_{axis}_npts"].setText(str(ar[axis]["npts"]))
            self.__dict__[f"_cb_{axis}_flip"].setChecked(ar[axis].get("flip", False))

        self._e_dft_dir.setText(self.cfg["dft"]["path"])
        self._e_nkx.setText(str(self.cfg["dft"]["k_grid"][0]))
        self._e_nky.setText(str(self.cfg["dft"]["k_grid"][1]))
        self._e_dft_nx.setText(str(self.cfg["dft"]["output_grid"][0]))
        self._e_dft_ny.setText(str(self.cfg["dft"]["output_grid"][1]))

        lat = self.cfg["lattice"]
        self._e_a.setText(str(lat["a"]))
        self._e_b.setText(str(lat["b"]))
        self._e_c.setText(str(lat["c"]))
        self._e_alpha.setText(str(lat["alpha"]))
        self._e_beta.setText(str(lat["beta"]))
        self._e_gamma.setText(str(lat["gamma"]))

        cal = self.cfg["calibration"]
        self._e_kx_shift.setText(str(cal["kx_shift"]))
        self._e_ky_shift.setText(str(cal["ky_shift"]))
        self._e_kx_gs.setText(str(cal["kx_grid_shift"]))
        self._e_ky_gs.setText(str(cal["ky_grid_shift"]))

        # Clear and rebuild band rows
        for w in list(self._band_widgets):
            self._remove_band_row(w)
        for ib in range(len(self.cfg["mrf"]["bands"])):
            self._add_band_row()
        for ib, band in enumerate(self.cfg["mrf"]["bands"]):
            if ib < len(self._band_rows):
                if "from_vbm" in band:
                    self._band_rows[ib]["index"].setText(f"vbm:{band['from_vbm']}")
                else:
                    self._band_rows[ib]["index"].setText(str(band["index"]))
                self._band_rows[ib]["eta"].setText(str(band["eta"]))

        bsfi = self.cfg["mrf"]["bsfi"]
        self._e_bsfi_range.setText(str(bsfi["offset_range"]))
        self._e_bsfi_step.setText(str(bsfi["offset_step"]))
        self._e_bsfi_fine.setText(str(bsfi["fine_tune_range"]))
        self._e_bsfi_wcorr.setText(str(bsfi["weights"]["correlation"]))
        self._e_bsfi_wint.setText(str(bsfi["weights"]["intensity"]))
        self._e_bsfi_wsnr.setText(str(bsfi["weights"]["snr"]))
        self._e_bsfi_wridge.setText(str(bsfi["weights"].get("ridge", 0.5)))
        self._e_bsfi_wpath.setText(str(bsfi["weights"].get("path_ridge", 0.8)))
        self._e_bsfi_rsigma.setText(str(bsfi.get("ridge_sigma", 0.1)))

        self._e_smooth.setText(", ".join(str(x) for x in self.cfg["mrf"]["smooth_sigma"]))
        self._e_max_shift.setText(str(self.cfg["mrf"].get("max_shift", 10)))
        self._cb_alignment.setCurrentText(self.cfg["mrf"].get("alignment", "hsp"))
        self._cb_offset_mode.setCurrentText(self.cfg["mrf"].get("offset_mode", "per_band"))
        self._cb_occ.setCurrentText("true" if self.cfg["mrf"].get("occupied_only", True) else "false")
        self._cb_device.setCurrentText(self.cfg["mrf"].get("device", "auto"))
        method = self.cfg["mrf"].get("path_interp_method", "cubic")
        self._cb_path_interp.setCurrentText(method if method in ("cubic", "linear", "nearest") else "cubic")
        self._e_path_step.setText(str(self.cfg["mrf"].get("path_sample_step", 0.005)))

        pp = self.cfg["preprocessing"]
        self._e_out_path.setText(pp["output_path"])
        self._e_out_grid.setText(str(pp["output_grid"]))
        method = pp.get("method", "kdtree")
        self._cb_method.setCurrentText(method if method in ("kdtree", "quadrant", "direct") else "kdtree")
        self._e_kd_radius.setText(str(pp["kd_radius"]))
        self._e_stride.setText(str(pp["stride"]))
        self._e_nrot.setText(str(pp["n_rotations"]))
        self._cb_sort_axes.setChecked(pp.get("sort_axes", False))
        self._cb_auto_grid.setChecked(pp.get("auto_grid", False))
        self._cb_normalize.setChecked(pp.get("normalize", True))
        self._e_workers.setText(str(pp.get("workers", 0)))
        q = pp.get("quadrant", {})
        self._cb_flip_kx.setChecked(q.get("flip_kx", True))
        self._cb_flip_ky.setChecked(q.get("flip_ky", True))
        self._e_smooth_radius.setText(str(q.get("smooth_radius", 0.02)))

    # ── actions ─────────────────────────────────────────────────────

    def _on_load(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load config", self.config_path,
                                                      "YAML (*.yaml *.yml);;All (*)")
        if p:
            self.config_path = p
            self.cfg = load_config(p)
            self._path_edit.setText(p)
            self._load_to_ui()

    def _on_save(self):
        self._sync_all_from_ui()
        save_config(self.cfg, self.config_path)
        self._path_edit.setText(self.config_path)
        QtWidgets.QMessageBox.information(self, "Saved", f"Config saved to {self.config_path}")

    def _on_save_as(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save config as",
                                                      self.config_path,
                                                      "YAML (*.yaml);;All (*)")
        if p:
            self.config_path = p
            self._on_save()


# ── entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PIMRE Configuration Setup GUI")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Config file path")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    window = ConfigSetupWindow(args.config)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
