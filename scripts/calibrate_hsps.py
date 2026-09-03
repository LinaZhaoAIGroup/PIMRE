#!/usr/bin/env python3
"""Interactive BZ high-symmetry point calibration with overlays.

Displays ARPES data (or DFT band map) with draggable overlays:
  - Circle at |Γ-K| radius
  - Regular hexagon (K vertices + M edge midpoints)
  - Orthogonal cross (K-K and M-M directions)

User adjusts rotation and scale to match the BZ, then saves to config.

Usage:
    # ARPES experimental data
    uv run python scripts/calibrate_hsps.py

    # DFT band map
    uv run python scripts/calibrate_hsps.py --mode dft --band-map test/band_map.h5

    uv run python scripts/calibrate_hsps.py --config configs/pimre_config.yaml
"""

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pimre.config import load_config, save_config
from pimre.dft.reader import load_band_map_any
from pimre.kpath.symmetry import (
    find_hsps_robust,
    lattice_to_reciprocal,
)
from pimre.utils.io import loadHDF

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "configs", "pimre_config.yaml")
CRYSTAL_DATA = [5.8077, 5.8077, 9.1297, 90, 90, 120]


class HspsCalibrationCanvas(FigureCanvas):
    """Matplotlib canvas with BZ overlay controls."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumSize(600, 600)

        self.data = None
        self.kx = None
        self.ky = None
        self.G = None
        self.crystal_data = None
        self.K_dist = None
        self.M_dist = None

        self.theta = 0.0
        self.scale = 1.0
        self.layer = 0
        self.show_circle = True
        self.show_hexagon = True
        self.show_cross = True
        self.show_data = True

        self._im = None
        self._circle = None
        self._hex_lines = []
        self._hex_scatter_k = None
        self._hex_scatter_m = None
        self._cross_lines = []

    def set_data(self, data, kx, ky, crystal_data):
        self.data = data
        self.kx = kx
        self.ky = ky
        self.crystal_data = crystal_data
        k_K, k_M = lattice_to_reciprocal(*crystal_data)
        self.K_dist = float(np.linalg.norm(k_K[:2]))
        self.M_dist = float(np.linalg.norm(k_M[:2]))
        self.G = (np.argmin(np.abs(kx)), np.argmin(np.abs(ky)))

    def set_params(self, theta, scale, layer):
        self.theta = theta
        self.scale = scale
        self.layer = min(layer, self.data.shape[0] - 1) if self.data is not None else 0
        self._draw()

    def _draw(self):
        self.ax.clear()
        if self.data is None:
            self.fig.canvas.draw_idle()
            return

        gx, gy = self.kx[self.G[0]], self.ky[self.G[1]]

        if self.show_data:
            self._im = self.ax.imshow(
                self.data[self.layer, :, :].T,
                aspect="auto", origin="lower", cmap="plasma",
                extent=[self.kx[0], self.kx[-1], self.ky[0], self.ky[-1]],
            )

        r = self.K_dist * self.scale
        r_m = self.M_dist * self.scale
        theta_rad = np.deg2rad(self.theta)

        if self.show_circle:
            circle = plt.Circle((gx, gy), r, fill=False, color="cyan",
                                linewidth=1.5, linestyle="--", alpha=0.8)
            self.ax.add_patch(circle)

        if self.show_hexagon:
            K_pts = []
            M_pts = []
            for i in range(6):
                ak = theta_rad + i * np.pi / 3
                am = theta_rad + np.pi / 6 + i * np.pi / 3
                K_pts.append((gx + r * np.cos(ak), gy + r * np.sin(ak)))
                M_pts.append((gx + r_m * np.cos(am), gy + r_m * np.sin(am)))

            K_arr = np.array(K_pts)
            M_arr = np.array(M_pts)
            self.ax.scatter(K_arr[:, 0], K_arr[:, 1], marker="D", c="magenta",
                            s=60, edgecolors="white", linewidths=0.5, zorder=5)
            self.ax.scatter(M_arr[:, 0], M_arr[:, 1], marker="s", c="cyan",
                            s=50, edgecolors="white", linewidths=0.5, zorder=5)

            for i in range(6):
                j = (i + 1) % 6
                self.ax.plot(
                    [K_arr[i, 0], K_arr[j, 0]], [K_arr[i, 1], K_arr[j, 1]],
                    "w-", linewidth=1.5, alpha=0.6, zorder=4,
                )

        if self.show_cross:
            for angle in [self.theta + 30, self.theta + 120]:
                rad = np.deg2rad(angle)
                dx = r * 1.5 * np.cos(rad)
                dy = r * 1.5 * np.sin(rad)
                self.ax.plot([gx - dx, gx + dx], [gy - dy, gy + dy],
                             "white", linewidth=1.0, linestyle=":", alpha=0.5, zorder=3)

        self.ax.scatter(gx, gy, marker="o", c="white", s=100,
                        edgecolors="black", linewidths=1.0, zorder=6)
        self.ax.annotate("Γ", (gx, gy), textcoords="offset points",
                         xytext=(5, 5), fontsize=12, color="white",
                         bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.6))

        self.ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)")
        self.ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)")
        self.ax.set_title(f"Layer {self.layer}  θ={self.theta:.1f}°  s={self.scale:.3f}")
        self.fig.canvas.draw_idle()


class CalibrateHspsWindow(QtWidgets.QMainWindow):
    def __init__(self, config_path, mode="arpes", band_map_path=None):
        super().__init__()
        self.config_path = config_path
        self.mode = mode
        self.band_map_path = band_map_path
        self.cfg = load_config(config_path)
        self.crystal = CRYSTAL_DATA

        self._load_data()
        self._build_ui()
        self._load_calibration()
        self._update_view()

        self.setWindowTitle("BZ High-Symmetry Point Calibration")
        self.resize(900, 850)

    def _load_data(self):
        if self.mode == "dft":
            E_dft, evb, ecb, kx, ky = load_band_map_any(
                self.band_map_path,
                drop_top_bands=self.cfg.get("dft", {}).get("drop_top_bands"))
            self.data = E_dft
            self.n_bands = E_dft.shape[0]
            self.E_grid = None
        else:
            pp = self.cfg["preprocessing"]
            prep_path = pp.get("output_path", "test/exp_preprocessed.h5")
            if not os.path.isabs(prep_path):
                prep_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", prep_path)
            if not os.path.exists(prep_path):
                raise FileNotFoundError(
                    f"Preprocessed data not found: {prep_path}\n"
                    "Run preprocessing first: uv run python scripts/preprocess_exp.py")
            data = loadHDF(prep_path)
            E = data["E"]
            kx = data["kx"]
            ky = data["ky"]
            V = data["V"]
            # Exact-shape matching: the saved layout is (E, kx, ky); a
            # shape[0]-based heuristic mis-transposes square grids.
            if V.shape == (kx.size, ky.size, E.size):
                V = np.transpose(V, (2, 0, 1))
            elif V.shape != (E.size, kx.size, ky.size):
                raise ValueError(
                    f"Unexpected data layout {V.shape} for axes E={E.size}, "
                    f"kx={kx.size}, ky={ky.size} in {prep_path}")
            self.data = V
            self.E_grid = E
            self.n_bands = V.shape[0]

        self.kx = kx
        self.ky = ky

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.canvas = HspsCalibrationCanvas()
        self.canvas.set_data(self.data, self.kx, self.ky, self.crystal)
        layout.addWidget(self.canvas)

        ctrl = QtWidgets.QWidget()
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(10, 5, 10, 5)

        # Layer slider
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Layer:"))
        self.layer_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.layer_slider.setRange(0, self.n_bands - 1)
        self.layer_slider.setValue(0)
        self.layer_slider.valueChanged.connect(self._on_layer)
        row.addWidget(self.layer_slider)
        self.layer_label = QtWidgets.QLabel("0")
        self.layer_label.setFixedWidth(40)
        row.addWidget(self.layer_label)
        ctrl_layout.addLayout(row)

        # Rotation slider
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Rotation:"))
        self.rot_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rot_slider.setRange(0, 600)
        self.rot_slider.setValue(0)
        self.rot_slider.setTickInterval(60)
        self.rot_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.rot_slider.valueChanged.connect(self._on_rotation)
        row2.addWidget(self.rot_slider)
        self.rot_label = QtWidgets.QLabel("0.0°")
        self.rot_label.setFixedWidth(60)
        row2.addWidget(self.rot_label)
        ctrl_layout.addLayout(row2)

        # Scale slider
        row3 = QtWidgets.QHBoxLayout()
        row3.addWidget(QtWidgets.QLabel("Scale:"))
        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale_slider.setRange(50, 150)
        self.scale_slider.setValue(100)
        self.scale_slider.setTickInterval(5)
        self.scale_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.scale_slider.valueChanged.connect(self._on_scale)
        row3.addWidget(self.scale_slider)
        self.scale_label = QtWidgets.QLabel("1.00")
        self.scale_label.setFixedWidth(60)
        row3.addWidget(self.scale_label)
        ctrl_layout.addLayout(row3)

        # Checkboxes
        row4 = QtWidgets.QHBoxLayout()
        self.cb_circle = QtWidgets.QCheckBox("Circle")
        self.cb_circle.setChecked(True)
        self.cb_circle.toggled.connect(self._on_check)
        self.cb_hexagon = QtWidgets.QCheckBox("Hexagon")
        self.cb_hexagon.setChecked(True)
        self.cb_hexagon.toggled.connect(self._on_check)
        self.cb_cross = QtWidgets.QCheckBox("Cross")
        self.cb_cross.setChecked(True)
        self.cb_cross.toggled.connect(self._on_check)
        row4.addWidget(self.cb_circle)
        row4.addWidget(self.cb_hexagon)
        row4.addWidget(self.cb_cross)
        row4.addStretch()
        ctrl_layout.addLayout(row4)

        # Buttons
        row5 = QtWidgets.QHBoxLayout()
        auto_btn = QtWidgets.QPushButton("Auto-detect")
        auto_btn.clicked.connect(self._auto_detect)
        row5.addWidget(auto_btn)
        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.clicked.connect(self._reset)
        row5.addWidget(reset_btn)
        save_btn = QtWidgets.QPushButton("Save to Config")
        save_btn.clicked.connect(self._save)
        save_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        row5.addWidget(save_btn)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.close)
        row5.addWidget(cancel_btn)
        ctrl_layout.addLayout(row5)

        layout.addWidget(ctrl)

    def _hsps_key(self):
        """Config key holding the HSP calibration for the current mode."""
        return "dft_hsps" if self.mode == "dft" else "hsps"

    def _load_calibration(self):
        hsps_cfg = self.cfg["calibration"].get(self._hsps_key(), {})
        if hsps_cfg.get("manual", False):
            self.rot_slider.setValue(int(hsps_cfg.get("rotation_angle", 0.0) * 10))
            self.scale_slider.setValue(int(hsps_cfg.get("scale", 1.0) * 100))

    def _on_layer(self, val):
        self.layer_label.setText(str(val))
        self._update_view()

    def _on_rotation(self, val):
        self.rot_label.setText(f"{val / 10:.1f}°")
        self._update_view()

    def _on_scale(self, val):
        self.scale_label.setText(f"{val / 100:.2f}")
        self._update_view()

    def _on_check(self):
        self._update_view()

    def _update_view(self):
        self.canvas.show_circle = self.cb_circle.isChecked()
        self.canvas.show_hexagon = self.cb_hexagon.isChecked()
        self.canvas.show_cross = self.cb_cross.isChecked()
        self.canvas.set_params(
            self.rot_slider.value() / 10.0,
            self.scale_slider.value() / 100.0,
            self.layer_slider.value(),
        )

    def _auto_detect(self):
        if self.mode == "dft":
            QtWidgets.QMessageBox.information(
                self, "Auto-detect",
                "Auto-detect is not available for DFT data.\n"
                "DFT HSPs are computed analytically from the k-grid.")
            return

        try:
            result = find_hsps_robust(
                self.data, self.kx, self.ky, self.crystal, self.E_grid)
            self.rot_slider.setValue(int(result.rotation_angle * 10))
            self.layer_slider.setValue(result.best_layer)
            self.layer_label.setText(str(result.best_layer))
            self._update_view()
            QtWidgets.QMessageBox.information(
                self, "Auto-detect Result",
                f"Best layer: {result.best_layer}\n"
                f"Rotation: {result.rotation_angle:.1f}°\n"
                f"Score: {result.registration_score:.4f}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def _reset(self):
        self.rot_slider.setValue(0)
        self.scale_slider.setValue(100)
        self.layer_slider.setValue(0)
        self._update_view()

    def _save(self):
        key = self._hsps_key()
        self.cfg["calibration"][key] = {
            "manual": True,
            "rotation_angle": self.rot_slider.value() / 10.0,
            "scale": self.scale_slider.value() / 100.0,
        }
        save_config(self.cfg, self.config_path)
        QtWidgets.QMessageBox.information(
            self, "Saved",
            f"Calibration saved to {self.config_path}\n\n"
            f"Mode: {self.mode} (config key: calibration.{key})\n"
            f"Rotation: {self.rot_slider.value() / 10:.1f}°\n"
            f"Scale: {self.scale_slider.value() / 100:.2f}")


def main():
    parser = argparse.ArgumentParser(description="BZ HSP calibration")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--mode", default="arpes", choices=["arpes", "dft"])
    parser.add_argument("--band-map", default=None,
                        help="Path to band_map.h5 (dft mode)")
    args = parser.parse_args()

    if args.mode == "dft" and not args.band_map:
        parser.error("--band-map is required for dft mode")

    app = QtWidgets.QApplication(sys.argv)
    window = CalibrateHspsWindow(args.config, args.mode, args.band_map)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
