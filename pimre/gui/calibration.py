"""Interactive calibration widgets for Angle and Momentum space."""

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from pimre.utils.interaction import DraggableHLine, DraggableVLine


class GammaCalibrator:
    """Interactive Gamma-point calibration in angle space."""

    def __init__(self, bands, kx_angle, ky_angle):
        self.bands = bands
        self.kx_angle = kx_angle
        self.ky_angle = ky_angle
        self.n_layers = bands.shape[0]
        self.kx_s = (kx_angle[0] + kx_angle[-1]) / 2
        self.ky_s = (ky_angle[0] + ky_angle[-1]) / 2
        self._build_ui()

    def _build_ui(self):
        self.fig = plt.figure("Stage 1: Gamma Point Calibration (Angle Space)", figsize=(10, 9))
        self.ax = self.fig.add_axes([0.12, 0.22, 0.83, 0.72])
        self.ax_slider = self.fig.add_axes([0.15, 0.08, 0.70, 0.04])

        self.layer = self.n_layers // 2
        self.im = self.ax.imshow(
            self.bands[self.layer],
            aspect="auto",
            extent=[self.ky_angle[0], self.ky_angle[-1],
                    self.kx_angle[0], self.kx_angle[-1]],
            cmap="plasma", origin="lower",
        )
        self.ax.set_xlabel("ky angle (deg)")
        self.ax.set_ylabel("kx angle (deg)")
        self.ax.set_title(f"Layer {self.layer} / {self.n_layers-1}")
        self.ax.grid(True, alpha=0.3)
        self.fig.canvas.draw()

        self.vline = DraggableVLine(self.ax, x=self.ky_s, color="red", linestyle="--", linewidth=2)
        self.hline = DraggableHLine(self.ax, y=self.kx_s, color="green", linestyle="--", linewidth=2)

        self.slider = Slider(self.ax_slider, "Energy layer", 0, self.n_layers - 1,
                             valinit=self.layer, valstep=1)
        self.slider.on_changed(self._on_slider)

    def _on_slider(self, val):
        self.layer = int(val)
        self.im.set_data(self.bands[self.layer])
        self.ax.set_title(f"Layer {self.layer} / {self.n_layers - 1}")
        self.fig.canvas.draw_idle()

    def run(self):
        print("\n" + "=" * 60)
        print("STAGE 1 – Angle-space Gamma calibration")
        print("=" * 60)
        print("  • Drag the red   vertical line to Gamma's ky position")
        print("  • Drag the green  horizontal line to Gamma's kx position")
        print("  • Use the slider to find the clearest layer")
        print("  • Close the window when done")
        self.fig.canvas.manager.window.raise_()
        plt.show(block=True)
        self.kx_s = self.hline.y
        self.ky_s = self.vline.x
        print(f"  → kx_shift = {self.kx_s:.4f}  (horizontal line)")
        print(f"  → ky_shift = {self.ky_s:.4f}  (vertical line)")
        return self.kx_s, self.ky_s


class GridCalibrator:
    """Interactive Gamma-point calibration in momentum space."""

    def __init__(self, E_mon_layer, kx, ky):
        self.E_mon_layer = E_mon_layer
        self.kx = kx
        self.ky = ky
        self.kx_gs = (kx[0] + kx[-1]) / 2
        self.ky_gs = (ky[0] + ky[-1]) / 2
        self._build_ui()

    def _build_ui(self):
        self.fig = plt.figure("Stage 2: Grid Shift Calibration (Momentum Space)", figsize=(9, 8))
        self.ax = self.fig.add_axes([0.12, 0.10, 0.83, 0.85])

        self.im = self.ax.imshow(
            self.E_mon_layer,
            aspect="auto",
            extent=[self.ky[0], self.ky[-1], self.kx[0], self.kx[-1]],
            cmap="plasma", origin="lower",
        )
        self.ax.set_xlabel(r"$k_y$ ($\AA^{-1}$)")
        self.ax.set_ylabel(r"$k_x$ ($\AA^{-1}$)")
        self.ax.set_title("Drag lines to centre Γ in momentum space")
        self.ax.grid(True, alpha=0.3)
        self.fig.canvas.draw()

        self.vline = DraggableVLine(self.ax, x=self.ky_gs, color="red", linestyle="--", linewidth=2)
        self.hline = DraggableHLine(self.ax, y=self.kx_gs, color="green", linestyle="--", linewidth=2)

    def run(self):
        print("\n" + "=" * 60)
        print("STAGE 2 – Momentum-space grid calibration")
        print("=" * 60)
        print("  • Drag the lines to centre the Gamma point in momentum space")
        print("  • Close the window when done")
        self.fig.canvas.manager.window.raise_()
        plt.show(block=True)
        self.kx_gs = self.hline.y
        self.ky_gs = self.vline.x
        print(f"  → kx_grid_shift = {self.kx_gs:.4f}  (horizontal line)")
        print(f"  → ky_grid_shift = {self.ky_gs:.4f}  (vertical line)")
        return self.kx_gs, self.ky_gs
