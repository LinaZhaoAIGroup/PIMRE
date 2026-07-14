"""Interactive matplotlib utilities: draggable lines, index tracker."""

import numpy as np
import matplotlib.pyplot as plt


class DraggableVLine:
    """A draggable vertical line on a matplotlib Axes."""

    def __init__(self, ax, x=0, **kwargs):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.line = ax.axvline(x, **kwargs)
        self.x = x
        self.txt = ax.text(
            x, 1.02, f"x={x:.2f}", transform=ax.get_xaxis_transform(), horizontalalignment="center"
        )
        self.press = None
        self.connect()

    def connect(self):
        self.cid_press = self.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.line.axes:
            return
        contains, _ = self.line.contains(event)
        if not contains:
            return
        self.press = self.line.get_xdata()[0], event.xdata

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.line.axes:
            return
        x0, xpress = self.press
        dx = event.xdata - xpress
        self.x = x0 + dx
        self.line.set_xdata([self.x, self.x])
        self.txt.set_x(self.x)
        self.txt.set_text(f"x={self.x:.2f}")
        self.canvas.draw_idle()

    def on_release(self, event):
        self.press = None
        print(f"Final vertical line x-value: {self.x:.2f}")

    def disconnect(self):
        self.canvas.mpl_disconnect(self.cid_press)
        self.canvas.mpl_disconnect(self.cid_release)
        self.canvas.mpl_disconnect(self.cid_motion)


class DraggableHLine:
    """A draggable horizontal line on a matplotlib Axes."""

    def __init__(self, ax, y=0, **kwargs):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.line = ax.axhline(y, **kwargs)
        self.y = y
        self.txt = ax.text(
            1.02, y, f"y={y:.2f}", transform=ax.get_yaxis_transform(), verticalalignment="center"
        )
        self.press = None
        self.connect()

    def connect(self):
        self.cid_press = self.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.line.axes:
            return
        contains, _ = self.line.contains(event)
        if not contains:
            return
        self.press = self.line.get_ydata()[0], event.ydata

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.line.axes:
            return
        y0, ypress = self.press
        dy = event.ydata - ypress
        self.y = y0 + dy
        self.line.set_ydata([self.y, self.y])
        self.txt.set_y(self.y)
        self.txt.set_text(f"y={self.y:.2f}")
        self.canvas.draw_idle()

    def on_release(self, event):
        self.press = None
        print(f"Final horizontal line y-value: {self.y:.2f}")

    def disconnect(self):
        self.canvas.mpl_disconnect(self.cid_press)
        self.canvas.mpl_disconnect(self.cid_release)
        self.canvas.mpl_disconnect(self.cid_motion)


def IndexTracker(data, dim=0, initial_layer=0):
    """Display a 3D data array with a slider to scroll through layers.

    Parameters
    ----------
    data : ndarray
        3D data array.
    dim : int
        Dimension of the data to iterate over.
    initial_layer : int
        Initial layer index to display.
    """
    from matplotlib.widgets import Slider

    if dim != 0:
        data = np.moveaxis(data, dim, 0)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    im = ax.imshow(data[initial_layer], aspect="auto", origin="lower")
    ax.set_title(f"Layer {initial_layer}")
    ax.autoscale(True)

    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(
        ax=ax_slider,
        label="Layer",
        valmin=0,
        valmax=data.shape[0] - 1,
        valinit=initial_layer,
        valstep=1,
    )

    def update(val):
        layer_index = int(slider.val)
        im.set_data(data[layer_index])
        ax.set_title(f"Layer {layer_index}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()