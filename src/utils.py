import time
from dataclasses import dataclass

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Clock:
    def __init__(self, fps):
        self.last_tick = 0
        self.fps = fps

    def tick(self) -> bool:
        if time.time() - self.last_tick > 1 / self.fps:
            self.last_tick = time.time()
            return True
        return False


@dataclass
class Window:
    """
    Window scale, coords, etc.
    """

    res: tuple[int, int] = (1280, 720)
    """X, Y resolution."""
    pos: tuple[float, float] = (0, 0)
    """Position of center XY in units."""
    scale: float = 5
    """Number of units the X direction spans."""

    def blank_image(self, dtype=np.uint8) -> np.ndarray:
        """
        Return image of shape (res[1], res[0], 3) filled with 0.
        """
        return np.zeros((*self.res[::-1], 3), dtype=dtype)

    def coord_to_px(self, x, y):
        """
        Convert from window coords to pixel coords.
        """
        x_px = (x - self.pos[0]) / self.scale * self.res[0] + self.res[0] / 2
        y_px = (y - self.pos[1]) / self.scale * self.res[1] + self.res[1] / 2
        x_px = int(x_px)
        y_px = int(y_px)
        return x_px, y_px


@dataclass
class ViewerState:
    """
    Communication between main viewer and worker thread.
    """

    run: bool = True
    window_changed: int = 0
    """This is incremented by 1 every time user changes window, to trigger rerender."""

    render_result: np.ndarray = None

    window: Window = Window()


def window_to_coords(window: Window, dtype=torch.complex128) -> torch.Tensor:
    """
    Generate grid of coordinates corresponding to each pixel in the window.

    i.e. ret[y][x] = x + y*j

    ret.shape = (window.res[1], window.res[0])

    On default device.
    """
    x_size = window.scale
    y_size = window.scale * window.res[1] / window.res[0]
    x = torch.linspace(
        window.pos[0] - x_size / 2,
        window.pos[0] + x_size / 2,
        window.res[0]
    )
    y = torch.linspace(
        window.pos[1] - y_size / 2,
        window.pos[1] + y_size / 2,
        window.res[1]
    )
    y, x = torch.meshgrid(y, x, indexing="ij")
    x = x.to(dtype)
    y = y.to(dtype)
    return x + y * 1j
