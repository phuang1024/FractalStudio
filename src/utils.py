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
    pos: tuple[float, float] = (0, 0)
    """Position of center XY in units."""
    scale: float = 1
    """Number of units the X direction spans."""


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
