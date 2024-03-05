import time
from dataclasses import dataclass

import numpy as np


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
class ViewerState:
    """
    Communication between main viewer and worker thread.
    """

    run: bool = True

    render_result: np.ndarray = None

    res: tuple[int, int] = (1280, 720)
    pos: tuple[float, float] = (0, 0)
    """Position of top left XY in units."""
    scale: float = 1
    """Number of units the X direction spans."""
