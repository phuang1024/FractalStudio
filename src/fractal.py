__all__ = (
    "Fractal",
    "ProgressiveType",
)

"""
Fractal algorithm base class.
"""

from enum import Enum

import numpy as np

from utils import Window


class ProgressiveType(Enum):
    """
    How to progressively render.

    NONE: No progressive; wait until render is complete, then display.
    UPRES: Progressive upres; start at low res, then increase resolution.
    SAMPLES: Progressive samples; maintain full resolution, but increase samples over time.
    """
    NONE = 0
    UPRES = 1
    SAMPLES = 2


class Fractal:
    progressive = ProgressiveType.NONE

    def __init__(self, **kwargs):
        pass

    def new_window(self):
        """
        Called when a new viewing window (i.e. x/y bounds and resolution) is requested.

        Do any reset procedures here.
        """

    def render(self, window: Window, **kwargs) -> np.ndarray:
        """
        Return np array of shape (res[1], res[0], 3), dtype uint8.
        """
        raise NotImplementedError
