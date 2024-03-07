__all__ = (
    "Fractal",
    "ProgressiveType",
)

"""
Fractal algorithm base class.
"""

from enum import Enum
from typing import Sequence

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
    """
    Fractal algorithm base class.

    Progressive rendering paradigms:

    None:
    Do the full render in ``render`` and return the image.

    Upres:
    ``render`` is called multiple times at increasing resolutions.
    Do the full render and return the image each time, at the requested resolution.

    Samples:
    Do reset procedures in ``new_window``.
    ``render`` is called multiple times at the same resolution, which may possibly
    represent adding samples to the current viewing window (as opposed to starting a new render).
    Each time, return the accumulated image so far.
    """

    progressive = ProgressiveType.NONE
    iter_num: int = 0
    """Iter number since last view window change. Set by render worker."""

    def __init__(self, **kwargs):
        pass

    def new_window(self, window: Window):
        """
        Called when a new viewing window (i.e. x/y bounds and resolution) is requested.

        Do any reset procedures here.

        It is guarenteed that this is called before the first call to ``render``.
        """

    def render(self, window: Window, **kwargs) -> np.ndarray:
        """
        Return np array of shape (res[1], res[0], 3), dtype uint8.
        """
        raise NotImplementedError

    def get_stats(self) -> Sequence[str]:
        """
        Return dict of stats to display.
        """
        return []
