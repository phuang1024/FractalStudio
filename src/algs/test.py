"""
Test rendering algorithm.
"""

import numpy as np

from fractal import *


class TestFractal(Fractal):
    def render(self, resolution):
        img = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        img[..., 0] = 255
        return img
