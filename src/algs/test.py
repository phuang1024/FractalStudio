"""
Test rendering algorithms.
"""

import time

import cv2
import numpy as np

from fractal import *


class SolidColor(Fractal):
    def render(self, window):
        img = np.zeros((window.res[1], window.res[0], 3), dtype=np.uint8)
        img[..., 0] = 255
        return img


class ImageResize(Fractal):
    progressive = ProgressiveType.UPRES

    def __init__(self, path: str):
        self.path = path
        self.image = cv2.imread(path)[..., ::-1]

    def render(self, window):
        # Simulate long render time
        time.sleep(0.1)
        return cv2.resize(self.image, window.res)
