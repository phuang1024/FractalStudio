"""
Test rendering algorithms.
"""

import time

import cv2
import numpy as np

from fractal import *


class SolidColor(Fractal):
    def render(self, resolution):
        img = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        img[..., 0] = 255
        return img


class ImageResize(Fractal):
    progressive = ProgressiveType.UPRES

    def __init__(self, path: str):
        self.path = path
        self.image = cv2.imread(path)[..., ::-1]

    def render(self, resolution):
        # Simulate long render time
        time.sleep(0.1)
        return cv2.resize(self.image, resolution)
