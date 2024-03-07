import cv2
import numpy as np

from fractal import *
from utils import *

from algs.buddhabrot import calc_buddhabrot, result_to_image


class Nebulabrot(Fractal):
    progressive = ProgressiveType.SAMPLES

    def __init__(
            self,
            iters_r: int = 5000,
            iters_g: int = 500,
            iters_b: int = 50,
            batch_size: int = int(1e5)
        ):
        self.iters_r = iters_r
        self.iters_g = iters_g
        self.iters_b = iters_b
        self.batch_size = batch_size

        self.exposure = 1.5
        self.result = None

        self.iters = [iters_r, iters_g, iters_b]

    def new_window(self, window):
        self.result = window.blank_image(dtype=np.uint32)

    def render(self, window):
        assert self.result is not None

        # Render
        batch_size = self.batch_size
        if self.iter_num < 5:
            batch_size = self.batch_size // 10
        for i in range(3):
            calc_buddhabrot(self.iters[i], batch_size, window, self.result[..., i])

        image = np.zeros_like(self.result, dtype=np.uint8)
        for i in range(3):
            intensity = result_to_image(self.result[..., i], self.exposure)
            s = 1 - intensity
            v = intensity
            hsv_image = np.stack([np.full_like(intensity, 0), s, v], axis=-1)
            hsv_image = (hsv_image * 255).astype(np.uint8)
            image[..., i] = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)[..., -1]

        return image
