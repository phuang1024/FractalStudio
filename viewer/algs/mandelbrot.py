import time

import numpy as np

from fractal import *
from utils import *


class Mandelbrot(Fractal):
    progressive = ProgressiveType.UPRES

    def __init__(self, iters: int = 100, gradient_type: str = "color"):
        """
        gradient_type:
            "none": Black and white.
            "gray": Gray scale.
            "color": Color gradient.
        """
        self.iters = iters
        self.gradient_type = gradient_type

        self.render_time = 0

    def compute_iters(self, window):
        result = calc_mandelbrot(window, self.iters)
        result = result.cpu().numpy()
        return result

    def render(self, window):
        time_start = time.time()

        result = self.compute_iters(window)

        if self.gradient_type == "none":
            image = window.blank_image()
            image[result == 0] = 255
        else:
            # Normalize to 0 to 255
            image = (result / self.iters * 255).astype(np.uint8)
            if self.gradient_type == "gray":
                image = np.stack([image, image, image], axis=-1)

        self.render_time = time.time() - time_start

        return image

    def get_stats(self):
        return [
            f"Render time: {self.render_time*1000:.3f}ms"
        ]


def calc_mandelbrot(window, iters):
    """
    Returns a 2D int array same shape as window.res
    Each element is the number of iterations it took for the corresponding
    complex number to escape the mandelbrot set.
    0 means it didn't escape (i.e. it's in the set).
    """
    coords = window_to_coords(window).to(DEVICE)
    z_values = torch.zeros_like(coords, device=DEVICE)
    # Number of iters it took to get abs(z) > 2
    result = torch.zeros_like(coords, dtype=torch.int32, device=DEVICE)

    for i in range(iters):
        z_values = z_values ** 2 + coords
        result[torch.logical_and(torch.abs(z_values) > 2, result == 0)] = i

    return result
