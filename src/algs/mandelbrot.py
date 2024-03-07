import numpy as np

from fractal import *
from utils import *


class Mandelbrot(Fractal):
    progressive = ProgressiveType.UPRES

    def __init__(self, iters: int = 100):
        self.iters = iters

    def render(self, window):
        coords = window_to_coords(window).to(DEVICE)
        z_values = torch.zeros_like(coords, device=DEVICE)
        # Number of iters it took to get abs(z) > 2
        result = torch.zeros_like(coords, dtype=torch.int32, device=DEVICE)

        for i in range(self.iters):
            z_values = z_values ** 2 + coords
            result[torch.logical_and(torch.abs(z_values) > 2, result == 0)] = i

        result = result.cpu().numpy()
        image = np.zeros(window.res[::-1], dtype=np.uint8)
        image[result == 0] = 255
        return image
