import numpy as np

from fractal import *
from utils import *

from algs.mandelbrot import calc_mandelbrot


class Buddhabrot(Fractal):
    progressive = ProgressiveType.SAMPLES

    def __init__(self, iters: int = 100):
        self.iters = iters

        self.batch_size = 1000
        self.result = None

    def new_window(self, window):
        self.result = window.blank_image(dtype=np.uint32)

    def render(self, window):
        assert self.result is not None

        # Sample normal dist of complex numbers
        c_values = np.random.normal(size=(self.batch_size, 2))
        c_values = c_values[:, 0] + c_values[:, 1] * 1j
        c_values = torch.tensor(c_values, dtype=torch.complex128, device=DEVICE)

        # Iterate mandelbrot to find which ones escape
        z_values = torch.zeros_like(c_values, device=DEVICE)
        in_mandelbrot = torch.ones(self.batch_size, dtype=torch.bool, device=DEVICE)
        for i in range(self.iters):
            z_values = z_values ** 2 + c_values
            in_mandelbrot = torch.logical_and(in_mandelbrot, torch.abs(z_values) <= 2)

        # Accumulate the ones that escape
        c_values = c_values[in_mandelbrot]
        z_values = torch.zeros_like(c_values, device=DEVICE)
        for i in range(self.iters):
            z_values = z_values ** 2 + c_values

            # TODO this is inefficient
            for z in z_values:
                x, y = window.coord_to_px(z.real, z.imag)
                if 0 <= x < window.res[0] and 0 <= y < window.res[1]:
                    self.result[y][x] += 1

        # Transform to image
        image = window.blank_image()
        max_val = np.max(self.result)
        image[:, :, :] = np.log(self.result + 1) / np.log(max_val + 1) * 255
        return image
