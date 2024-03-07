import cv2
import numpy as np

from fractal import *
from utils import *

from algs.mandelbrot import calc_mandelbrot


class Buddhabrot(Fractal):
    progressive = ProgressiveType.SAMPLES

    def __init__(self, iters: int = 100, batch_size: int = 100000, hue: float = 0.69):
        self.iters = iters
        self.batch_size = batch_size
        self.hue = hue

        self.exposure = 1.5
        self.result = None

    def new_window(self, window):
        self.result = window.blank_image(dtype=np.uint32)[..., 0]

    def render(self, window):
        assert self.result is not None

        # Render
        calc_buddhabrot(self.iters, self.batch_size, window, self.result)

        # Transform to image
        intensity = self.result
        intensity = np.clip(intensity, 0, np.max(intensity) * 0.8)
        intensity = intensity / np.max(intensity)
        intensity = np.tanh(intensity * self.exposure)
        intensity = intensity / np.max(intensity)

        s = 1 - intensity
        v = intensity
        hsv_image = np.stack([np.ones_like(intensity) * self.hue, s, v], axis=-1)
        hsv_image = (hsv_image * 255).astype(np.uint8)
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)[..., ::-1]

        return image


def calc_buddhabrot(iters, batch_size, window, result):
    """
    Add results in place to ``result``.
    """
    # Sample normal dist of complex numbers
    c_values = np.random.normal(size=(batch_size, 2))
    c_values = c_values[:, 0] + c_values[:, 1] * 1j
    c_values = torch.tensor(c_values, dtype=torch.complex128, device=DEVICE)

    # Iterate mandelbrot to find which ones escape
    z_values = torch.zeros_like(c_values, device=DEVICE)
    not_in_set = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
    for i in range(iters):
        z_values = z_values ** 2 + c_values
        not_in_set = torch.logical_or(not_in_set, torch.abs(z_values) > 2)

    # Accumulate the ones that escape
    c_values = c_values[not_in_set]
    z_values = torch.zeros_like(c_values, device=DEVICE)
    for i in range(iters):
        z_values = z_values ** 2 + c_values

        # Convert z_values to pixel coords
        x_scale = window.scale
        y_scale = window.scale * window.res[1] / window.res[0]
        coords = torch.empty((z_values.shape[0], 2), dtype=torch.int64, device=DEVICE)
        coords[:, 0] = ((z_values.real - window.pos[0]) / x_scale * window.res[0] + window.res[0] / 2).to(torch.int64)
        coords[:, 1] = ((z_values.imag - window.pos[1]) / y_scale * window.res[1] + window.res[1] / 2).to(torch.int64)

        in_bounds = torch.logical_and(
            torch.logical_and(coords[:, 0] >= 0, coords[:, 0] < window.res[0]),
            torch.logical_and(coords[:, 1] >= 0, coords[:, 1] < window.res[1])
        )
        coords = coords[in_bounds]
        coords = coords.cpu().numpy()
        result[coords[:, 1], coords[:, 0]] += 1
