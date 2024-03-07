import numpy as np

from fractal import *
from utils import *


class Mandelbrot(Fractal):
    progressive = ProgressiveType.UPRES

    def __init__(self, iters: int = 100, draw_gradient: bool = True):
        self.iters = iters
        self.draw_gradient = draw_gradient

    def render(self, window):
        result = calc_mandelbrot(window, self.iters)
        result = result.cpu().numpy()

        image = np.zeros(window.res[::-1], dtype=np.uint8)

        if self.draw_gradient:
            # Normalize to 0-1
            result = result / self.iters
            # Apply gradient
            result = np.log(result + 1)
            result = result * 255
            result = result.astype(np.uint8)
            image = result
        else:
            image[result == 0] = 255

        return image


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
