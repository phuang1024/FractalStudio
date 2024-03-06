import numpy as np

from fractal import *


class Mandelbrot(Fractal):
    progressive = ProgressiveType.UPRES

    def __init__(self, iters: int = 100):
        self.iters = iters

    def render(self, window):
        pass
