from algs.mandelbrot import Mandelbrot
from algs.buddhacu import CudaWorker


class Mandelcu(Mandelbrot):
    def __init__(self, iters=10000, gradient_type="color"):
        super().__init__(iters, gradient_type)
        self.worker = CudaWorker("mandel.out")

    def compute_iters(self, window):
        return self.worker.query(window, self.iters, 0)
