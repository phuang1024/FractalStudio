"""
CUDA accelerated counterpart to nebulabrot.
"""

from algs.cudabrot import CudaWorker
from algs.nebulabrot import Nebulabrot


class Cubulabrot(Nebulabrot):
    def __init__(
            self,
            iters_r: int = 1000,
            iters_g: int = 500,
            iters_b: int = 50,
            batch_size: int = int(1e4)
        ):
        super().__init__(iters_r, iters_g, iters_b, batch_size)
        self.worker = CudaWorker()

    def render_samples(self, window, batch_size):
        for i in range(3):
            data = self.worker.query(window, self.iters[i], batch_size)
            self.result[..., i] += data
            self.samples += batch_size * self.worker.num_threads
