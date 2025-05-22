"""
CUDA accelerated buddhabrot rendering, optimized for performance.
"""

import os
from subprocess import Popen, PIPE

import numpy as np

from algs.buddhabrot import Buddhabrot

PARENT = os.path.dirname(os.path.abspath(__file__))


class CudaWorker:
    """
    Wrapper around CUDA worker subprocess.
    """

    num_threads = 64 * 128

    def __init__(self, exe_name):
        self.process = Popen([os.path.join(PARENT, exe_name)], stdin=PIPE, stdout=PIPE)

    def query(self, window, iters, batch_size):
        assert self.process.returncode is None

        self.process.stdin.write(
            f"{window.res[0]} {window.res[1]} {iters} {batch_size} {window.xmin} {window.xmax} {window.ymin} {window.ymax}\n".encode()
        )
        self.process.stdin.flush()
        # Read result from process (int32).
        data = self.process.stdout.read(window.res[0] * window.res[1] * 4)
        data = np.frombuffer(data, dtype=np.int32)

        if len(data) == window.res[0] * window.res[1]:
            # Window might have changed while kernel was running.
            data = data.reshape(window.res[1], window.res[0])
            return data

        else:
            return None


class Buddhacu(Buddhabrot):
    def __init__(self, iters: int = 1000, batch_size: int = int(5e3), hue: float = 0.69):
        super().__init__(iters, batch_size, hue)
        self.worker = CudaWorker("buddha.out")

    def render_samples(self, window, batch_size):
        data = self.worker.query(window, self.iters, batch_size)
        if data is not None:
            self.result += data
            self.samples += batch_size * self.worker.num_threads
