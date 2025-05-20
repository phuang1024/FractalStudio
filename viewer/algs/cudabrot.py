"""
CUDA accelerated buddhabrot rendering, optimized for performance.
"""

import os
from subprocess import Popen, PIPE

import numpy as np

from algs.buddhabrot import Buddhabrot

PARENT = os.path.dirname(os.path.abspath(__file__))


class Cudabrot(Buddhabrot):
    num_threads = 16 * 32

    def __init__(self, iters: int = 1000, batch_size: int = int(1e3), hue: float = 0.69):
        self.iters = iters
        self.batch_size = batch_size
        self.hue = hue

        self.exposure = 1.5
        self.result = None

        self.process = Popen([os.path.join(PARENT, "cudabrot.out")], stdin=PIPE, stdout=PIPE)

    def render_samples(self, window):
        assert self.process.returncode is None

        self.process.stdin.write(f"{window.res[0]} {window.res[1]} {self.iters} {self.batch_size} {window.xmin} {window.xmax} {window.ymin} {window.ymax}\n".encode())
        self.process.stdin.flush()
        # Read result from process (int32).
        data = self.process.stdout.read(window.res[0] * window.res[1] * 4)
        data = np.frombuffer(data, dtype=np.int32).reshape(window.res[1], window.res[0])
        self.result += data.astype(np.uint32)
        self.samples += self.batch_size * self.num_threads
