import os
import time
from subprocess import Popen, PIPE

import cv2
import numpy as np

PARENT = os.path.dirname(os.path.realpath(__file__))
KERNEL = os.path.join(PARENT, "a.out")
WIDTH = 4096
HEIGHT = 4096


def query_kernel(proc, x_start, x_end, y_start, y_end):
    args = [x_start, x_end, y_start, y_end]
    proc.stdin.write(" ".join(map(str, args)).encode())
    proc.stdin.write(b"\n")
    proc.stdin.flush()

    img = np.empty((WIDTH*HEIGHT), dtype=np.uint8)
    data = b""
    i = 0
    while i < WIDTH*HEIGHT:
        remaining = WIDTH*HEIGHT - i
        data = proc.stdout.read(remaining)

        img[i:i+len(data)] = np.frombuffer(data, dtype=np.uint8)
        i += len(data)

    img = img * 255
    img = img.reshape((HEIGHT, WIDTH))
    return img


proc = Popen([KERNEL, str(WIDTH), str(HEIGHT)], stdin=PIPE, stdout=PIPE)
img = query_kernel(proc, -2.5, 1.5, -2, 2)

start = time.time()
img = query_kernel(proc, -2.5, 1.5, -2, 2)
elapse = time.time() - start

cv2.imwrite("out.png", img)
print(f"Time: {elapse:.2f} seconds")

proc.kill()
