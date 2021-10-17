#
#  Mandelbrot
#  Compute the mandelbrot set with a GPU.
#  Copyright Patrick Huang 2021
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import os
import time
import numpy as np
import cv2
from subprocess import Popen, PIPE

WIDTH = 1920
HEIGHT = 1920

PARENT = os.path.dirname(os.path.realpath(__file__))
EXE = os.path.join(PARENT, "a.out")

args = [EXE, str(WIDTH), str(HEIGHT), "-3", "1", "-2", "2"]
proc = Popen(args, stdout=PIPE)
time.sleep(3)

img = np.empty((WIDTH*HEIGHT), dtype=np.uint8)
for i in range(WIDTH*HEIGHT):
    in_set = proc.stdout.read(1)[0]
    img[i] = in_set * 255

img = img.reshape((HEIGHT, WIDTH))
cv2.imwrite("out.png", img)
