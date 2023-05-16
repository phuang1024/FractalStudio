import argparse
from colorsys import hsv_to_rgb
from math import tanh

import cv2
import numpy as np

WIDTH = 1000
HEIGHT = 2000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input image")
    parser.add_argument("output", help="output image")
    parser.add_argument("-e", "--exposure", type=float, default=1)
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.int64)
    data = data.reshape((HEIGHT, WIDTH))
    print(np.max(data))
    data = np.sqrt(data)
    data = np.clip(data, np.min(data), np.max(data) * 0.5)
    if np.max(data) > 0:
        data = data / np.max(data)

    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            n = data[y, x] * args.exposure
            s = 1 - tanh(n)
            v = tanh(n)
            r, g, b = hsv_to_rgb(0.66, s, v)
            image[y, x] = (b * 255, g * 255, r * 255)

    cv2.imwrite(args.output, image)


if __name__ == "__main__":
    main()
