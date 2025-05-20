"""
Mandelbrot render with Hyprland like color scheme.
"""

import argparse

import cv2
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bounds", type=str, help="xmin:xmax:ymin:ymax")
    parser.add_argument("--width", type=int, default=2000)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()

    xmin, xmax, ymin, ymax = map(float, args.bounds.split(":"))

    with open("mandelbrot.bin", "rb") as fp:
        data = np.fromfile(fp, dtype=np.uint8).reshape(args.height, args.width)


if __name__ == "__main__":
    main()
