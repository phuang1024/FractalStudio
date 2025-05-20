"""
Mandelbrot render with Hyprland like color scheme.
"""

import argparse

import cv2
import numpy as np

GREEN = (0, 255, 153)
BLUE = (51, 204, 255)


def interp(data, min1, max1, min2, max2, clamp=False):
    fac = (data - min1) / (max1 - min1)
    if clamp:
        fac = np.clip(fac, 0, 1)
    return fac * (max2 - min2) + min2


def color_mandel(mandel):
    """
    Mandelbrot based coloring.
    Interp from green to blue to black based on iters.
    """
    img = np.zeros((mandel.shape[0], mandel.shape[1], 3), dtype=np.float32)

    # Color points that escape
    iters = mandel.copy()
    iters[mandel <= 0] = 1  # Placeholder > 0
    iters = np.log(iters.astype(np.float32))
    iters = interp(iters, np.min(iters), np.max(iters), 0, 1, clamp=True)

    for ch in range(3):
        img[..., ch] = interp(iters, 0.5, 0.75, BLUE[ch], GREEN[ch], clamp=True)
        img[..., ch] *= interp(iters, 0.2, 0.6, 0, 1, clamp=True)
    img[mandel < 0] = (0, 0, 0)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=2000)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()

    with open("mandelbrot.img", "rb") as fp:
        mandel = np.fromfile(fp, dtype=np.int32).reshape(args.height, args.width)

    img = color_mandel(mandel)

    img = np.clip(img.astype(np.uint8), 0, 255)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("hyprbrot.png", img)


if __name__ == "__main__":
    main()
