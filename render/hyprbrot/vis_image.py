import argparse

import cv2
import numpy as np


def interp(data, min1, max1, min2, max2):
    return (data - min1) * (max2 - min2) / (max1 - min1) + min2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("--width", type=int, default=2000)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()

    with open(args.image, "rb") as f:
        data = f.read()
        data = np.frombuffer(data, dtype=np.int32).reshape((args.height, args.width))

    img = interp(data, np.min(data), np.max(data), 0, 255).astype(np.uint8)
    cv2.imshow("a", img)
    while True:
        if cv2.waitKey(0) == ord("q"):
            break


if __name__ == "__main__":
    main()
