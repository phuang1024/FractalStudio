import argparse
import os
import sys
import time
from pathlib import Path
from subprocess import Popen, PIPE

import cv2
import numpy as np
import pygame
pygame.init()

ROOT = Path(__file__).absolute().parent


def query_kernel(proc, width, height, x_start, x_end, y_start, y_end):
    start = time.time()

    args = [x_start, x_end, y_start, y_end]
    proc.stdin.write(" ".join(map(str, args)).encode())
    proc.stdin.write(b"\n")
    proc.stdin.flush()

    img = np.frombuffer(proc.stdout.read(width*height), dtype=np.uint8).copy()
    img = img.reshape((height, width))
    img[img == 127] = 255

    elapse = time.time() - start
    return img, elapse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["image", "live"])
    parser.add_argument("--kernel", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-iters", type=int, default=256)
    args = parser.parse_args()
    width = args.width
    height = args.height

    kernel_path = ROOT / f"kernel.{args.kernel}.out"
    proc = Popen([kernel_path, str(width), str(height), str(args.max_iters)], stdin=PIPE, stdout=PIPE)

    if args.mode == "image":
        img, elapse = query_kernel(proc, width, height, -2.5, 1.5, -2, 2)
        cv2.imwrite("out.png", img)
        print(f"Time: {elapse*1000:.3f} ms")

    elif args.mode == "live":
        window = pygame.display.set_mode((width, height))
        center = np.array([0, 0], dtype=float)
        x_size = 3

        drag_start_mouse = None
        drag_start_center = None

        run = True
        first_loop = True
        while run:
            changed = first_loop
            first_loop = False

            time.sleep(0.01)
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        x_size *= 0.91
                        changed = True
                    elif event.button == 5:
                        x_size /= 0.91
                        changed = True
                    else:
                        drag_start_mouse = np.array(pygame.mouse.get_pos())
                        drag_start_center = center.copy()

            pressed = pygame.mouse.get_pressed()
            if any(pressed):
                diff = drag_start_mouse - np.array(pygame.mouse.get_pos())
                center = drag_start_center + diff * x_size / width
                changed = True

            if changed:
                y_size = x_size * height / width
                bounds = (center[0] - x_size/2, center[0] + x_size/2, center[1] - y_size/2, center[1] + y_size/2)
                img, elapse = query_kernel(proc, width, height, *bounds)
                sys.stdout.write(f"\rTime: {elapse*1000:.3f} ms   ")

                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = img.swapaxes(0, 1)
                surface = pygame.surfarray.make_surface(img)
                window.blit(surface, (0, 0))

        print()

    proc.kill()


if __name__ == "__main__":
    main()
