import argparse
import os
import sys
import time
from subprocess import Popen, PIPE

import cv2
import numpy as np
import pygame
pygame.init()

PARENT = os.path.dirname(os.path.realpath(__file__))
KERNEL = os.path.join(PARENT, "a.out")


def query_kernel(proc, width, height, x_start, x_end, y_start, y_end):
    start = time.time()

    args = [x_start, x_end, y_start, y_end]
    proc.stdin.write(" ".join(map(str, args)).encode())
    proc.stdin.write(b"\n")
    proc.stdin.flush()

    img = np.empty((width*height), dtype=np.uint8)
    data = b""
    i = 0
    while i < width*height:
        remaining = width*height - i
        data = proc.stdout.read(remaining)

        img[i:i+len(data)] = np.frombuffer(data, dtype=np.uint8)
        i += len(data)

    img = img * 255
    img = img.reshape((height, width))

    elapse = time.time() - start
    return img, elapse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["image", "live"])
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    width = args.width
    height = args.height

    proc = Popen([KERNEL, str(width), str(height)], stdin=PIPE, stdout=PIPE)

    if args.mode == "image":
        img, elapse = query_kernel(proc, width, height, -2.5, 1.5, -2, 2)
        cv2.imwrite("out.png", img)
        print(f"Time: {elapse:.3f} seconds")

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
                sys.stdout.write(f"\rTime: {elapse:.3f} seconds")

                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = img.swapaxes(0, 1)
                surface = pygame.surfarray.make_surface(img)
                window.blit(surface, (0, 0))

        print()

    proc.kill()


if __name__ == "__main__":
    main()
