import argparse
import os
import time
from pathlib import Path
from subprocess import Popen, PIPE
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import cv2
import numpy as np
import pygame
pygame.init()

ROOT = Path(__file__).absolute().parent

FONT = pygame.font.SysFont("ubuntu", 16)


def query_kernel(proc, width, height, max_iters, x_start, x_end, y_start, y_end):
    start = time.time()

    args = [width, height, max_iters, x_start, x_end, y_start, y_end]
    proc.stdin.write(" ".join(map(str, args)).encode())
    proc.stdin.write(b"\n")
    proc.stdin.flush()

    img = np.frombuffer(proc.stdout.read(width*height), dtype=np.uint32).copy()
    img = img.reshape((height, width))

    elapse = time.time() - start
    return img, elapse


def format_number(num: float) -> str:
    """
    Based on order, appends milli, micro, etc to it.
    """
    suffixes = (
        (1e3, "kilo"),
        (1, ""),
        (1e-3, "milli"),
        (1e-6, "micro"),
        (1e-9, "nano"),
        (1e-12, "pico"),
        (1e-15, "femto"),
        (1e-18, "atto"),
    )

    for thres, suf in suffixes:
        if num > thres or suf == "atto":
            return f"{num/thres:.4f} {suf}"

    raise ValueError("This should not happen")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["image", "live"])
    parser.add_argument("--kernel", default="kernel.cuda.out", help="Path to kernel executable.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-iters", type=int, default=256)
    args = parser.parse_args()
    width = args.width
    height = args.height
    max_iters = args.max_iters

    kernel_path = ROOT / args.kernel
    proc = Popen([kernel_path], stdin=PIPE, stdout=PIPE)

    if args.mode == "image":
        img, elapse = query_kernel(proc, width, height, max_iters, -2.5, 1.5, -2, 2)
        cv2.imwrite("out.png", img)
        print(f"Time: {elapse*1000:.3f} ms")

    elif args.mode == "live":
        window = pygame.display.set_mode((width, height))
        center = np.array([0, 0], dtype=float)
        x_size = 3

        drag_start_mouse = None
        drag_start_center = None
        render_time = 0

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
                    elif event.button == 5:
                        x_size /= 0.91
                    else:
                        drag_start_mouse = np.array(pygame.mouse.get_pos())
                        drag_start_center = center.copy()
                    changed = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        max_iters += 1
                    elif event.key == pygame.K_RIGHT:
                        max_iters += 20
                    elif event.key == pygame.K_DOWN:
                        max_iters -= 1
                    elif event.key == pygame.K_LEFT:
                        max_iters -= 20
                    max_iters = max(max_iters, 0)
                    changed = True

            pressed = pygame.mouse.get_pressed()
            if any(pressed):
                diff = drag_start_mouse - np.array(pygame.mouse.get_pos())
                center = drag_start_center + diff * x_size / width
                changed = True

            if changed:
                x_min = center[0] - x_size/2
                x_max = center[0] + x_size/2
                y_size = x_size * height / width
                y_min = center[1] - y_size/2
                y_max = center[1] + y_size/2

                img, render_time = query_kernel(proc, width, height, max_iters, x_min, x_max, y_min, y_max)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = img.swapaxes(0, 1)
                surface = pygame.surfarray.make_surface(img)
                window.blit(surface, (0, 0))

                stats = [
                        f"X: {x_min:.4f} to {x_max:.4f}; range {format_number(x_size)}",
                        f"Y: {y_min:.4f} to {y_max:.4f}; range {format_number(y_size)}",
                        f"Render time: {render_time*1000:.4f} ms",
                        f"Max iters: {max_iters}",
                ]
                for i, stat in enumerate(stats):
                    y = 21 * (i+1)
                    text = FONT.render(stat, True, (128, 128, 128))
                    window.blit(text, (18, y))

    proc.kill()


if __name__ == "__main__":
    main()
