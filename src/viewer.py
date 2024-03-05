import time
from threading import Thread

import cv2
import pygame

from fractal import Fractal, ProgressiveType
from utils import Clock, ViewerState

pygame.init()


def render_worker(alg: Fractal, state: ViewerState):
    last_window_changed = -1

    upres_iter = 0

    while state.run:
        if alg.progressive == ProgressiveType.NONE:
            if state.window_changed != last_window_changed:
                state.render_result = alg.render(state.res)
                last_window_changed = state.window_changed

        elif alg.progressive == ProgressiveType.UPRES:
            if state.window_changed != last_window_changed:
                upres_iter = 5
                last_window_changed = state.window_changed
            if upres_iter >= 0:
                scale = 2 ** upres_iter
                res = (state.res[0] // scale, state.res[1] // scale)
                state.render_result = cv2.resize(alg.render(res), state.res)
                upres_iter -= 1


def viewer(args, algorithm):
    clock_redraw = Clock(30)

    window = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("FractalStudio")

    state = ViewerState()
    worker_thread = Thread(target=render_worker, args=(algorithm, state))
    worker_thread.start()

    while state.run:
        time.sleep(1 / 60)

        window_changed = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state.run = False
            elif event.type == pygame.VIDEORESIZE:
                window_changed = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    state.run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                window_changed = True
                if event.button == 4:
                    state.scale *= 1.1
                elif event.button == 5:
                    state.scale /= 1.1

        pygame.display.update()

        if window_changed:
            state.window_changed += 1
            window.fill((0, 0, 0))

        state.res = window.get_size()

        if clock_redraw.tick():
            if state.render_result is not None:
                # convert np array to pygame surface
                render = pygame.surfarray.make_surface(state.render_result.swapaxes(0, 1))
                window.blit(render, (0, 0))

    pygame.quit()
