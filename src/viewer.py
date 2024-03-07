import time
from copy import copy
from threading import Thread

import cv2
import numpy as np
import pygame

from fractal import Fractal, ProgressiveType
from utils import Clock, ViewerState

pygame.init()

UPRES_STEPS = 3


def render_worker(alg: Fractal, state: ViewerState):
    last_window_changed = -1

    upres_iter = 0

    while state.run:
        if alg.progressive == ProgressiveType.NONE:
            if state.window_changed != last_window_changed:
                state.render_result = alg.render(state.window)
                last_window_changed = state.window_changed

        elif alg.progressive == ProgressiveType.UPRES:
            if state.window_changed != last_window_changed:
                upres_iter = UPRES_STEPS
                last_window_changed = state.window_changed
            if upres_iter >= 0:
                scale = 2 ** upres_iter
                res = (state.window.res[0] // scale, state.window.res[1] // scale)
                new_window = copy(state.window)
                new_window.res = res
                state.render_result = cv2.resize(alg.render(new_window), state.window.res)
                upres_iter -= 1


def viewer(args, algorithm):
    clock_redraw = Clock(30)

    window = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("FractalStudio")

    state = ViewerState()
    worker_thread = Thread(target=render_worker, args=(algorithm, state))
    worker_thread.start()

    # Store state at mousedown.
    click_mouse_pos = None
    click_window_pos = None
    # Updated every iter. For performance.
    last_mouse_pos = None

    while state.run:
        time.sleep(1 / 60)

        window_changed = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state.run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    state.run = False

            elif event.type == pygame.VIDEORESIZE:
                window_changed = True

            elif event.type == pygame.MOUSEBUTTONDOWN:
                window_changed = True
                if event.button == 4:
                    state.window.scale /= 1.1
                elif event.button == 5:
                    state.window.scale *= 1.1
                elif event.button == 1:
                    click_mouse_pos = np.array(event.pos)
                    click_window_pos = state.window.pos

        # Handle mouse drag
        mouse_pressed = pygame.mouse.get_pressed()
        mouse_pos = np.array(pygame.mouse.get_pos())
        if mouse_pressed[0] and (mouse_pos != last_mouse_pos).any():
            window_changed = True
            mouse_delta = mouse_pos - click_mouse_pos
            scaling = state.window.scale / state.window.res[0]
            state.window.pos = click_window_pos - mouse_delta * scaling

        # Trigger redraw.
        if window_changed:
            state.window_changed += 1
            #window.fill((0, 0, 0))

        state.window.res = window.get_size()

        if clock_redraw.tick():
            if state.render_result is not None:
                # convert np array to pygame surface
                render = pygame.surfarray.make_surface(state.render_result.swapaxes(0, 1))
                window.blit(render, (0, 0))

        last_mouse_pos = mouse_pos

        pygame.display.update()

    pygame.quit()
