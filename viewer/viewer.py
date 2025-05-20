import os
import time
from copy import copy
from threading import Thread

import cv2
import numpy as np
import pygame

from fractal import Fractal, ProgressiveType
from utils import Clock, ViewerState

pygame.init()

FONT = pygame.font.SysFont("courier", 16)

UPRES_STEPS = 3


def render_worker(alg: Fractal, state: ViewerState):
    last_window_changed = -1
    iter_num = 0

    upres_iter = 0

    while state.run:
        window_changed = state.window_changed != last_window_changed
        if window_changed:
            alg.new_window(state.window)
            iter_num = 0

        if alg.progressive == ProgressiveType.NONE:
            if window_changed:
                state.render_result = alg.render(state.window)

        elif alg.progressive == ProgressiveType.UPRES:
            if window_changed:
                upres_iter = UPRES_STEPS
            if upres_iter >= 0:
                scale = 2 ** upres_iter
                res = (state.window.res[0] // scale, state.window.res[1] // scale)
                new_window = copy(state.window)
                new_window.res = res
                state.render_result = cv2.resize(alg.render(new_window), state.window.res)
                upres_iter -= 1

        elif alg.progressive == ProgressiveType.SAMPLES:
            state.render_result = alg.render(state.window)

        if window_changed:
            last_window_changed = state.window_changed

        alg.iter_num = iter_num
        iter_num += 1


def viewer(args, algorithm):
    clock_redraw = Clock(15)

    window = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("FractalStudio")
    if os.path.isfile("icon.jpg"):
        icon = pygame.image.load("icon.jpg")
        pygame.display.set_icon(icon)

    state = ViewerState()
    worker_thread = Thread(target=render_worker, args=(algorithm, state))
    worker_thread.start()

    stats_enabled = True

    # Store state at mousedown.
    click_mouse_pos = None
    click_window_pos = None
    # Updated every iter.
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
                elif event.key == pygame.K_s:
                    stats_enabled = not stats_enabled

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

            elif event.type == pygame.MOUSEBUTTONUP:
                window_changed = True

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
                # Convert np array to pygame surface
                render = pygame.surfarray.make_surface(state.render_result.swapaxes(0, 1))
                window.blit(render, (0, 0))

                # Draw stats
                if stats_enabled:
                    stats = algorithm.get_stats()
                    draw_stats(window, stats)

        last_mouse_pos = mouse_pos

        pygame.display.update()

        # TODO implement draw stats.

    pygame.quit()


def draw_stats(window, stats):
    if len(stats) == 0:
        return

    max_width = max(FONT.render(stat, True, (255, 255, 255)).get_width() for stat in stats)
    rect = pygame.Surface((max_width + 20, 20 * len(stats) + 15))
    rect.fill((0, 0, 0))
    rect.set_alpha(170)
    window.blit(rect, (10, 10))

    x = 20
    y = 20
    for stat in stats:
        text = FONT.render(stat, True, (255, 255, 255))
        window.blit(text, (x, y))
        y += 20
