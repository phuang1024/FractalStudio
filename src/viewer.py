import time
from threading import Thread

import pygame

from fractal import Fractal, ProgressiveType
from utils import Clock, ViewerState

pygame.init()


def render_worker(algorithm: Fractal, state: ViewerState):
    while state.run:
        state.render_result = algorithm.render(state.res)


def viewer(args, algorithm):
    clock_redraw = Clock(30)

    window = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("FractalStudio")

    state = ViewerState()
    worker_thread = Thread(target=render_worker, args=(algorithm, state))
    worker_thread.start()

    while state.run:
        time.sleep(1 / 60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state.run = False

        pygame.display.update()

        if clock_redraw.tick():
            if state.render_result is not None:
                # convert np array to pygame surface
                render = pygame.surfarray.make_surface(state.render_result.swapaxes(0, 1))
                window.blit(render, (0, 0))

    pygame.quit()
