#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include "utils.hpp"

const int WIDTH = 2000;
const int HEIGHT = 1000;
const int ITERS = 1000;
const int SAMPLES = (int)1e6;


int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Not enough arguments." << std::endl;
        return 1;
    }

    float xmin = std::stof(argv[1]),
          xmax = std::stof(argv[2]),
          ymin = std::stof(argv[3]),
          ymax = std::stof(argv[4]);

    image_t img = allocate_image(WIDTH, HEIGHT);
    memset(img, 0, WIDTH * HEIGHT * sizeof(int));

    for (int i = 0; i < SAMPLES; i++) {
        // Sample random C value
        float cx = float(rand()) / RAND_MAX * 4 - 2,
              cy = float(rand()) / RAND_MAX * 4 - 2;


        // Check if in set.
        float zx = 0, zy = 0;
        bool in_set = true;
        for (int i = 0; i < ITERS; ++i) {
            iter_mandelbrot(zx, zy, cx, cy);
            if (fabs(zx) > 2 || fabs(zy) > 2) {
                in_set = false;
                break;
            }
        }

        // Update image.
        if (!in_set) {
            float zx = 0, zy = 0;
            for (int i = 0; i < ITERS; ++i) {
                iter_mandelbrot(zx, zy, cx, cy);
                int px = interp(zx, xmin, xmax, 0, WIDTH),
                    py = interp(zy, ymin, ymax, 0, HEIGHT);
                if (px >= 0 && px < WIDTH && py >= 0 && py < HEIGHT) {
                    img[py * WIDTH + px]++;
                }
                if (fabs(zx) > 10 || fabs(zy) > 10) {
                    break;
                }
            }
        }
    }

    save_image(img, WIDTH, HEIGHT, "buddhabrot.img");
    free_image(img);
}
