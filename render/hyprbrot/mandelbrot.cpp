#include <algorithm>
#include <iostream>
#include "utils.hpp"

const int WIDTH = 2000;
const int HEIGHT = 1000;
const int ITERS = 1000;


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

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            float cx = interp(x, 0, WIDTH, xmin, xmax),
                  cy = interp(y, 0, HEIGHT, ymin, ymax);
            float zx = 0, zy = 0;

            int i;
            bool in_set = true;
            for (i = 0; i < ITERS && zx * zx + zy * zy < 4.0f; ++i) {
                iter_mandelbrot(zx, zy, cx, cy);
                if (zx * zx + zy * zy > 4) {
                    in_set = false;
                    break;
                }
            }
            if (in_set) {
                i = -1;
            }
            img[y * WIDTH + x] = i;
        }
    }

    save_image(img, WIDTH, HEIGHT, "mandelbrot.img");
    free_image(img);
}
