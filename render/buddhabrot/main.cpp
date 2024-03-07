/*
Saves image to out.img
Use convert.py to convert it.
*/

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdio.h>

using ll = long long;

ll ITERS = 1000;
ll SAMPLES = (ll)1e8;
constexpr double SAMPLE_BOUND = 10;
constexpr double
    XMIN = -2.5,
    XMAX = 2.5,
    YMIN = -1.25,
    YMAX = 1.25;
constexpr int
    WIDTH = 2000,
    HEIGHT = 1000;


int index(int x, int y) {
    return y*WIDTH + x;
}

int pos_to_pixel(double min, double max, double x, int res) {
    return (x - min) / (max - min) * (double)res;
}

double pixel_to_pos(double min, double max, int x, int res) {
    return (double)x / (double)res * (max - min) + min;
}


void mandel_step(double x, double y, double& cx, double& cy) {
    double nx = cx*cx - cy*cy + x;
    double ny = 2*cx*cy + y;
    cx = nx;
    cy = ny;
}


bool point_in_set(double x, double y) {
    double cx = 0, cy = 0;
    for (int i = 0; i < ITERS; i++) {
        if (fabs(cx) > 2 || fabs(cy) > 2)
            return false;
        mandel_step(x, y, cx, cy);
    }
    return true;
}


void update_image(double x, double y, ll* image) {
    double cx = 0, cy = 0;
    for (int i = 0; i < ITERS; i++) {
        if (fabs(cx) > 10 || fabs(cy) > 10)
            return;
        mandel_step(x, y, cx, cy);
        int px = pos_to_pixel(XMIN, XMAX, cx, WIDTH);
        int py = pos_to_pixel(YMIN, YMAX, cy, HEIGHT);
        if (0 <= px && px < WIDTH && 0 <= py && py < HEIGHT) {
            image[index(px, py)]++;
        }
    }
}


double randf(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}


/**
 * argv: ./a.out [iters]
*/
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./a.out [iters] [samples]" << std::endl;
        return 1;
    }
    ITERS = std::stoi(argv[1]);
    SAMPLES = std::stoi(argv[2]);
    std::cerr << "Iters is " << ITERS << std::endl;
    std::cerr << "Samples is " << SAMPLES << std::endl;

    ll* image = new ll[WIDTH*HEIGHT];
    memset(image, 0, WIDTH*HEIGHT*sizeof(ll));

    for (ll i = 0; i < SAMPLES; i++) {
        if (i % (ll)1e6 == 0 || i == SAMPLES - 1) {
            // Print progress bar.
            printf("\r%15lld / %15lld -- %2lld %%", i, SAMPLES, i * 100 / SAMPLES);
            std::cout << std::flush;
        }

        double x = randf(-SAMPLE_BOUND, SAMPLE_BOUND);
        double y = randf(-SAMPLE_BOUND, SAMPLE_BOUND);
        if (!point_in_set(x, y)) {
            update_image(x, y, image);
        }
    }
    std::cout << std::endl;

    FILE* f = fopen("out.img", "wb");
    fwrite(image, sizeof(ll), WIDTH*HEIGHT, f);
    fclose(f);

    delete[] image;
}
