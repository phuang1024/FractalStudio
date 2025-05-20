#pragma once

#include <fstream>
#include <string>

typedef int* image_t;


inline image_t allocate_image(int width, int height) {
    return new int[width * height];
}

inline void save_image(image_t image, int width, int height, std::string file) {
    std::ofstream fp(file, std::ios::binary);
    fp.write((char*)image, width * height * sizeof(float));
    fp.close();
}

inline void free_image(image_t image) {
    delete[] image;
}

inline float interp(float x, float min1, float max1, float min2, float max2) {
    return min2 + (x - min1) * (max2 - min2) / (max1 - min1);
}

inline void iter_mandelbrot(float& zx, float& zy, float cx, float cy) {
    float tmp = zx * zx - zy * zy + cx;
    zy = 2 * zx * zy + cy;
    zx = tmp;
}
