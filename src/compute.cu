//
//  Mandelbrot
//  Compute the mandelbrot set with a GPU.
//  Copyright Patrick Huang 2021
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <https://www.gnu.org/licenses/>.
//

/**
 * Number of iterations to do.
 */
#define  ITERS  50

#include <stdio.h>
#include <string>


/**
 * Whether point is in mandelbrot set.
 */
__device__ char point(const double re, const double im) {
    double pt_re = 0, pt_im = 0;  // simulated point

    for (int i = 0; i < ITERS; i++) {
        const double a = pt_re, b = pt_im;
        pt_re = a*a - b*b;
        pt_im = 2 * a * b;
        pt_re += re;
        pt_im += im;

        if (pt_re*pt_re + pt_im*pt_im > 5)
            return 0;
    }
    return 1;
}

/**
 * Kernel to call from host.
 * Prints results to stdout according to docs.
 */
__global__ void compute(const int width, const int height, const double x_start, const double x_end,
const double y_start, const double y_end, char* data) {
    const double x_scl = (x_end-x_start) / (double)width;
    const double y_scl = (y_end-y_start) / (double)height;

    const int start = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = start; i < width*height; i += stride) {
        const int px_x = i % width, px_y = i / width;
        const double x = x_start + (double)px_x*x_scl;
        const double y = y_start + (double)px_y*y_scl;

        const char in_set = point(x, y);
        data[i] = in_set;
    }
}


/**
 * argv: ./a.out width height x_start x_end y_start y_end
 */
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Invalid CLI args. Read documentation for help.\n");
        return 1;
    }

    const int width = std::stoi(argv[1]);
    const int height = std::stoi(argv[2]);
    const double x_start = std::stod(argv[3]);
    const double x_end = std::stod(argv[4]);
    const double y_start = std::stod(argv[5]);
    const double y_end = std::stod(argv[6]);

    char* data;
    cudaMallocManaged(&data, width * height);

    compute<<<64, 64>>>(width, height, x_start, x_end, y_start, y_end, data);
    cudaDeviceSynchronize();

    fwrite(data, 1, width*height, stdout);
    fflush(stdout);
}
