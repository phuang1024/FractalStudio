// Max iterations
#define  ITERS  100

#include <iostream>
#include <stdio.h>
#include <string>


/**
 * Returns number of iterations before norm >= 4.
 * Returns -1 if remains bounded.
 */
__device__ char point_in_set(const double re, const double im) {
    double pt_re = 0, pt_im = 0;  // simulated point

    for (int i = 0; i < ITERS; i++) {
        const double a = pt_re, b = pt_im;
        pt_re = a*a - b*b;
        pt_im = 2 * a * b;
        pt_re += re;
        pt_im += im;

        if (pt_re*pt_re + pt_im*pt_im > 5) {
            return 0;
            if (i > 127)
                i = 127;
            return i;
        }
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

        char result = point_in_set(x, y);
        data[i] = result;
    }
}


/**
 * Usage:
 * ./a.out width height
 * Then send "x_start x_end y_start y_end\n" to stdin
 * Read result from stdout.
 */
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Invalid CLI args. Read documentation for help.\n");
        return 1;
    }

    const int width = std::stoi(argv[1]);
    const int height = std::stoi(argv[2]);

    while (true) {
        double x_start, x_end, y_start, y_end;
        std::cin >> x_start >> x_end >> y_start >> y_end;

        char* data;
        cudaMallocManaged(&data, width * height);

        compute<<<64, 64>>>(width, height, x_start, x_end, y_start, y_end, data);
        cudaDeviceSynchronize();

        fwrite(data, 1, width*height, stdout);
        fflush(stdout);
    }
}
