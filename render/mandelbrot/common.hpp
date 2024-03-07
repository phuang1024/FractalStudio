constexpr int CU_THREAD_BLOCKS = 64;
constexpr int CU_THREADS_PER_BLOCK = 64;

constexpr int CPU_THREADS = 16;

#include <iostream>
#include <stdio.h>
#include <string>

// Change precision and speed.
using FLOAT = double;


/**
 * Query from python to compute a region.
 */
struct Query {
    int width, height;
    int max_iters;
    FLOAT x_start, x_end, y_start, y_end;
};


/**
 * Returns color of point.
 * 255 is in the set.
 * 0-200 fades away.
 */
#ifdef USING_CUDA
__device__
#endif
unsigned int point_color(const FLOAT re, const FLOAT im, int max_iters) {
    FLOAT pt_re = 0, pt_im = 0;  // simulated point

    for (int i = 0; i < max_iters; i++) {
        const FLOAT a = pt_re, b = pt_im;
        pt_re = a*a - b*b;
        pt_im = 2 * a * b;
        pt_re += re;
        pt_im += im;

        if (pt_re*pt_re + pt_im*pt_im > 4) {
            return 200 * i / max_iters;
        }
    }
    return 255;
}

/**
 * Kernel to call from host.
 * Prints results to stdout according to docs.
 */
#ifdef USING_CUDA
__global__
#endif
void compute(Query q, unsigned int* data, int start, int stride) {
    const FLOAT x_scl = (q.x_end-q.x_start) / (FLOAT)q.width;
    const FLOAT y_scl = (q.y_end-q.y_start) / (FLOAT)q.height;

    #ifdef USING_CUDA
    start = blockIdx.x * blockDim.x + threadIdx.x;
    stride = blockDim.x * gridDim.x;
    #endif

    for (int i = start; i < q.width*q.height; i += stride) {
        const int px_x = i % q.width, px_y = i / q.width;
        const FLOAT x = q.x_start + (FLOAT)px_x*x_scl;
        const FLOAT y = q.y_start + (FLOAT)px_y*y_scl;

        unsigned int result = point_color(x, y, q.max_iters);
        data[i] = result;
    }
}
