constexpr int CU_THREAD_BLOCKS = 64;
constexpr int CU_THREADS_PER_BLOCK = 64;

constexpr int CPU_THREADS = 16;

#include <iostream>
#include <stdio.h>
#include <string>


/**
 * Query from python to compute a region.
 */
struct Query {
    int width, height;
    int max_iters;
    double x_start, x_end, y_start, y_end;
};


/**
 * Returns number of iterations before norm >= 4.
 * IMPORTANT:
 * - Return 127 means it STAYS BOUNDED; it is in the set
 * - Return 0-126 means it is not in the set; return is how many iterations it took.
 */
#ifdef USING_CUDA
__device__
#endif
char point_in_set(const double re, const double im, int max_iters) {
    double pt_re = 0, pt_im = 0;  // simulated point

    for (int i = 0; i < max_iters; i++) {
        const double a = pt_re, b = pt_im;
        pt_re = a*a - b*b;
        pt_im = 2 * a * b;
        pt_re += re;
        pt_im += im;

        if (pt_re*pt_re + pt_im*pt_im > 5) {
            return 126 * i / max_iters;
        }
    }
    return 127;
}

/**
 * Kernel to call from host.
 * Prints results to stdout according to docs.
 */
#ifdef USING_CUDA
__global__
#endif
void compute(Query q, char* data, int start, int stride) {
    const double x_scl = (q.x_end-q.x_start) / (double)q.width;
    const double y_scl = (q.y_end-q.y_start) / (double)q.height;

    #ifdef USING_CUDA
    start = blockIdx.x * blockDim.x + threadIdx.x;
    stride = blockDim.x * gridDim.x;
    #endif

    for (int i = start; i < q.width*q.height; i += stride) {
        const int px_x = i % q.width, px_y = i / q.width;
        const double x = q.x_start + (double)px_x*x_scl;
        const double y = q.y_start + (double)px_y*y_scl;

        char result = point_in_set(x, y, q.max_iters);
        data[i] = result;
    }
}
