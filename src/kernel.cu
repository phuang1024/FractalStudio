// for common.hpp
#define USING_CUDA

#include "common.hpp"


int main(int argc, char** argv) {
    Query q;
    q.width = width;
    q.height = height;
    q.max_iters = max_iters;

    while (true) {
        int width, height, max_iters;
        double x_start, x_end, y_start, y_end;
        std::cin >> width >> height >> max_iters >> x_start >> x_end >> y_start >> y_end;

        unsigned char* data;
        cudaMallocManaged(&data, width * height);

        Query q;
        q.width = width;
        q.height = height;
        q.max_iters = max_iters;
        q.x_start = x_start;
        q.x_end = x_end;
        q.y_start = y_start;
        q.y_end = y_end;

        compute<<<CU_THREAD_BLOCKS, CU_THREADS_PER_BLOCK>>>(q, data, 0, 0);
        cudaDeviceSynchronize();

        fwrite(data, 1, width*height, stdout);
        fflush(stdout);
    }
}
