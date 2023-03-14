// for common.hpp
#define USING_CUDA

#include "common.hpp"


int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Invalid CLI args. Read documentation for help.\n");
        return 1;
    }

    const int width = std::stoi(argv[1]);
    const int height = std::stoi(argv[2]);
    const int max_iters = std::stoi(argv[3]);

    Query q;
    q.width = width;
    q.height = height;
    q.max_iters = max_iters;

    while (true) {
        double x_start, x_end, y_start, y_end;
        std::cin >> x_start >> x_end >> y_start >> y_end;

        char* data;
        cudaMallocManaged(&data, width * height);

        q.x_start = x_start;
        q.x_end = x_end;
        q.y_start = y_start;
        q.y_end = y_end;
        q.data = data;

        compute<<<THREAD_BLOCKS, THREADS_PER_BLOCK>>>(width, height, x_start, x_end, y_start, y_end, data);
        cudaDeviceSynchronize();

        fwrite(data, 1, width*height, stdout);
        fflush(stdout);
    }
}
