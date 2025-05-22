#include <iostream>
#include <cuda_runtime.h>

typedef double FLOAT;

constexpr int threads_per_block = 64;
constexpr int blocks_per_grid = 128;


__device__
inline FLOAT px_to_coord(const int px, const FLOAT min, const FLOAT max, const int size) {
    return (FLOAT)px * (max - min) / size + min;
}


__global__
void mandelbrot(int* img, const int width, const int height, const int iters,
                const FLOAT xmin, const FLOAT xmax, const FLOAT ymin, const FLOAT ymax) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * gridDim.x;

    for (int i = tid; i < width * height; i += num_threads) {
        const int px = i % width,
                  py = i / width;
        // Compute corresponding c value.
        const FLOAT cx = px_to_coord(px, xmin, xmax, width),
                    cy = px_to_coord(py, ymin, ymax, height);

        // Iterate
        FLOAT zx = 0.0f, zy = 0.0f;
        img[i] = 0;
        for (int j = 0; j < iters; j++) {
            const FLOAT tmp = zx * zx - zy * zy + cx;
            zy = 2 * zx * zy + cy;
            zx = tmp;
            if (zx > 2 || zx < -2 || zy > 2 || zy < -2) {
                img[i] = j;
                break;
            }
        }
    }
}


// Usage: ./a.out
// Send to stdin: width height iters samples xmin xmax ymin ymax
// samples is not used, but kept to maintain compatibility with buddha.cu
// Image data will be written to stdout.
int main() {
    while (true) {
        int width, height, iters, samples;
        FLOAT xmin, xmax, ymin, ymax;
        std::cin >> width >> height >> iters >> samples >> xmin >> xmax >> ymin >> ymax;

        // Allocate image.
        int* img = nullptr;
        cudaMallocManaged(&img, width * height * sizeof(int));
        cudaMemset(img, 0, width * height * sizeof(int));

        mandelbrot<<<blocks_per_grid, threads_per_block>>>(
            img, width, height, iters,
            xmin, xmax, ymin, ymax
        );
        cudaDeviceSynchronize();

        if (cudaGetLastError() != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
            return 1;
        }

        // Write img to stdout.
        std::cout.write((char*)img, width * height * sizeof(int));
        std::cout.flush();

        cudaFree(img);
    }
}
