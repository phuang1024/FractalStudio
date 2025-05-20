#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

constexpr int threads_per_block = 64;
constexpr int blocks_per_grid = 32;


__device__
inline int coord_to_px(const float co, const float min, const float max, const int size) {
    // Map coordinate to pixel.
    const float px = (co - min) / (max - min) * size;
    return (int)px;
}


__global__
void buddhabrot(int* img, const int width, const int height, const int iters, const int samples,
                           const float xmin, const float xmax, const float ymin, const float ymax,
                           curandState* states, int rand_seed) {
    // Initialize curand state.
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState* state = &states[tid];
    curand_init(rand_seed, tid, 0, state);

    for (int i = 0; i < samples; i++) {
        // Sample c value
        const float cx = (curand_uniform(state) - 0.5f) * 10.0f,
                    cy = (curand_uniform(state) - 0.5f) * 10.0f;

        // Iterate
        float zx = 0.0f, zy = 0.0f;
        bool in_set = true;
        for (int j = 0; j < iters; j++) {
            // Compute next z value
            float tmp = zx * zx - zy * zy + cx;
            zy = 2.0f * zx * zy + cy;
            zx = tmp;
            // Check divergence
            if (zx > 2.0f || zx < -2.0f ||
                zy > 2.0f || zy < -2.0f) {
                in_set = false;
                break;
            }
        }
        if (!in_set) {
            // Update values.
            float zx = 0.0f, zy = 0.0f;
            for (int j = 0; j <= iters; j++) {
                float tmp = zx * zx - zy * zy + cx;
                zy = 2.0f * zx * zy + cy;
                zx = tmp;
                const int px = coord_to_px(zx, xmin, xmax, width),
                          py = coord_to_px(zy, ymin, ymax, height);
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    atomicAdd(&img[py * width + px], 1);
                }
                // Check divergence
                if (zx > 2.0f || zx < -2.0f ||
                    zy > 2.0f || zy < -2.0f) {
                    break;
                }
            }
        }
    }
}


// Usage: ./a.out
// Send to stdin: width height iters samples xmin xmax ymin ymax
// Image data will be written to stdout.
int main() {
    // Initialize curand.
    curandState* states;
    cudaMalloc(&states, blocks_per_grid * threads_per_block * sizeof(curandState));
    int rand_seed = 0;

    while (true) {
        int width, height, iters, samples;
        float xmin, xmax, ymin, ymax;
        std::cin >> width >> height >> iters >> samples >> xmin >> xmax >> ymin >> ymax;

        // Allocate image.
        int* img = nullptr;
        cudaMallocManaged(&img, width * height * sizeof(int));
        cudaMemset(img, 0, width * height * sizeof(int));

        buddhabrot<<<blocks_per_grid, threads_per_block>>>(
            img, width, height, iters, samples, xmin, xmax, ymin, ymax, states, rand_seed
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
        rand_seed += 1;
    }

    cudaFree(states);
}
