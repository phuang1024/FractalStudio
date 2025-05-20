#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

constexpr int RAND_SEED = 0;

constexpr int threads_per_block = 32;
constexpr int blocks_per_grid = 16;


__device__
inline int coord_to_px(const float co, const float min, const float max, const int size) {
    // Map coordinate to pixel.
    const float px = (co - min) / (max - min) * size;
    return (int)px;
}


__global__
void buddhabrot(int* img, const int width, const int height, const int iters, const int samples,
                           const float xmin, const float xmax, const float ymin, const float ymax,
                           curandState* states) {
    // Initialize curand state.
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState* state = &states[tid];
    curand_init(RAND_SEED, tid, 0, state);

    // Allocate memory to store z sequence.
    float* data_re;
    float* data_im;
    cudaMalloc(&data_re, (iters + 1) * sizeof(float));
    cudaMalloc(&data_im, (iters + 1) * sizeof(float));

    for (int i = 0; i < samples; i++) {
        // Sample c value
        const float cx = (curand_uniform(state) - 0.5f) * 10.0f,
                    cy = (curand_uniform(state) - 0.5f) * 10.0f;
        // Initialize z0 = 0
        data_re[0] = 0.0f;
        data_im[0] = 0.0f;

        // Iterate
        bool in_set = true;
        for (int j = 0; j < iters; j++) {
            // Compute next z value
            data_re[j + 1] = data_re[j] * data_re[j] - data_im[j] * data_im[j] + cx;
            data_im[j + 1] = 2.0f * data_re[j] * data_im[j] + cy;
            // Check divergence
            if (data_re[j + 1] > 2.0f || data_re[j + 1] < -2.0f ||
                data_im[j + 1] > 2.0f || data_im[j + 1] < -2.0f) {
                in_set = false;
                break;
            }
        }
        if (!in_set) {
            // Update values.
            for (int j = 1; j <= iters; j++) {
                const int px = coord_to_px(data_re[j], xmin, xmax, width),
                          py = coord_to_px(data_im[j], ymin, ymax, height);
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    atomicAdd(&img[py * width + px], 1);
                }
            }
        }
    }

    // Free memory.
    cudaFree(data_re);
    cudaFree(data_im);
}


// Usage: ./a.out
// Send to stdin: width height iters samples xmin xmax ymin ymax
// Image data will be written to stdout.
int main() {
    // Initialize curand.
    curandState* states;
    cudaMalloc(&states, blocks_per_grid * threads_per_block * sizeof(curandState));

    while (true) {
        int width, height, iters, samples;
        float xmin, xmax, ymin, ymax;
        std::cin >> width >> height >> iters >> samples >> xmin >> xmax >> ymin >> ymax;

        // Allocate image.
        int* img = nullptr;
        cudaMallocManaged(&img, width * height * sizeof(int));
        cudaMemset(img, 0, width * height * sizeof(int));

        buddhabrot<<<blocks_per_grid, threads_per_block>>>(img, width, height, iters, samples, xmin, xmax, ymin, ymax, states);
        cudaDeviceSynchronize();

        if (cudaGetLastError() != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
            return 1;
        }

        // Write img to stdout.
        std::cout.write((char*)img, width * height * sizeof(int));
        std::cout.flush();

        cudaFree(img);

        //std::cerr << "Done!" << std::endl;
    }

    cudaFree(states);
}
