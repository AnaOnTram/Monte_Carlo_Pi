#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

const int BLOCK_SIZE = 256;

__global__ void estimate_pi(int *block_counts, int n, unsigned long long seed) {
    __shared__ int shared_count;
    if (threadIdx.x == 0) {
        shared_count = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        curandState_t state;
        curand_init(seed, idx, 0, &state);

        float x = curand_uniform(&state);
        float y = curand_uniform(&state);

        if (x*x + y*y <= 1.0f) {
            atomicAdd(&shared_count, 1);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        block_counts[blockIdx.x] = shared_count;
    }
}

void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int n;
    std::cout << "Enter number of samples: ";
    std::cin >> n;

    if (n <= 0) {
        std::cerr << "Invalid sample size" << std::endl;
        return EXIT_FAILURE;
    }

    unsigned long long seed = 12345; 

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *d_block_counts;
    checkCudaError(cudaMalloc(&d_block_counts, gridSize * sizeof(int)));
    
    int *h_block_counts = new int[gridSize];

    estimate_pi<<<gridSize, BLOCK_SIZE>>>(d_block_counts, n, seed);
    checkCudaError(cudaGetLastError());

    checkCudaError(cudaMemcpy(h_block_counts, d_block_counts, 
                             gridSize * sizeof(int), cudaMemcpyDeviceToHost));

    int total_count = 0;
    for (int i = 0; i < gridSize; ++i) {
        total_count += h_block_counts[i];
    }

    double pi_estimate = 4.0 * total_count / static_cast<double>(n);
    double error = std::fabs(pi_estimate - M_PI);

    std::cout << "Estimated pi: " << pi_estimate 
              << "\nError vs M_PI: " << error 
              << std::endl;

    cudaFree(d_block_counts);
    delete[] h_block_counts;
    
    return EXIT_SUCCESS;
}
