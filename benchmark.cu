#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>

const int SAMPLES_PER_THREAD = 1000;
const int BLOCK_SIZE = 256;

__global__ void estimate_pi_optimized(unsigned long long *global_hits, int total_threads, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_threads) return;

    curandState_t state;
    curand_init(seed, tid, 0, &state);

    int local_hits = 0;

    for (int i = 0; i < SAMPLES_PER_THREAD; i++) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        
        if (x * x + y * y <= 1.0f) {
            local_hits++;
        }
    }

    unsigned active = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_hits += __shfl_down_sync(active, local_hits, offset);
    }

    if ((threadIdx.x & 0x1f) == 0) {
        atomicAdd(global_hits, (unsigned long long)local_hits);
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    unsigned long long requested_n;
    std::cout << "Enter approximate total number of samples (e.g. 100000000): ";
    std::cin >> requested_n;

    if (requested_n <= 0) return EXIT_FAILURE;

    int total_threads = (requested_n + SAMPLES_PER_THREAD - 1) / SAMPLES_PER_THREAD;
    int gridSize = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    unsigned long long total_samples = (unsigned long long)total_threads * SAMPLES_PER_THREAD;

    std::cout << "Benchmarking with:" << std::endl;
    std::cout << "  - Grid Size: " << gridSize << std::endl;
    std::cout << "  - Block Size: " << BLOCK_SIZE << std::endl;
    std::cout << "  - Samples Per Thread: " << SAMPLES_PER_THREAD << std::endl;
    std::cout << "  - Total Samples: " << total_samples << std::endl;

    unsigned long long seed = 12345;
    
    unsigned long long *d_total_hits;
    checkCudaError(cudaMalloc(&d_total_hits, sizeof(unsigned long long)), "Malloc");
    checkCudaError(cudaMemset(d_total_hits, 0, sizeof(unsigned long long)), "Memset");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    estimate_pi_optimized<<<gridSize, BLOCK_SIZE>>>(d_total_hits, total_threads, seed);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_time_ms = 0.0f;
    cudaEventElapsedTime(&kernel_time_ms, start, stop);

    unsigned long long h_total_hits;
    checkCudaError(cudaMemcpy(&h_total_hits, d_total_hits, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "Memcpy");

    double pi_estimate = 4.0 * (double)h_total_hits / (double)total_samples;
    double error = std::fabs(pi_estimate - M_PI);
    
    double throughput = (double)total_samples / (kernel_time_ms / 1000.0);
    double giga_samples = throughput / 1e9;

    std::cout << "\nPerformance Metrics:\n";
    std::cout << "------------------------\n";
    std::cout << "Estimated Pi: " << pi_estimate << std::endl;
    std::cout << "Error: " << error << std::endl;
    std::cout << "Kernel Time: " << kernel_time_ms << " ms" << std::endl;
    std::cout << "Throughput: " << throughput << " samples/s (" << giga_samples << " GS/s)" << std::endl;
    std::cout << "------------------------\n";

    cudaFree(d_total_hits);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return EXIT_SUCCESS;
}