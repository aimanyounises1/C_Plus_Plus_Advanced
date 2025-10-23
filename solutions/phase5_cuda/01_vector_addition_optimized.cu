/*
 * CUDA Vector Addition - Multiple Optimization Levels
 *
 * This example demonstrates the progression from naive to optimized CUDA kernels.
 * Essential for Nvidia interviews - shows understanding of:
 * - Memory coalescing
 * - Grid-stride loops
 * - Error handling
 * - Performance optimization
 *
 * Compile: nvcc -o vector_add 01_vector_addition_optimized.cu -O3 -arch=sm_70
 * Run: ./vector_add
 */

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

// Error checking macro - ALWAYS use this in production code
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Version 1: Naive Implementation
// Each thread handles exactly one element
// ============================================================================
__global__ void vectorAddNaive(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ============================================================================
// Version 2: Grid-Stride Loop
// More flexible - handles any array size, better for large data
// Each thread can process multiple elements
// This is NVIDIA's recommended pattern!
// ============================================================================
__global__ void vectorAddGridStride(const float* a, const float* b, float* c, int n) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate grid stride (total number of threads in grid)
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop: each thread processes multiple elements
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// ============================================================================
// Version 3: Vectorized Memory Access (using float4)
// Improves memory bandwidth utilization
// Loads 128 bits (4 floats) per transaction instead of 32 bits
// ============================================================================
__global__ void vectorAddVectorized(const float4* a, const float4* b, float4* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        float4 a_val = a[i];
        float4 b_val = b[i];
        float4 c_val;

        c_val.x = a_val.x + b_val.x;
        c_val.y = a_val.y + b_val.y;
        c_val.z = a_val.z + b_val.z;
        c_val.w = a_val.w + b_val.w;

        c[i] = c_val;
    }
}

// ============================================================================
// Host Functions
// ============================================================================

void initializeArray(float* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool verifyResult(const float* a, const float* b, const float* c, int n, float epsilon = 1e-5) {
    for (int i = 0; i < n; i++) {
        float expected = a[i] + b[i];
        if (std::abs(c[i] - expected) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": expected "
                      << expected << ", got " << c[i] << std::endl;
            return false;
        }
    }
    return true;
}

void benchmarkKernel(const char* name,
                     void (*kernel)(const float*, const float*, float*, int),
                     const float* d_a, const float* d_b, float* d_c,
                     int n, int blockSize, int numBlocks) {

    // Warm-up run
    kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    const int numRuns = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numRuns; i++) {
        kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avgTime = duration.count() / (double)numRuns;
    double bandwidth = (3.0 * n * sizeof(float)) / (avgTime * 1e-6) / 1e9; // GB/s

    std::cout << name << ":\n";
    std::cout << "  Average time: " << avgTime << " μs\n";
    std::cout << "  Bandwidth: " << bandwidth << " GB/s\n";
    std::cout << std::endl;
}

int main() {
    const int N = 32 * 1024 * 1024; // 32M elements (128 MB per array)
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    std::cout << "CUDA Vector Addition Performance Comparison\n";
    std::cout << "============================================\n";
    std::cout << "Array size: " << N << " elements (" << (N * sizeof(float) / 1e6) << " MB)\n";
    std::cout << "Block size: " << blockSize << "\n";
    std::cout << "Num blocks: " << numBlocks << "\n\n";

    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Memory Bandwidth: " << (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6) << " GB/s\n\n";

    // Allocate host memory
    float *h_a, *h_b, *h_c;
    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];

    // Initialize arrays
    initializeArray(h_a, N);
    initializeArray(h_b, N);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    // ========================================================================
    // Benchmark Version 1: Naive
    // ========================================================================
    benchmarkKernel("Version 1: Naive", vectorAddNaive, d_a, d_b, d_c, N, blockSize, numBlocks);

    // Verify correctness
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    if (!verifyResult(h_a, h_b, h_c, N)) {
        std::cerr << "Naive version failed verification!\n";
        return 1;
    }

    // ========================================================================
    // Benchmark Version 2: Grid-Stride
    // ========================================================================
    // Use fewer blocks to show grid-stride advantage
    const int optimalBlocks = 256; // Enough to saturate GPU
    benchmarkKernel("Version 2: Grid-Stride", vectorAddGridStride, d_a, d_b, d_c, N, blockSize, optimalBlocks);

    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    if (!verifyResult(h_a, h_b, h_c, N)) {
        std::cerr << "Grid-stride version failed verification!\n";
        return 1;
    }

    // ========================================================================
    // Benchmark Version 3: Vectorized (if N is divisible by 4)
    // ========================================================================
    if (N % 4 == 0) {
        const int N_vec = N / 4;
        const int numBlocks_vec = (N_vec + blockSize - 1) / blockSize;

        // Reinterpret pointers as float4
        float4* d_a_vec = reinterpret_cast<float4*>(d_a);
        float4* d_b_vec = reinterpret_cast<float4*>(d_b);
        float4* d_c_vec = reinterpret_cast<float4*>(d_c);

        // Warm-up
        vectorAddVectorized<<<numBlocks_vec, blockSize>>>(d_a_vec, d_b_vec, d_c_vec, N_vec);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        const int numRuns = 100;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < numRuns; i++) {
            vectorAddVectorized<<<numBlocks_vec, blockSize>>>(d_a_vec, d_b_vec, d_c_vec, N_vec);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double avgTime = duration.count() / (double)numRuns;
        double bandwidth = (3.0 * N * sizeof(float)) / (avgTime * 1e-6) / 1e9;

        std::cout << "Version 3: Vectorized (float4):\n";
        std::cout << "  Average time: " << avgTime << " μs\n";
        std::cout << "  Bandwidth: " << bandwidth << " GB/s\n";
        std::cout << std::endl;

        CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
        if (!verifyResult(h_a, h_b, h_c, N)) {
            std::cerr << "Vectorized version failed verification!\n";
            return 1;
        }
    }

    std::cout << "All versions passed verification! ✓\n";

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}

/*
 * KEY TAKEAWAYS FOR NVIDIA INTERVIEWS:
 *
 * 1. Memory Bandwidth: Vector addition is memory-bound, not compute-bound.
 *    Performance is limited by how fast you can move data.
 *
 * 2. Grid-Stride Loop: NVIDIA's recommended pattern. Benefits:
 *    - Scalable to any problem size
 *    - Better for large datasets
 *    - More flexible kernel launch configuration
 *
 * 3. Vectorized Access: Using float4 can improve bandwidth utilization
 *    by 2-4x in some cases, but requires aligned memory.
 *
 * 4. Always Profile: Use Nsight Compute to see actual memory throughput:
 *    ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./vector_add
 *
 * 5. Error Handling: Production code MUST check CUDA errors. Use macros
 *    or helper functions.
 *
 * INTERVIEW QUESTIONS TO EXPECT:
 * - Why is vector addition memory-bound?
 * - What is memory coalescing and why does it matter?
 * - How do you choose block size and grid size?
 * - What's the difference between global, shared, and register memory?
 * - How would you optimize this further?
 *
 * PROFILING COMMANDS:
 * - nvprof ./vector_add                          (Basic timing)
 * - ncu --set full ./vector_add                  (Detailed metrics)
 * - ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./vector_add
 */
