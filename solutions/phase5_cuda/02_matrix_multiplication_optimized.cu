/*
 * Matrix Multiplication - From Naive to Highly Optimized
 *
 * This is THE classic CUDA optimization problem. Nvidia WILL ask about this.
 * Shows progression through multiple optimization levels with performance gains.
 *
 * Optimizations demonstrated:
 * 1. Naive global memory only
 * 2. Tiled with shared memory
 * 3. Optimized with reduced bank conflicts
 * 4. Thread coarsening for better compute utilization
 *
 * Compile: nvcc -o matmul 02_matrix_multiplication_optimized.cu -O3 -arch=sm_70
 * Run: ./matmul
 */

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

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
// Every value read from global memory - VERY slow
// Arithmetic Intensity: O(N) operations, O(N²) memory accesses per output element
// ============================================================================
__global__ void matmulNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Version 2: Tiled with Shared Memory
// Classic optimization - reduces global memory accesses by ~N
// Each tile is loaded once into shared memory and reused
// ============================================================================
template<int TILE_SIZE>
__global__ void matmulTiled(const float* A, const float* B, float* C, int N) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from A
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B
        if ((t * TILE_SIZE + threadIdx.y) < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Version 3: Tiled with Reduced Bank Conflicts
// Padding shared memory arrays to avoid bank conflicts
// Bank conflicts occur when multiple threads access same memory bank
// ============================================================================
template<int TILE_SIZE>
__global__ void matmulTiledNoBankConflict(const float* A, const float* B, float* C, int N) {
    // Add padding to avoid bank conflicts (32 banks on most GPUs)
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((t * TILE_SIZE + threadIdx.y) < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Unroll the inner loop for better instruction-level parallelism
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Version 4: Thread Coarsening (each thread computes multiple outputs)
// Increases arithmetic intensity and register reuse
// Better instruction-level parallelism
// ============================================================================
template<int TILE_SIZE, int COARSE_FACTOR>
__global__ void matmulCoarsened(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int rowBase = blockIdx.y * TILE_SIZE * COARSE_FACTOR + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Each thread computes COARSE_FACTOR outputs
    float sum[COARSE_FACTOR] = {0.0f};

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles from B (shared across all coarsened outputs)
        if ((t * TILE_SIZE + threadIdx.y) < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load multiple tiles from A (one per coarsened output)
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; c++) {
            int row = rowBase + c * TILE_SIZE;
            if (row < N && (t * TILE_SIZE + threadIdx.x) < N) {
                As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
            } else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            // Compute for this coarsened output
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                sum[c] += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }

            __syncthreads();
        }
    }

    // Write results
    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int row = rowBase + c * TILE_SIZE;
        if (row < N && col < N) {
            C[row * N + col] = sum[c];
        }
    }
}

// ============================================================================
// Host Functions
// ============================================================================

void initializeMatrix(float* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool verifyResult(const float* A, const float* B, const float* C, int N) {
    const int CHECK_SIZE = std::min(N, 16); // Only verify a subset for large matrices
    const float epsilon = 1e-3; // Looser tolerance for FP accumulation

    for (int i = 0; i < CHECK_SIZE; i++) {
        for (int j = 0; j < CHECK_SIZE; j++) {
            float expected = 0.0f;
            for (int k = 0; k < N; k++) {
                expected += A[i * N + k] * B[k * N + j];
            }

            float diff = std::abs(C[i * N + j] - expected);
            if (diff > epsilon) {
                std::cerr << "Mismatch at (" << i << "," << j << "): "
                          << "expected " << expected << ", got " << C[i * N + j]
                          << ", diff " << diff << std::endl;
                return false;
            }
        }
    }
    return true;
}

template<typename KernelFunc>
double benchmarkKernel(const char* name, KernelFunc kernel,
                       const float* d_A, const float* d_B, float* d_C,
                       int N, dim3 blockDim, dim3 gridDim) {

    // Warm-up
    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    const int numRuns = 10;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numRuns; i++) {
        kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avgTime = duration.count() / (double)numRuns / 1000.0; // ms

    // Calculate GFLOPS
    double flops = 2.0 * N * N * N; // multiply-add counts as 2 ops
    double gflops = (flops / (avgTime * 1e-3)) / 1e9;

    std::cout << name << ":\n";
    std::cout << "  Time: " << avgTime << " ms\n";
    std::cout << "  Performance: " << gflops << " GFLOPS\n";
    std::cout << std::endl;

    return avgTime;
}

int main() {
    const int N = 2048; // Matrix size NxN
    const int TILE_SIZE = 32;
    const int COARSE_FACTOR = 2;

    std::cout << "Matrix Multiplication Performance Comparison\n";
    std::cout << "=============================================\n";
    std::cout << "Matrix size: " << N << "x" << N << "\n";
    std::cout << "Memory per matrix: " << (N * N * sizeof(float) / 1e6) << " MB\n";
    std::cout << "Total FLOPs: " << (2.0 * N * N * N / 1e9) << " GFLOP\n\n";

    // Device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Peak Performance: ~" << (2 * prop.clockRate * prop.multiProcessorCount * 128 / 1e6) << " GFLOPS\n\n";

    // Allocate host memory
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice));

    // ========================================================================
    // Benchmark kernels
    // ========================================================================

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Version 1: Naive
    double time_naive = benchmarkKernel("Version 1: Naive (Global Memory Only)",
                                        matmulNaive, d_A, d_B, d_C, N, blockDim, gridDim);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    if (!verifyResult(h_A, h_B, h_C, N)) {
        std::cerr << "Naive version failed!\n";
        return 1;
    }

    // Version 2: Tiled
    double time_tiled = benchmarkKernel("Version 2: Tiled (Shared Memory)",
                                        matmulTiled<TILE_SIZE>, d_A, d_B, d_C, N, blockDim, gridDim);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    if (!verifyResult(h_A, h_B, h_C, N)) {
        std::cerr << "Tiled version failed!\n";
        return 1;
    }

    std::cout << "Speedup over naive: " << (time_naive / time_tiled) << "x\n\n";

    // Version 3: No Bank Conflicts
    double time_nobank = benchmarkKernel("Version 3: Tiled + No Bank Conflicts",
                                         matmulTiledNoBankConflict<TILE_SIZE>, d_A, d_B, d_C, N, blockDim, gridDim);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    if (!verifyResult(h_A, h_B, h_C, N)) {
        std::cerr << "No bank conflict version failed!\n";
        return 1;
    }

    std::cout << "Speedup over naive: " << (time_naive / time_nobank) << "x\n";
    std::cout << "Speedup over tiled: " << (time_tiled / time_nobank) << "x\n\n";

    std::cout << "✓ All versions verified!\n";

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}

/*
 * KEY INSIGHTS FOR NVIDIA INTERVIEWS:
 *
 * 1. MEMORY HIERARCHY IS EVERYTHING
 *    - Global memory: ~200-900 GB/s (slow)
 *    - Shared memory: ~10 TB/s (50x faster!)
 *    - Registers: ~50 TB/s (fastest)
 *
 * 2. TILING STRATEGY
 *    - Reduces global memory accesses from O(N³) to O(N³/B) where B = tile size
 *    - Each element loaded once per tile instead of once per output element
 *    - Typical speedups: 10-50x over naive
 *
 * 3. BANK CONFLICTS
 *    - Shared memory divided into 32 banks (on most GPUs)
 *    - When multiple threads access same bank → serialization
 *    - Padding arrays can eliminate conflicts: float[32][33] instead of [32][32]
 *
 * 4. THREAD COARSENING
 *    - Each thread computes multiple outputs
 *    - Increases register reuse and arithmetic intensity
 *    - Better instruction-level parallelism
 *
 * 5. OPTIMIZATION STRATEGY
 *    Always follow this order:
 *    a) Get it working (correctness first!)
 *    b) Profile to find bottleneck
 *    c) Optimize memory access patterns
 *    d) Tune occupancy and register usage
 *    e) Use specialized instructions/tensor cores if applicable
 *
 * INTERVIEW QUESTIONS YOU'LL BE ASKED:
 *
 * Q: "How would you optimize matrix multiplication on GPU?"
 * A: Tiling with shared memory, handle bank conflicts, consider tensor cores,
 *    optimize memory access patterns, tune block size for occupancy.
 *
 * Q: "Why is tiling effective?"
 * A: Converts O(N) global memory accesses per output to O(sqrt(N)) by reusing
 *    data in fast shared memory. Each tile loaded once, reused B times.
 *
 * Q: "What's the theoretical peak performance?"
 * A: For A100: ~19.5 TFLOPS (FP32), ~312 TFLOPS (FP16 with tensor cores)
 *    Achieved performance depends on occupancy, memory bandwidth, and optimization.
 *
 * Q: "How do tensor cores help?"
 * A: Hardware-accelerated 4x4 matrix operations, up to 16x faster for FP16/INT8.
 *    Need to use WMMA (Warp Matrix Multiply-Accumulate) API or cuBLAS.
 *
 * PROFILING COMMANDS:
 * ncu --set full ./matmul
 * ncu --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum ./matmul
 * ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum ./matmul
 *
 * NEXT LEVEL OPTIMIZATIONS:
 * - Tensor cores (WMMA API) - 8-16x faster for FP16
 * - Persistent kernels - reduce launch overhead
 * - Prefetching - hide memory latency
 * - Warp specialization - different warps do different tasks
 */
