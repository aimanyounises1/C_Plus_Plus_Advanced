/*
 * Parallel Reduction - Essential CUDA Pattern
 *
 * Reduction (sum, max, min, etc.) is fundamental to parallel computing.
 * This example shows the evolution from naive to highly optimized.
 *
 * Common in Nvidia interviews - demonstrates understanding of:
 * - Shared memory and synchronization
 * - Warp-level primitives
 * - Sequential addressing vs interleaved
 * - Avoiding bank conflicts
 * - Warp shuffle instructions
 *
 * Compile: nvcc -o reduction 03_parallel_reduction.cu -O3 -arch=sm_70
 */

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <numeric>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Version 1: Interleaved Addressing (BAD - lots of divergence)
// Don't use this! Shown for educational purposes
// ============================================================================
__global__ void reduceInterleaved(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Interleaved addressing - CAUSES WARP DIVERGENCE
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {  // BAD: only half threads active in first iteration
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================================
// Version 2: Sequential Addressing (GOOD - no divergence)
// All threads in warp active together
// ============================================================================
__global__ void reduceSequential(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Sequential addressing - all threads in warp active
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================================
// Version 3: First Add During Load (reduces iterations)
// Each block processes 2x blockDim elements
// ============================================================================
__global__ void reduceFirstAdd(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // First add during global load - reduces shared memory operations
    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================================
// Version 4: Unroll Last Warp (no __syncthreads needed for last 32 threads)
// Warp executes in lockstep - no sync needed within warp
// ============================================================================
__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduceUnrollWarp(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // Reduce until we have 1 warp left
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Last warp - no __syncthreads needed
    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================================
// Version 5: Complete Unrolling (for fixed block size)
// Compiler can optimize better with compile-time constants
// ============================================================================
template<unsigned int blockSize>
__device__ void warpReduceTemplate(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template<unsigned int blockSize>
__global__ void reduceCompleteUnroll(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;

    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockSize < n) sum += input[i + blockSize];

    sdata[tid] = sum;
    __syncthreads();

    // Completely unroll reduction
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

    if (tid < 32) warpReduceTemplate<blockSize>(sdata, tid);

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================================
// Version 6: Warp Shuffle (FASTEST - no shared memory needed!)
// Modern approach using warp-level primitives
// Requires compute capability 3.0+
// ============================================================================
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduceWarpShuffle(float* input, float* output, int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load and first add
    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];

    // Warp-level reduction using shuffle
    sum = warpReduceSum(sum);

    // Shared memory only for cross-warp reduction
    __shared__ float warpSums[32]; // Max 32 warps per block (1024 threads)

    int lane = tid % 32;
    int wid = tid / 32;

    if (lane == 0) warpSums[wid] = sum;
    __syncthreads();

    // Final reduction by first warp
    if (tid < 32) {
        sum = (tid < (blockDim.x / 32)) ? warpSums[tid] : 0.0f;
        sum = warpReduceSum(sum);
    }

    if (tid == 0) output[blockIdx.x] = sum;
}

// ============================================================================
// Host Functions
// ============================================================================

template<typename KernelFunc>
float benchmarkReduction(const char* name, KernelFunc kernel,
                         float* d_input, float* d_output, float* d_temp,
                         int n, int blockSize, int gridSize,
                         int sharedMemSize = 0) {

    // Warm-up
    kernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // May need multiple passes
    int numPasses = 0;
    int currentSize = gridSize;
    float* currentInput = d_output;
    float* currentOutput = d_temp;

    while (currentSize > 1) {
        int nextGridSize = (currentSize + blockSize * 2 - 1) / (blockSize * 2);
        kernel<<<nextGridSize, blockSize, sharedMemSize>>>(currentInput, currentOutput, currentSize);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(currentInput, currentOutput);
        currentSize = nextGridSize;
        numPasses++;
    }

    // Timed runs
    const int numRuns = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int run = 0; run < numRuns; run++) {
        // First pass
        kernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, n);

        // Additional passes if needed
        currentSize = gridSize;
        currentInput = d_output;
        currentOutput = d_temp;

        while (currentSize > 1) {
            int nextGridSize = (currentSize + blockSize * 2 - 1) / (blockSize * 2);
            kernel<<<nextGridSize, blockSize, sharedMemSize>>>(currentInput, currentOutput, currentSize);

            std::swap(currentInput, currentOutput);
            currentSize = nextGridSize;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    float avgTime = duration.count() / (float)numRuns;
    float bandwidth = (n * sizeof(float)) / (avgTime * 1e-6) / 1e9;

    std::cout << name << ":\n";
    std::cout << "  Time: " << avgTime << " μs\n";
    std::cout << "  Bandwidth: " << bandwidth << " GB/s\n";
    std::cout << "  Passes needed: " << (numPasses + 1) << "\n";
    std::cout << std::endl;

    return avgTime;
}

int main() {
    const int N = 16 * 1024 * 1024; // 16M elements
    const int blockSize = 256;
    const int gridSize = (N + blockSize * 2 - 1) / (blockSize * 2);

    std::cout << "Parallel Reduction Performance Comparison\n";
    std::cout << "==========================================\n";
    std::cout << "Array size: " << N << " elements\n";
    std::cout << "Block size: " << blockSize << "\n";
    std::cout << "Grid size: " << gridSize << "\n\n";

    // Allocate host memory
    float* h_input = new float[N];
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f; // Simple test - sum should equal N
    }

    // CPU reference
    float cpu_sum = std::accumulate(h_input, h_input + N, 0.0f);
    std::cout << "Expected sum: " << cpu_sum << "\n\n";

    // Allocate device memory
    float *d_input, *d_output, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, gridSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp, gridSize * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    int sharedMemSize = blockSize * sizeof(float);

    // Benchmark all versions
    float time_seq = benchmarkReduction("Version 2: Sequential Addressing",
                                        reduceSequential, d_input, d_output, d_temp,
                                        N, blockSize, gridSize, sharedMemSize);

    float time_firstadd = benchmarkReduction("Version 3: First Add During Load",
                                             reduceFirstAdd, d_input, d_output, d_temp,
                                             N, blockSize, gridSize, sharedMemSize);

    float time_unroll = benchmarkReduction("Version 4: Unroll Last Warp",
                                           reduceUnrollWarp, d_input, d_output, d_temp,
                                           N, blockSize, gridSize, sharedMemSize);

    float time_complete = benchmarkReduction("Version 5: Complete Unroll",
                                             reduceCompleteUnroll<blockSize>, d_input, d_output, d_temp,
                                             N, blockSize, gridSize, sharedMemSize);

    float time_shuffle = benchmarkReduction("Version 6: Warp Shuffle",
                                            reduceWarpShuffle, d_input, d_output, d_temp,
                                            N, blockSize, gridSize, 0);

    // Verify result
    float h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "GPU result: " << h_result << "\n";
    std::cout << "Error: " << std::abs(h_result - cpu_sum) << "\n\n";

    std::cout << "Speedup Summary:\n";
    std::cout << "  First Add vs Sequential: " << (time_seq / time_firstadd) << "x\n";
    std::cout << "  Unroll Warp vs Sequential: " << (time_seq / time_unroll) << "x\n";
    std::cout << "  Complete Unroll vs Sequential: " << (time_seq / time_complete) << "x\n";
    std::cout << "  Warp Shuffle vs Sequential: " << (time_seq / time_shuffle) << "x\n";

    // Cleanup
    delete[] h_input;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_temp));

    return 0;
}

/*
 * KEY CONCEPTS FOR NVIDIA INTERVIEWS:
 *
 * 1. WARP DIVERGENCE
 *    - Threads in a warp execute in lockstep (SIMT)
 *    - Divergent branches serialize execution
 *    - Sequential addressing eliminates divergence
 *
 * 2. SHARED MEMORY vs REGISTERS
 *    - Shared: ~100 cycles latency, 48-164 KB per SM
 *    - Registers: ~1 cycle, 64K 32-bit registers per SM
 *    - Warp shuffle uses registers - much faster!
 *
 * 3. SYNCHRONIZATION COST
 *    - __syncthreads() has overhead
 *    - Warp is implicitly synchronized (SIMT)
 *    - Reducing sync points improves performance
 *
 * 4. OPTIMIZATION PROGRESSION
 *    Sequential (2-3x) → First Add (1.5x) → Unroll Warp (1.2x) → Shuffle (2x)
 *    Typical total speedup: 6-10x from naive to optimized
 *
 * 5. WARP-LEVEL PRIMITIVES (Modern CUDA)
 *    - __shfl_down_sync(): Share data within warp
 *    - __shfl_sync(): Broadcast from one lane
 *    - __shfl_up_sync(): Shift data up
 *    - __shfl_xor_sync(): XOR-based shuffle
 *    These are CRITICAL for modern high-performance CUDA!
 *
 * INTERVIEW QUESTIONS:
 *
 * Q: "Explain warp divergence and how to avoid it."
 * A: When threads in a warp take different branches, they serialize.
 *    Use sequential addressing instead of interleaved, ensure
 *    all threads in warp follow same control flow.
 *
 * Q: "Why is warp shuffle faster than shared memory?"
 * A: Shuffle uses register-to-register communication within warp,
 *    ~1 cycle latency. Shared memory ~100 cycles, requires sync.
 *
 * Q: "How would you reduce 1 billion elements?"
 * A: Multi-pass reduction: Each block reduces partial sum,
 *    then reduce block results. For 1B elements with 256 threads:
 *    Pass 1: 1B → 2M blocks
 *    Pass 2: 2M → 4K blocks
 *    Pass 3: 4K → 8 blocks
 *    Pass 4: 8 → 1 (final sum)
 *
 * Q: "What's the theoretical peak for reduction?"
 * A: Limited by memory bandwidth. For sum, arithmetic intensity
 *    is very low (1 add per 4-8 bytes loaded). Expect ~70-90%
 *    of peak memory bandwidth.
 *
 * RELATED PATTERNS:
 * - Scan (prefix sum) - builds on reduction
 * - Histogram - atomic operations on shared memory
 * - Sort - bitonic sort uses similar reduction structure
 *
 * PROFILING:
 * ncu --metrics smsp__average_warps_issue_stalled_short_scoreboard.pct ./reduction
 * ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./reduction
 */
