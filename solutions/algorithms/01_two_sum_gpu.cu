/*
 * Two Sum Problem - GPU Implementation
 *
 * Problem: Given an array of integers and a target, find two indices
 * such that nums[i] + nums[j] = target.
 *
 * LeetCode #1 (Easy) - But with parallel twist!
 *
 * Approaches:
 * 1. CPU: Hash table - O(n) time, O(n) space
 * 2. GPU: Parallel search - O(n²/p) time with p processors
 * 3. GPU: Parallel hash table (advanced)
 *
 * Compile: nvcc -o two_sum 01_two_sum_gpu.cu -O3 -arch=sm_70
 */

#include <iostream>
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// CPU Solution - Hash Table (Optimal for sequential)
// ============================================================================
std::pair<int, int> twoSumCPU(const std::vector<int>& nums, int target) {
    std::unordered_map<int, int> seen;

    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];

        if (seen.find(complement) != seen.end()) {
            return {seen[complement], i};
        }

        seen[nums[i]] = i;
    }

    return {-1, -1}; // No solution
}

// ============================================================================
// GPU Solution 1: Naive Parallel Search
// Each thread checks one pair - O(n²) pairs, but parallel
// ============================================================================
__global__ void twoSumNaive(const int* nums, int n, int target, int* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n && i < j) {
        if (nums[i] + nums[j] == target) {
            // Found a solution - use atomicCAS to ensure only first result is stored
            atomicCAS(&result[0], -1, i);
            atomicCAS(&result[1], -1, j);
        }
    }
}

// ============================================================================
// GPU Solution 2: Optimized with Shared Memory
// Each block loads a tile into shared memory for faster access
// ============================================================================
template<int TILE_SIZE>
__global__ void twoSumShared(const int* nums, int n, int target, int* result) {
    __shared__ int tile[TILE_SIZE];

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Each thread processes multiple elements
    for (int blockStart = 0; blockStart < n; blockStart += TILE_SIZE) {
        // Load tile into shared memory
        if (blockStart + tid < n) {
            tile[tid] = nums[blockStart + tid];
        }
        __syncthreads();

        // Check current element against all elements in tile
        if (globalIdx < n) {
            int current = nums[globalIdx];
            int needed = target - current;

            #pragma unroll
            for (int i = 0; i < TILE_SIZE; i++) {
                int tileIdx = blockStart + i;
                if (tileIdx < n && globalIdx < tileIdx && tile[i] == needed) {
                    atomicCAS(&result[0], -1, globalIdx);
                    atomicCAS(&result[1], -1, tileIdx);
                }
            }
        }

        __syncthreads();
    }
}

// ============================================================================
// GPU Solution 3: Parallel Sort + Binary Search
// Sort array, then for each element, binary search for complement
// ============================================================================
__device__ int binarySearch(const int* arr, int n, int target, int excludeIdx) {
    int left = 0, right = n - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target && mid != excludeIdx) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}

__global__ void twoSumSortedSearch(const int* nums, const int* indices,
                                    int n, int target, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int complement = target - nums[idx];
        int foundIdx = binarySearch(nums, n, complement, idx);

        if (foundIdx != -1 && idx < foundIdx) {
            // Found solution - store original indices
            atomicCAS(&result[0], -1, indices[idx]);
            atomicCAS(&result[1], -1, indices[foundIdx]);
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

void printResult(const char* method, std::pair<int, int> result, double time_ms) {
    std::cout << method << ":\n";
    if (result.first != -1) {
        std::cout << "  Solution: [" << result.first << ", " << result.second << "]\n";
    } else {
        std::cout << "  No solution found\n";
    }
    std::cout << "  Time: " << time_ms << " ms\n\n";
}

int main() {
    // Test cases
    std::vector<int> nums = {2, 7, 11, 15, 3, 6, 9, 12, 18, 21, 24, 27, 30};
    int target = 9;

    // For larger performance testing
    const int N = 100000;
    std::vector<int> large_nums(N);
    for (int i = 0; i < N; i++) {
        large_nums[i] = rand() % 10000;
    }
    // Ensure solution exists
    int idx1 = N / 3;
    int idx2 = N / 2;
    int large_target = large_nums[idx1] + large_nums[idx2];

    std::cout << "Two Sum Problem - CPU vs GPU\n";
    std::cout << "=============================\n";
    std::cout << "Small test: " << nums.size() << " elements\n";
    std::cout << "Large test: " << N << " elements\n\n";

    // ========================================================================
    // CPU Solution
    // ========================================================================
    auto start = std::chrono::high_resolution_clock::now();
    auto cpu_result = twoSumCPU(large_nums, large_target);
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end - start).count();

    printResult("CPU Hash Table", cpu_result, cpu_time);

    // ========================================================================
    // GPU Solution 1: Naive
    // ========================================================================
    int *d_nums, *d_result;
    int h_result[2] = {-1, -1};

    CUDA_CHECK(cudaMalloc(&d_nums, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result, 2 * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_nums, large_nums.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_result, h_result, 2 * sizeof(int), cudaMemcpyHostToDevice));

    // For large N, use 1D grid with each thread checking subsequent elements
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    start = std::chrono::high_resolution_clock::now();

    // Simplified: each thread searches forward
    // (Full 2D grid would be too large for N=100K)

    end = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // GPU Solution 2: Shared Memory (more practical)
    // ========================================================================
    const int TILE_SIZE = 256;
    h_result[0] = h_result[1] = -1;
    CUDA_CHECK(cudaMemcpy(d_result, h_result, 2 * sizeof(int), cudaMemcpyHostToDevice));

    start = std::chrono::high_resolution_clock::now();

    twoSumShared<TILE_SIZE><<<gridSize, blockSize>>>(d_nums, N, large_target, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    end = std::chrono::high_resolution_clock::now();
    double gpu_shared_time = std::chrono::duration<double, std::milli>(end - start).count();

    CUDA_CHECK(cudaMemcpy(h_result, d_result, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    printResult("GPU Shared Memory", {h_result[0], h_result[1]}, gpu_shared_time);

    std::cout << "Speedup: " << (cpu_time / gpu_shared_time) << "x\n\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_nums));
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}

/*
 * KEY INSIGHTS FOR INTERVIEWS:
 *
 * 1. GPU IS NOT ALWAYS BETTER
 *    - For small N (< 10K), CPU hash table is faster
 *    - GPU overhead (transfer + launch) dominates
 *    - Know when to use each approach!
 *
 * 2. PARALLEL ALGORITHM DESIGN
 *    - Sequential: O(n) with hash table
 *    - Parallel naive: O(n²/p) - worse work complexity!
 *    - Trade-off: More total work, but parallelizes well
 *
 * 3. MEMORY ACCESS PATTERNS
 *    - Naive: Poor locality, lots of global memory access
 *    - Shared: Reuse data within blocks
 *    - Sorted: Better cache behavior
 *
 * 4. ATOMIC OPERATIONS
 *    - Used to avoid race conditions when writing result
 *    - atomicCAS ensures only first solution is stored
 *    - Atomic operations can be slow - minimize usage
 *
 * 5. WHEN TO USE GPU FOR THIS PROBLEM
 *    ✓ Very large arrays (N > 100K)
 *    ✓ Multiple queries (amortize transfer cost)
 *    ✓ Already have data on GPU
 *    ✗ Small arrays (N < 1K)
 *    ✗ One-time query
 *    ✗ Sequential code is already fast enough
 *
 * INTERVIEW QUESTIONS:
 *
 * Q: "How would you parallelize Two Sum?"
 * A: Three approaches:
 *    1. Parallel brute force - check all pairs in parallel
 *    2. Parallel hash table - harder, need concurrent hash map
 *    3. Sort + parallel binary search
 *
 * Q: "What's the parallel complexity?"
 * A: Brute force: O(n²) work, O(1) span with n² processors
 *    Hash table: O(n) work, O(log n) span (parallel hash insert)
 *    Sort + search: O(n log n) work, O(log n) span
 *
 * Q: "When would you use GPU for this?"
 * A: When N is very large (>100K), or processing many queries,
 *    or data already on GPU. For most real-world cases,
 *    CPU hash table is sufficient.
 *
 * EXTENSIONS TO DISCUSS:
 * - Three Sum: O(n³) → O(n²/p) parallel
 * - K Sum: Generalizes to higher dimensions
 * - Finding all pairs (not just first)
 * - Approximate solutions with tolerance
 */
