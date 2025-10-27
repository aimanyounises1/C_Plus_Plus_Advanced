/*
 * ==================================================================================================
 * Exercise: Cache Basics and Performance
 * ==================================================================================================
 * Difficulty: Advanced | Time: 50-60 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Understand CPU cache hierarchy (L1, L2, L3)
 * 2. Learn cache lines and cache misses
 * 3. Master spatial and temporal locality
 * 4. Practice cache-aware programming
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - GPU memory hierarchy (global, shared, L1/L2)
 * - Coalesced memory access patterns
 * - Cache line optimization for CPU-GPU transfers
 * - Understanding memory bandwidth bottlenecks
 * ==================================================================================================
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
using namespace std;
using namespace chrono;

/*
 * EXERCISE 1: Cache Line Effects (15 min)
 * Typical cache line: 64 bytes
 */

void rowMajorAccess(int** matrix, int rows, int cols) {
    // Good: Access consecutive memory (spatial locality)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j]++;
        }
    }
}

void columnMajorAccess(int** matrix, int rows, int cols) {
    // Bad: Jump between cache lines (many cache misses)
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            matrix[i][j]++;
        }
    }
}

void cacheLineDemo() {
    const int SIZE = 1000;

    // Allocate 2D array
    int** matrix = new int*[SIZE];
    for (int i = 0; i < SIZE; i++) {
        matrix[i] = new int[SIZE];
        memset(matrix[i], 0, SIZE * sizeof(int));
    }

    // Row-major (cache-friendly)
    auto start = high_resolution_clock::now();
    rowMajorAccess(matrix, SIZE, SIZE);
    auto end = high_resolution_clock::now();
    auto rowTime = duration_cast<microseconds>(end - start).count();

    // Column-major (cache-unfriendly)
    start = high_resolution_clock::now();
    columnMajorAccess(matrix, SIZE, SIZE);
    end = high_resolution_clock::now();
    auto colTime = duration_cast<microseconds>(end - start).count();

    cout << "Row-major access: " << rowTime << " µs" << endl;
    cout << "Column-major access: " << colTime << " µs" << endl;
    cout << "Slowdown: " << (double)colTime / rowTime << "x" << endl;

    // Cleanup
    for (int i = 0; i < SIZE; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

/*
 * EXERCISE 2: Temporal Locality (10 min)
 * Reusing recently accessed data
 */

void poorTemporalLocality(vector<int>& data) {
    int n = data.size();
    // Access data once, then never again
    for (int i = 0; i < n; i++) {
        data[i] = i * 2;
    }
    for (int i = 0; i < n; i++) {
        data[i] += 10;
    }
    for (int i = 0; i < n; i++) {
        data[i] *= 3;
    }
}

void goodTemporalLocality(vector<int>& data) {
    int n = data.size();
    // Process each element completely while in cache
    for (int i = 0; i < n; i++) {
        data[i] = i * 2;
        data[i] += 10;
        data[i] *= 3;
    }
}

void temporalLocalityDemo() {
    const int SIZE = 10000000;
    vector<int> data1(SIZE), data2(SIZE);

    auto start = high_resolution_clock::now();
    poorTemporalLocality(data1);
    auto end = high_resolution_clock::now();
    auto poorTime = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    goodTemporalLocality(data2);
    end = high_resolution_clock::now();
    auto goodTime = duration_cast<microseconds>(end - start).count();

    cout << "Poor temporal locality: " << poorTime << " µs" << endl;
    cout << "Good temporal locality: " << goodTime << " µs" << endl;
    cout << "Speedup: " << (double)poorTime / goodTime << "x" << endl;
}

/*
 * EXERCISE 3: Cache Blocking (Tiling) (15 min)
 * Divide computation into cache-sized blocks
 */

void naiveMatrixMultiply(const vector<vector<double>>& A,
                         const vector<vector<double>>& B,
                         vector<vector<double>>& C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void blockedMatrixMultiply(const vector<vector<double>>& A,
                           const vector<vector<double>>& B,
                           vector<vector<double>>& C, int n, int blockSize) {
    // Process matrix in cache-friendly blocks
    for (int i0 = 0; i0 < n; i0 += blockSize) {
        for (int j0 = 0; j0 < n; j0 += blockSize) {
            for (int k0 = 0; k0 < n; k0 += blockSize) {
                // Process block
                for (int i = i0; i < min(i0 + blockSize, n); i++) {
                    for (int j = j0; j < min(j0 + blockSize, n); j++) {
                        for (int k = k0; k < min(k0 + blockSize, n); k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

void cacheBlockingDemo() {
    const int N = 256;
    const int BLOCK_SIZE = 32;

    vector<vector<double>> A(N, vector<double>(N, 1.0));
    vector<vector<double>> B(N, vector<double>(N, 1.0));
    vector<vector<double>> C1(N, vector<double>(N, 0.0));
    vector<vector<double>> C2(N, vector<double>(N, 0.0));

    auto start = high_resolution_clock::now();
    naiveMatrixMultiply(A, B, C1, N);
    auto end = high_resolution_clock::now();
    auto naiveTime = duration_cast<milliseconds>(end - start).count();

    start = high_resolution_clock::now();
    blockedMatrixMultiply(A, B, C2, N, BLOCK_SIZE);
    end = high_resolution_clock::now();
    auto blockedTime = duration_cast<milliseconds>(end - start).count();

    cout << "Naive multiply: " << naiveTime << " ms" << endl;
    cout << "Blocked multiply: " << blockedTime << " ms" << endl;
    cout << "Speedup: " << (double)naiveTime / blockedTime << "x" << endl;
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is a cache?
 * A: Small, fast memory between CPU and main memory
 *    Stores frequently accessed data for faster access
 *
 * Q2: CPU cache hierarchy?
 * A: L1: ~32KB, 1-2 cycles, per-core
 *    L2: ~256KB, ~10 cycles, per-core
 *    L3: ~8MB, ~40 cycles, shared
 *    Main memory: GBs, ~200 cycles
 *
 * Q3: What is a cache line?
 * A: Smallest unit of cache (typically 64 bytes)
 *    Reading 1 byte loads entire cache line
 *
 * Q4: Types of cache misses?
 * A: Compulsory: First access (unavoidable)
 *    Capacity: Cache too small for working set
 *    Conflict: Multiple addresses map to same cache line
 *
 * Q5: What is spatial locality?
 * A: Accessing nearby memory locations
 *    Example: Array traversal in order
 *
 * Q6: What is temporal locality?
 * A: Reusing same memory location soon after first access
 *    Example: Loop variable
 *
 * Q7: How to improve cache performance?
 * A: - Access data sequentially (spatial locality)
 *    - Reuse data while in cache (temporal locality)
 *    - Use cache blocking/tiling
 *    - Align data structures
 *    - Minimize working set size
 *
 * Q8: What is cache blocking?
 * A: Divide computation into cache-sized chunks
 *    Process each block completely before moving to next
 *    Keeps working set in cache
 *
 * Q9: Row-major vs column-major?
 * A: C/C++ uses row-major (rows contiguous in memory)
 *    Access rows for better cache performance
 *
 * Q10: How to measure cache performance?
 * A: - Profiling tools (perf, vtune)
 *    - Hardware performance counters
 *    - Cache miss rates
 *    - Memory bandwidth utilization
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 *
 * GPU Memory Hierarchy (similar concepts):
 * - Global memory: Like RAM (slow, large, ~200 cycles)
 * - L2 cache: Shared across SMs (~5MB, automatic)
 * - L1 cache/Shared memory: Per-SM (~128KB combined)
 * - Registers: Per-thread (fastest, limited)
 *
 * Cache optimization transfers directly to GPU:
 * 1. Coalesced access = Spatial locality
 *    - Threads in warp access consecutive addresses
 *    - 128-byte cache lines (2x CPU)
 *
 * 2. Shared memory = Manual cache
 *    - Explicitly managed L1 cache
 *    - Block/tile data in shared memory
 *
 * 3. Matrix multiply optimization:
 *    - CPU: Cache blocking
 *    - GPU: Tile into shared memory
 *    - Same principle, different implementation
 *
 * Example GPU equivalent:
 * __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
 * // Load tile into shared memory (manual caching)
 * // Compute on tile while in fast memory
 * // Same as cache blocking on CPU!
 *
 * COMPILATION: g++ -std=c++11 -O2 01_cache_basics.cpp -o cache_basics
 * ==================================================================================================
 */

int main() {
    cout << "=== Cache Performance Practice ===" << endl;

    cout << "\n1. Cache Line Effects (Row vs Column Major):" << endl;
    cacheLineDemo();

    cout << "\n2. Temporal Locality:" << endl;
    temporalLocalityDemo();

    cout << "\n3. Cache Blocking (Matrix Multiply):" << endl;
    cacheBlockingDemo();

    return 0;
}
