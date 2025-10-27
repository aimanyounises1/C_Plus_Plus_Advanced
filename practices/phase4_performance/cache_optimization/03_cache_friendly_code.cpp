/*
 * ==================================================================================================
 * Exercise: Writing Cache-Friendly Code
 * ==================================================================================================
 * Difficulty: Advanced | Time: 45-55 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master data structure layout optimization
 * 2. Learn AoS vs SoA patterns
 * 3. Practice loop fusion and interchange
 * 4. Understand memory access patterns
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Structure of Arrays (SoA) for GPU coalescing
 * - Memory layout optimization for kernels
 * - Warp-level memory access patterns
 * - Minimizing global memory transactions
 * ==================================================================================================
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
using namespace std;
using namespace chrono;

/*
 * EXERCISE 1: Array of Structures (AoS) vs Structure of Arrays (SoA) (15 min)
 */

// AoS: Data interleaved (poor cache usage)
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};

// SoA: Data separated (better cache usage)
struct ParticlesSoA {
    vector<float> x, y, z;
    vector<float> vx, vy, vz;
};

void updateAoS(vector<Particle>& particles, float dt) {
    for (size_t i = 0; i < particles.size(); i++) {
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
}

void updateSoA(ParticlesSoA& particles, float dt, size_t count) {
    // Access only position arrays (better cache locality)
    for (size_t i = 0; i < count; i++) {
        particles.x[i] += particles.vx[i] * dt;
        particles.y[i] += particles.vy[i] * dt;
        particles.z[i] += particles.vz[i] * dt;
    }
}

void aosVsSoaDemo() {
    const int N = 1000000;
    const float dt = 0.01f;

    // AoS setup
    vector<Particle> particlesAoS(N);
    for (int i = 0; i < N; i++) {
        particlesAoS[i] = {1.0f, 1.0f, 1.0f, 0.1f, 0.1f, 0.1f};
    }

    // SoA setup
    ParticlesSoA particlesSoA;
    particlesSoA.x.resize(N, 1.0f);
    particlesSoA.y.resize(N, 1.0f);
    particlesSoA.z.resize(N, 1.0f);
    particlesSoA.vx.resize(N, 0.1f);
    particlesSoA.vy.resize(N, 0.1f);
    particlesSoA.vz.resize(N, 0.1f);

    // Benchmark AoS
    auto start = high_resolution_clock::now();
    for (int iter = 0; iter < 100; iter++) {
        updateAoS(particlesAoS, dt);
    }
    auto end = high_resolution_clock::now();
    auto aosTime = duration_cast<milliseconds>(end - start).count();

    // Benchmark SoA
    start = high_resolution_clock::now();
    for (int iter = 0; iter < 100; iter++) {
        updateSoA(particlesSoA, dt, N);
    }
    end = high_resolution_clock::now();
    auto soaTime = duration_cast<milliseconds>(end - start).count();

    cout << "AoS time: " << aosTime << " ms" << endl;
    cout << "SoA time: " << soaTime << " ms" << endl;
    cout << "Speedup: " << (double)aosTime / soaTime << "x" << endl;
}

/*
 * EXERCISE 2: Loop Fusion (10 min)
 */

void separateLoops(vector<float>& a, vector<float>& b, vector<float>& c, int n) {
    // Multiple passes over data (poor cache reuse)
    for (int i = 0; i < n; i++) {
        a[i] = b[i] + 1.0f;
    }
    for (int i = 0; i < n; i++) {
        c[i] = a[i] * 2.0f;
    }
}

void fusedLoop(vector<float>& a, vector<float>& b, vector<float>& c, int n) {
    // Single pass (better cache reuse)
    for (int i = 0; i < n; i++) {
        a[i] = b[i] + 1.0f;
        c[i] = a[i] * 2.0f;
    }
}

void loopFusionDemo() {
    const int N = 10000000;
    vector<float> a1(N), b(N, 1.0f), c1(N);
    vector<float> a2(N), c2(N);

    auto start = high_resolution_clock::now();
    separateLoops(a1, b, c1, N);
    auto end = high_resolution_clock::now();
    auto separateTime = duration_cast<milliseconds>(end - start).count();

    start = high_resolution_clock::now();
    fusedLoop(a2, b, c2, N);
    end = high_resolution_clock::now();
    auto fusedTime = duration_cast<milliseconds>(end - start).count();

    cout << "Separate loops: " << separateTime << " ms" << endl;
    cout << "Fused loop: " << fusedTime << " ms" << endl;
    cout << "Speedup: " << (double)separateTime / fusedTime << "x" << endl;
}

/*
 * EXERCISE 3: Loop Interchange (10 min)
 */

void poorLoopOrder(vector<vector<int>>& matrix, int rows, int cols) {
    // Column-major (cache-unfriendly for row-major storage)
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            matrix[i][j] += 1;
        }
    }
}

void goodLoopOrder(vector<vector<int>>& matrix, int rows, int cols) {
    // Row-major (cache-friendly)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] += 1;
        }
    }
}

void loopInterchangeDemo() {
    const int SIZE = 1000;
    vector<vector<int>> matrix1(SIZE, vector<int>(SIZE, 0));
    vector<vector<int>> matrix2(SIZE, vector<int>(SIZE, 0));

    auto start = high_resolution_clock::now();
    poorLoopOrder(matrix1, SIZE, SIZE);
    auto end = high_resolution_clock::now();
    auto poorTime = duration_cast<milliseconds>(end - start).count();

    start = high_resolution_clock::now();
    goodLoopOrder(matrix2, SIZE, SIZE);
    end = high_resolution_clock::now();
    auto goodTime = duration_cast<milliseconds>(end - start).count();

    cout << "Poor loop order: " << poorTime << " ms" << endl;
    cout << "Good loop order: " << goodTime << " ms" << endl;
    cout << "Speedup: " << (double)poorTime / goodTime << "x" << endl;
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: AoS vs SoA?
 * A: AoS (Array of Structures): {x,y,z}, {x,y,z}, {x,y,z}
 *    SoA (Structure of Arrays): {x,x,x}, {y,y,y}, {z,z,z}
 *    SoA has better cache locality when accessing single field
 *
 * Q2: When to use SoA?
 * A: - Processing subset of fields
 *    - SIMD vectorization
 *    - GPU coalesced access
 *    - High-performance computing
 *
 * Q3: What is loop fusion?
 * A: Combining multiple loops into one
 *    Improves cache reuse, reduces loop overhead
 *
 * Q4: What is loop interchange?
 * A: Swapping inner and outer loops
 *    Use to access memory in sequential order
 *
 * Q5: Data layout best practices?
 * A: - Match access pattern to memory layout
 *    - Group frequently accessed fields
 *    - Separate hot and cold data
 *    - Align to cache line boundaries
 *
 * Q6: How to identify cache-unfriendly code?
 * A: - Large stride access (skipping elements)
 *    - Random/pointer-chasing access
 *    - Working set exceeds cache size
 *    - Frequent data structure traversal
 *
 * Q7: Loop optimizations for cache?
 * A: - Loop fusion (merge loops)
 *    - Loop interchange (swap loops)
 *    - Loop tiling/blocking (chunk processing)
 *    - Loop unrolling (reduce overhead)
 *
 * Q8: Impact of cache-friendly code?
 * A: Can achieve 2-10x speedup just from better layout
 *    No algorithm change, just memory organization
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 *
 * Cache-friendly patterns are ESSENTIAL for GPU performance:
 *
 * 1. SoA is MANDATORY for GPU coalescing:
 *    // BAD (AoS) - Uncoalesced access
 *    struct Particle { float x, y, z; };
 *    Particle* particles;
 *    // Thread 0 reads particles[0].x
 *    // Thread 1 reads particles[1].x
 *    // Threads access non-consecutive addresses!
 *
 *    // GOOD (SoA) - Coalesced access
 *    float *x, *y, *z;
 *    // Thread 0 reads x[0]
 *    // Thread 1 reads x[1]
 *    // Threads access consecutive addresses - coalesced!
 *
 * 2. Loop patterns in CUDA:
 *    // CPU: Loop fusion
 *    for (i) { a[i] = ...; c[i] = a[i] * 2; }
 *
 *    // GPU: Each thread processes one element
 *    __global__ void kernel(float* a, float* c) {
 *        int i = blockIdx.x * blockDim.x + threadIdx.x;
 *        a[i] = ...;
 *        c[i] = a[i] * 2;  // Natural fusion
 *    }
 *
 * 3. Matrix operations:
 *    // CPU: Row-major loop order
 *    for (i) for (j) matrix[i][j]++;
 *
 *    // GPU: 2D thread layout
 *    int i = blockIdx.y * blockDim.y + threadIdx.y;
 *    int j = blockIdx.x * blockDim.x + threadIdx.x;
 *    matrix[i * width + j]++;  // Coalesced if j varies in warp
 *
 * Real NVIDIA interview question:
 * "Why is SoA better than AoS for GPU kernels?"
 * Answer: Coalesced memory access. Threads in a warp access consecutive
 * memory, maximizing memory bandwidth (one 128-byte transaction vs many).
 *
 * COMPILATION: g++ -std=c++11 -O2 03_cache_friendly_code.cpp -o cache_friendly
 * ==================================================================================================
 */

int main() {
    cout << "=== Cache-Friendly Code Practice ===" << endl;

    cout << "\n1. AoS vs SoA:" << endl;
    aosVsSoaDemo();

    cout << "\n2. Loop Fusion:" << endl;
    loopFusionDemo();

    cout << "\n3. Loop Interchange:" << endl;
    loopInterchangeDemo();

    return 0;
}
