/*
 * ==================================================================================================
 * Exercise: Loop Optimizations
 * ==================================================================================================
 * Difficulty: Intermediate/Advanced | Time: 40-50 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master loop unrolling techniques
 * 2. Learn loop invariant code motion
 * 3. Practice loop fusion and fission
 * 4. Understand strength reduction
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Loop unrolling in CUDA kernels
 * - Warp-level optimizations
 * - Reducing loop overhead
 * - Maximizing instruction-level parallelism
 * ==================================================================================================
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
using namespace std;
using namespace chrono;

/*
 * EXERCISE 1: Loop Unrolling (15 min)
 */

int sumNormal(const vector<int>& data) {
    int sum = 0;
    for (size_t i = 0; i < data.size(); i++) {
        sum += data[i];
    }
    return sum;
}

int sumUnrolled4(const vector<int>& data) {
    int sum = 0;
    size_t i = 0;

    // Process 4 elements at a time
    for (; i + 3 < data.size(); i += 4) {
        sum += data[i];
        sum += data[i+1];
        sum += data[i+2];
        sum += data[i+3];
    }

    // Handle remaining elements
    for (; i < data.size(); i++) {
        sum += data[i];
    }

    return sum;
}

void loopUnrollingDemo() {
    const int N = 100000000;
    vector<int> data(N, 1);

    auto start = high_resolution_clock::now();
    int result1 = sumNormal(data);
    auto end = high_resolution_clock::now();
    auto normalTime = duration_cast<milliseconds>(end - start).count();

    start = high_resolution_clock::now();
    int result2 = sumUnrolled4(data);
    end = high_resolution_clock::now();
    auto unrolledTime = duration_cast<milliseconds>(end - start).count();

    cout << "Normal loop: " << normalTime << " ms (sum=" << result1 << ")" << endl;
    cout << "Unrolled x4: " << unrolledTime << " ms (sum=" << result2 << ")" << endl;
    cout << "Speedup: " << (double)normalTime / unrolledTime << "x" << endl;
}

/*
 * EXERCISE 2: Loop Invariant Code Motion (10 min)
 */

void poorInvariantHandling(vector<int>& data, int multiplier) {
    // BAD: Compute sqrt in every iteration
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = data[i] * sqrt(multiplier * multiplier);
    }
}

void goodInvariantHandling(vector<int>& data, int multiplier) {
    // GOOD: Hoist invariant computation
    int invariant = sqrt(multiplier * multiplier);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = data[i] * invariant;
    }
}

void loopInvariantDemo() {
    const int N = 10000000;
    vector<int> data1(N, 5), data2(N, 5);

    auto start = high_resolution_clock::now();
    poorInvariantHandling(data1, 3);
    auto end = high_resolution_clock::now();
    auto poorTime = duration_cast<milliseconds>(end - start).count();

    start = high_resolution_clock::now();
    goodInvariantHandling(data2, 3);
    end = high_resolution_clock::now();
    auto goodTime = duration_cast<milliseconds>(end - start).count();

    cout << "Poor handling: " << poorTime << " ms" << endl;
    cout << "Good handling: " << goodTime << " ms" << endl;
    cout << "Speedup: " << (double)poorTime / goodTime << "x" << endl;
}

/*
 * EXERCISE 3: Strength Reduction (10 min)
 */

void multiplicationInLoop(vector<int>& result, int n) {
    // BAD: Expensive multiplication every iteration
    for (int i = 0; i < n; i++) {
        result[i] = i * 7;
    }
}

void additionInLoop(vector<int>& result, int n) {
    // GOOD: Use addition instead of multiplication
    int value = 0;
    for (int i = 0; i < n; i++) {
        result[i] = value;
        value += 7;  // Faster than i * 7
    }
}

void strengthReductionDemo() {
    const int N = 100000000;
    vector<int> result1(N), result2(N);

    auto start = high_resolution_clock::now();
    multiplicationInLoop(result1, N);
    auto end = high_resolution_clock::now();
    auto multTime = duration_cast<milliseconds>(end - start).count();

    start = high_resolution_clock::now();
    additionInLoop(result2, N);
    end = high_resolution_clock::now();
    auto addTime = duration_cast<milliseconds>(end - start).count();

    cout << "Multiplication: " << multTime << " ms" << endl;
    cout << "Addition: " << addTime << " ms" << endl;
    cout << "Speedup: " << (double)multTime / addTime << "x" << endl;
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is loop unrolling?
 * A: Process multiple iterations in one loop iteration
 *    Reduces loop overhead and enables ILP
 *
 * Q2: Benefits of loop unrolling?
 * A: - Reduce branch mispredictions
 *    - Enable instruction-level parallelism
 *    - Better register utilization
 *    - Reduce loop counter overhead
 *
 * Q3: Downsides of unrolling?
 * A: - Increased code size
 *    - May hurt instruction cache
 *    - Compiler can do it automatically
 *
 * Q4: Loop invariant code motion?
 * A: Move computations that don't change out of loop
 *    Compiler usually does this at -O2+
 *
 * Q5: What is strength reduction?
 * A: Replace expensive operations with cheaper ones
 *    e.g., multiplication → addition
 *
 * Q6: Loop fusion vs fission?
 * A: Fusion: Combine multiple loops (better cache)
 *    Fission: Split loop (enable parallelization)
 *
 * Q7: When NOT to unroll?
 * A: - Loop body already large
 *    - Unpredictable loop count
 *    - Memory-bound (not compute-bound)
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 *
 * Loop optimizations are CRITICAL for GPU kernels:
 *
 * 1. Unrolling in CUDA:
 *    // Manual unroll
 *    #pragma unroll
 *    for (int i = 0; i < 4; i++) {
 *        sum += data[i];
 *    }
 *
 *    // Partial unroll
 *    #pragma unroll 4
 *    for (int i = 0; i < N; i++) {
 *        // Process
 *    }
 *
 * 2. Warp-level unrolling:
 *    // Each warp processes multiple elements
 *    for (int i = threadIdx.x; i < N; i += blockDim.x * 4) {
 *        result[i] = ...;
 *        result[i + blockDim.x] = ...;
 *        result[i + blockDim.x*2] = ...;
 *        result[i + blockDim.x*3] = ...;
 *    }
 *
 * 3. Loop invariant in kernels:
 *    __global__ void kernel(float* data, int n) {
 *        // Hoist out of loop
 *        float factor = sqrt(blockDim.x);
 *        for (int i = 0; i < n; i++) {
 *            data[i] *= factor;
 *        }
 *    }
 *
 * 4. Strength reduction in GPU:
 *    // BAD: Division in loop
 *    for (int i = 0; i < n; i++) {
 *        x = i / 16;
 *    }
 *
 *    // GOOD: Bit shift
 *    for (int i = 0; i < n; i++) {
 *        x = i >> 4;  // Division by power of 2
 *    }
 *
 * Real NVIDIA interview question:
 * "Why unroll loops in CUDA kernels?"
 * Answer: Reduces instruction count, enables ILP, hides memory latency,
 * better register allocation. Critical for compute-bound kernels.
 *
 * COMPILATION: g++ -std=c++11 -O2 02_loop_optimizations.cpp -o loop_opt
 * ==================================================================================================
 */

int main() {
    cout << "=== Loop Optimizations Practice ===" << endl;

    cout << "\n1. Loop Unrolling:" << endl;
    loopUnrollingDemo();

    cout << "\n2. Loop Invariant Code Motion:" << endl;
    loopInvariantDemo();

    cout << "\n3. Strength Reduction:" << endl;
    strengthReductionDemo();

    return 0;
}
