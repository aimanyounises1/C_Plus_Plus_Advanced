/*
 * ==================================================================================================
 * Exercise: Memory Alignment and Performance
 * ==================================================================================================
 * Difficulty: Advanced | Time: 40-50 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Understand memory alignment requirements
 * 2. Learn alignas and alignof
 * 3. Master false sharing prevention
 * 4. Practice aligned memory allocation
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - GPU memory coalescing requirements
 * - Aligned memory for efficient transfers
 * - False sharing in multi-threaded GPU code
 * - Structure padding for performance
 * ==================================================================================================
 */

#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <new>
using namespace std;
using namespace chrono;

/*
 * EXERCISE 1: Alignment Basics (10 min)
 */

struct Unaligned {
    char a;
    int b;
    char c;
    double d;
};

struct alignas(64) Aligned {
    char a;
    int b;
    char c;
    double d;
};

void alignmentBasics() {
    cout << "Unaligned struct size: " << sizeof(Unaligned) << " bytes" << endl;
    cout << "Aligned struct size: " << sizeof(Aligned) << " bytes" << endl;

    cout << "\nalignof:" << endl;
    cout << "char: " << alignof(char) << endl;
    cout << "int: " << alignof(int) << endl;
    cout << "double: " << alignof(double) << endl;
    cout << "Unaligned: " << alignof(Unaligned) << endl;
    cout << "Aligned: " << alignof(Aligned) << endl;
}

/*
 * EXERCISE 2: False Sharing (15 min)
 * Cache line: 64 bytes, shared between threads
 */

// BAD: False sharing - counters on same cache line
struct BadCounters {
    atomic<int> counter1;
    atomic<int> counter2;
};

// GOOD: Cache line separation
struct alignas(64) GoodCounters {
    atomic<int> counter1;
    char padding[60];  // Ensure different cache lines
};

void incrementBad(BadCounters& counters, int id, int iterations) {
    for (int i = 0; i < iterations; i++) {
        if (id == 0) counters.counter1++;
        else counters.counter2++;
    }
}

void incrementGood(GoodCounters& c1, GoodCounters& c2, int id, int iterations) {
    for (int i = 0; i < iterations; i++) {
        if (id == 0) c1.counter1++;
        else c2.counter1++;
    }
}

void falseSharingDemo() {
    const int ITERATIONS = 10000000;

    // Bad: False sharing
    BadCounters bad{};
    auto start = high_resolution_clock::now();
    thread t1(incrementBad, ref(bad), 0, ITERATIONS);
    thread t2(incrementBad, ref(bad), 1, ITERATIONS);
    t1.join();
    t2.join();
    auto end = high_resolution_clock::now();
    auto badTime = duration_cast<milliseconds>(end - start).count();

    // Good: No false sharing
    GoodCounters good1{}, good2{};
    start = high_resolution_clock::now();
    thread t3(incrementGood, ref(good1), ref(good2), 0, ITERATIONS);
    thread t4(incrementGood, ref(good1), ref(good2), 1, ITERATIONS);
    t3.join();
    t4.join();
    end = high_resolution_clock::now();
    auto goodTime = duration_cast<milliseconds>(end - start).count();

    cout << "With false sharing: " << badTime << " ms" << endl;
    cout << "Without false sharing: " << goodTime << " ms" << endl;
    cout << "Speedup: " << (double)badTime / goodTime << "x" << endl;
}

/*
 * EXERCISE 3: Aligned Memory Allocation (10 min)
 */

void alignedAllocationDemo() {
    const int SIZE = 1024;

    // Aligned allocation (C++17)
    void* ptr = aligned_alloc(64, SIZE);
    if (ptr) {
        cout << "Aligned pointer: " << ptr << endl;
        cout << "Address % 64: " << (size_t)ptr % 64 << endl;
        free(ptr);
    }

    // Or use new with alignment
    struct alignas(64) AlignedArray {
        int data[SIZE / sizeof(int)];
    };

    AlignedArray* arr = new AlignedArray();
    cout << "Aligned array address: " << arr << endl;
    cout << "Address % 64: " << (size_t)arr % 64 << endl;
    delete arr;
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is memory alignment?
 * A: Requirement that data address be multiple of data size
 *    int (4 bytes) should be at address divisible by 4
 *
 * Q2: Why is alignment important?
 * A: - CPU loads aligned data faster (single memory access)
 *    - Unaligned access may require multiple loads
 *    - Some architectures don't support unaligned access
 *
 * Q3: What is structure padding?
 * A: Compiler adds padding bytes to align members
 *    struct { char c; int i; } -> 8 bytes (4 padding after c)
 *
 * Q4: How to control alignment?
 * A: - alignas(N): Align to N bytes
 *    - alignof(T): Query alignment of type T
 *    - aligned_alloc: Allocate aligned memory
 *
 * Q5: What is false sharing?
 * A: Two threads access different variables on same cache line
 *    Each write invalidates other's cache
 *    Causes performance degradation
 *
 * Q6: How to prevent false sharing?
 * A: - Pad structures to cache line size (64 bytes)
 *    - alignas(64) for cache line alignment
 *    - Separate frequently updated variables
 *
 * Q7: Cache line size?
 * A: Typically 64 bytes on modern CPUs
 *    128 bytes on some GPUs
 *
 * Q8: Performance impact of misalignment?
 * A: - 2-5x slower for unaligned loads
 *    - May cause exceptions on some platforms
 *    - False sharing can cause 10x slowdown
 *
 * Q9: When to use alignas?
 * A: - SIMD operations (16/32/64 byte alignment)
 *    - Prevent false sharing (64 byte alignment)
 *    - Hardware requirements (GPU, DMA)
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 *
 * Memory alignment is CRITICAL for GPU performance:
 *
 * 1. Coalesced Memory Access:
 *    - Threads in warp must access aligned, consecutive addresses
 *    - Misaligned access causes multiple transactions
 *    - 128-byte alignment for optimal coalescing
 *
 * 2. GPU Structure Alignment:
 *    struct __align__(16) float4 { float x, y, z, w; };
 *    - Ensures efficient memory transactions
 *    - Vectorized loads/stores
 *
 * 3. Shared Memory Bank Conflicts:
 *    - Similar to false sharing
 *    - Threads accessing same bank cause serialization
 *    - Pad shared memory arrays to avoid conflicts
 *
 * 4. cudaMalloc Alignment:
 *    - cudaMalloc returns 256-byte aligned pointers
 *    - Ensures coalesced access from offset 0
 *
 * Example GPU code:
 * // BAD: Unaligned access
 * float* unaligned = d_ptr + 1;  // Not aligned!
 * kernel<<<grid, block>>>(unaligned);
 *
 * // GOOD: Aligned access
 * float* aligned = d_ptr;  // Aligned by cudaMalloc
 * kernel<<<grid, block>>>(aligned);
 *
 * // Avoid false sharing in shared memory:
 * __shared__ float data[32][33];  // +1 to avoid bank conflicts
 *
 * COMPILATION: g++ -std=c++17 -pthread -O2 02_memory_alignment.cpp -o alignment
 * ==================================================================================================
 */

int main() {
    cout << "=== Memory Alignment Practice ===" << endl;

    cout << "\n1. Alignment Basics:" << endl;
    alignmentBasics();

    cout << "\n2. False Sharing:" << endl;
    falseSharingDemo();

    cout << "\n3. Aligned Allocation:" << endl;
    alignedAllocationDemo();

    return 0;
}
