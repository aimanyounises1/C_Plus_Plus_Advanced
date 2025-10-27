/*
 * ==================================================================================================
 * Exercise: Memory Prefetching
 * ==================================================================================================
 * Difficulty: Advanced | Time: 40-50 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Understand hardware and software prefetching
 * 2. Learn __builtin_prefetch usage
 * 3. Master prefetching strategies
 * 4. Practice hiding memory latency
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Hiding global memory latency in CUDA
 * - Prefetching data into shared memory
 * - Double buffering techniques
 * - Async memory operations
 * ==================================================================================================
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
using namespace std;
using namespace chrono;

/*
 * EXERCISE 1: Software Prefetching Basics (15 min)
 */

int sumWithoutPrefetch(const vector<int>& data) {
    int sum = 0;
    for (size_t i = 0; i < data.size(); i++) {
        sum += data[i];
    }
    return sum;
}

int sumWithPrefetch(const vector<int>& data) {
    int sum = 0;
    const int PREFETCH_DISTANCE = 16;  // Prefetch 16 elements ahead

    for (size_t i = 0; i < data.size(); i++) {
        // Prefetch future elements
        if (i + PREFETCH_DISTANCE < data.size()) {
            __builtin_prefetch(&data[i + PREFETCH_DISTANCE], 0, 3);
            // Parameters: (address, rw, locality)
            // rw: 0=read, 1=write
            // locality: 0-3 (0=no temporal locality, 3=high)
        }
        sum += data[i];
    }
    return sum;
}

void prefetchBasicsDemo() {
    const int N = 100000000;
    vector<int> data(N);

    // Initialize with random data to prevent optimization
    for (int i = 0; i < N; i++) {
        data[i] = i % 100;
    }

    auto start = high_resolution_clock::now();
    int sum1 = sumWithoutPrefetch(data);
    auto end = high_resolution_clock::now();
    auto noPrefetchTime = duration_cast<milliseconds>(end - start).count();

    start = high_resolution_clock::now();
    int sum2 = sumWithPrefetch(data);
    end = high_resolution_clock::now();
    auto prefetchTime = duration_cast<milliseconds>(end - start).count();

    cout << "Without prefetch: " << noPrefetchTime << " ms (sum=" << sum1 << ")" << endl;
    cout << "With prefetch: " << prefetchTime << " ms (sum=" << sum2 << ")" << endl;
    cout << "Speedup: " << (double)noPrefetchTime / prefetchTime << "x" << endl;
}

/*
 * EXERCISE 2: Prefetching with Pointer Chasing (15 min)
 */

struct Node {
    int data;
    Node* next;
};

int traverseListNoPrefetch(Node* head) {
    int sum = 0;
    Node* current = head;

    while (current != nullptr) {
        sum += current->data;
        current = current->next;
    }

    return sum;
}

int traverseListWithPrefetch(Node* head) {
    int sum = 0;
    Node* current = head;

    while (current != nullptr) {
        // Prefetch next node before accessing current
        if (current->next != nullptr) {
            __builtin_prefetch(current->next, 0, 1);
        }

        sum += current->data;
        current = current->next;
    }

    return sum;
}

void pointerChasingDemo() {
    const int N = 1000000;

    // Create linked list
    Node* head = new Node{0, nullptr};
    Node* current = head;

    for (int i = 1; i < N; i++) {
        current->next = new Node{i, nullptr};
        current = current->next;
    }

    auto start = high_resolution_clock::now();
    int sum1 = traverseListNoPrefetch(head);
    auto end = high_resolution_clock::now();
    auto noPrefetchTime = duration_cast<milliseconds>(end - start).count();

    start = high_resolution_clock::now();
    int sum2 = traverseListWithPrefetch(head);
    end = high_resolution_clock::now();
    auto prefetchTime = duration_cast<milliseconds>(end - start).count();

    cout << "List traversal without prefetch: " << noPrefetchTime << " ms" << endl;
    cout << "List traversal with prefetch: " << prefetchTime << " ms" << endl;
    cout << "Speedup: " << (double)noPrefetchTime / prefetchTime << "x" << endl;

    // Cleanup
    current = head;
    while (current != nullptr) {
        Node* next = current->next;
        delete current;
        current = next;
    }
}

/*
 * EXERCISE 3: Matrix Multiplication with Prefetching (10 min)
 */

void matmulNoPrefetch(const vector<vector<float>>& A,
                      const vector<vector<float>>& B,
                      vector<vector<float>>& C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matmulWithPrefetch(const vector<vector<float>>& A,
                        const vector<vector<float>>& B,
                        vector<vector<float>>& C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Prefetch next row of B
            if (j + 1 < n) {
                for (int k = 0; k < min(n, 8); k++) {
                    __builtin_prefetch(&B[k][j + 1], 0, 2);
                }
            }

            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrixPrefetchDemo() {
    const int N = 256;

    vector<vector<float>> A(N, vector<float>(N, 1.0f));
    vector<vector<float>> B(N, vector<float>(N, 1.0f));
    vector<vector<float>> C1(N, vector<float>(N, 0.0f));
    vector<vector<float>> C2(N, vector<float>(N, 0.0f));

    auto start = high_resolution_clock::now();
    matmulNoPrefetch(A, B, C1, N);
    auto end = high_resolution_clock::now();
    auto noPrefetchTime = duration_cast<milliseconds>(end - start).count();

    start = high_resolution_clock::now();
    matmulWithPrefetch(A, B, C2, N);
    end = high_resolution_clock::now();
    auto prefetchTime = duration_cast<milliseconds>(end - start).count();

    cout << "Matrix multiply without prefetch: " << noPrefetchTime << " ms" << endl;
    cout << "Matrix multiply with prefetch: " << prefetchTime << " ms" << endl;
    cout << "Speedup: " << (double)noPrefetchTime / prefetchTime << "x" << endl;
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is prefetching?
 * A: Loading data into cache before it's needed
 *    Hides memory latency by overlapping memory access with computation
 *
 * Q2: Hardware vs software prefetching?
 * A: Hardware: CPU automatically detects patterns and prefetches
 *    Software: Programmer explicitly requests prefetch
 *
 * Q3: __builtin_prefetch parameters?
 * A: __builtin_prefetch(addr, rw, locality)
 *    addr: Address to prefetch
 *    rw: 0 for read, 1 for write
 *    locality: 0-3 (0=no reuse, 3=high reuse)
 *
 * Q4: When does prefetching help?
 * A: - Predictable access patterns
 *    - High memory latency
 *    - Sufficient computation to overlap
 *    - Enough cache space for prefetched data
 *
 * Q5: When does prefetching hurt?
 * A: - Unpredictable/random access
 *    - Already cache-friendly code
 *    - Prefetch distance too large (evicts useful data)
 *    - Excessive prefetch (bandwidth waste)
 *
 * Q6: How far ahead to prefetch?
 * A: Depends on:
 *    - Memory latency (~200 cycles)
 *    - Computation per iteration
 *    - Cache size
 *    Typically 8-64 elements ahead
 *
 * Q7: Prefetching for linked lists?
 * A: Prefetch next->next node
 *    Gives time for memory fetch during current processing
 *
 * Q8: Does compiler auto-prefetch?
 * A: Modern compilers do for regular patterns
 *    Manual prefetch still helps for:
 *    - Irregular patterns
 *    - Pointer chasing
 *    - Complex data structures
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 *
 * Prefetching concepts transfer directly to GPU optimization:
 *
 * 1. Shared Memory Prefetching (Explicit):
 *    // CPU: Software prefetch
 *    __builtin_prefetch(&data[i+16], 0, 3);
 *
 *    // GPU: Load into shared memory (manual cache)
 *    __shared__ float sharedData[BLOCK_SIZE];
 *    sharedData[threadIdx.x] = globalData[i];  // Prefetch
 *    __syncthreads();
 *    // Use sharedData (fast) instead of globalData (slow)
 *
 * 2. Double Buffering:
 *    // Load next tile while computing current tile
 *    __shared__ float tileA[2][TILE_SIZE];
 *    int phase = 0;
 *
 *    // Prefetch first tile
 *    load_tile(tileA[0], ...);
 *
 *    for (tile) {
 *        __syncthreads();
 *        // Compute on tileA[phase]
 *        compute(tileA[phase]);
 *
 *        // Simultaneously load next tile into tileA[1-phase]
 *        if (has_next_tile)
 *            load_tile(tileA[1-phase], ...);
 *
 *        phase = 1 - phase;
 *    }
 *
 * 3. Texture Memory (Hardware Prefetch):
 *    // GPU texture cache has hardware prefetching
 *    // Good for 2D spatial locality
 *    texture<float> tex;
 *    float value = tex2D(tex, x, y);
 *
 * 4. Async Memory Copy (CUDA 11+):
 *    // Async copy hides latency
 *    __pipeline_memcpy_async(&shared[i], &global[i], size);
 *    __pipeline_commit();
 *    // Do other work
 *    __pipeline_wait_prior(0);
 *
 * Real NVIDIA interview question:
 * "How do you hide global memory latency in CUDA?"
 * Answer:
 * 1. Prefetch into shared memory (manual caching)
 * 2. Double buffering (overlap compute and load)
 * 3. Increase occupancy (more warps hide latency)
 * 4. Async memory operations (CUDA 11+)
 *
 * COMPILATION: g++ -std=c++11 -O2 04_prefetching.cpp -o prefetch
 * ==================================================================================================
 */

int main() {
    cout << "=== Memory Prefetching Practice ===" << endl;

    cout << "\n1. Basic Prefetching:" << endl;
    prefetchBasicsDemo();

    cout << "\n2. Prefetching with Pointer Chasing:" << endl;
    pointerChasingDemo();

    cout << "\n3. Matrix Multiplication with Prefetching:" << endl;
    matrixPrefetchDemo();

    return 0;
}
