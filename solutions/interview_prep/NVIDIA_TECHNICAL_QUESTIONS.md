# Nvidia Software Engineer Interview Questions

This document contains real and common technical interview questions asked at Nvidia for Software Engineer positions, organized by topic with detailed answers and explanations.

---

## Table of Contents
1. [CUDA & GPU Programming](#cuda--gpu-programming)
2. [GPU Architecture](#gpu-architecture)
3. [Memory & Performance Optimization](#memory--performance-optimization)
4. [Algorithms & Data Structures](#algorithms--data-structures)
5. [System Design](#system-design)
6. [C++ Fundamentals](#c-fundamentals)
7. [Behavioral Questions](#behavioral-questions)

---

## CUDA & GPU Programming

### Q1: Explain the CUDA memory hierarchy. When would you use each type?

**Answer:**

CUDA has several memory types, each with different characteristics:

1. **Global Memory**
   - Size: GBs (largest)
   - Speed: ~200-900 GB/s (slowest)
   - Scope: All threads across all blocks
   - Lifetime: Persistent across kernel launches
   - Use when: Large datasets, initial data loading
   - Example: Input/output arrays

2. **Shared Memory**
   - Size: 48-164 KB per SM
   - Speed: ~10 TB/s (very fast)
   - Scope: All threads within a block
   - Lifetime: Duration of block execution
   - Use when: Data reuse within block, inter-thread communication
   - Example: Tiling in matrix multiplication

3. **Registers**
   - Size: 64K 32-bit registers per SM
   - Speed: ~50 TB/s (fastest)
   - Scope: Private to each thread
   - Lifetime: Duration of thread execution
   - Use when: Local variables, loop counters
   - Example: Accumulator variables

4. **Constant Memory**
   - Size: 64 KB
   - Speed: Fast if cached, slow otherwise
   - Scope: Read-only for all threads
   - Lifetime: Persistent
   - Use when: Small read-only data accessed uniformly
   - Example: Kernel parameters, lookup tables

5. **Texture Memory**
   - Size: Same as global (cached)
   - Speed: Fast for 2D/3D spatial locality
   - Scope: Read-only
   - Use when: Image processing, spatial data access
   - Example: Image filtering

6. **Local Memory**
   - Actually in global memory (misnomer!)
   - Used for register spills
   - Avoid by reducing register usage

**Key Decision Tree:**
```
Need to share between threads in block? → Shared Memory
Small constant data (< 64KB)? → Constant Memory
2D/3D spatial access pattern? → Texture Memory
Thread-private? → Registers
Everything else → Global Memory
```

---

### Q2: What is memory coalescing and why is it important?

**Answer:**

**Memory Coalescing** is when consecutive threads in a warp access consecutive memory addresses, allowing the memory controller to combine multiple accesses into fewer transactions.

**Why It Matters:**
- Un-coalesced accesses can reduce bandwidth by 8-32x
- GPUs fetch memory in 32, 64, or 128-byte segments
- Wasted bandwidth = wasted performance

**Example of Coalesced Access:**
```cpp
__global__ void coalesced(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx];  // ✓ GOOD: Sequential access
}
```

**Example of Uncoalesced Access:**
```cpp
__global__ void uncoalesced(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx * 32];  // ✗ BAD: Strided access
}
```

**How to Achieve Coalescing:**
1. Access memory sequentially (thread N accesses address N)
2. Align allocations to 128-byte boundaries
3. Use struct-of-arrays instead of array-of-structs when possible
4. Avoid strided access patterns

**Impact:**
- Good coalescing: 600-800 GB/s effective bandwidth
- Poor coalescing: 50-100 GB/s effective bandwidth
- **12-16x performance difference!**

---

### Q3: Explain warp divergence and how to minimize it.

**Answer:**

**Warp Divergence** occurs when threads within a warp (32 threads) take different execution paths due to conditional branches.

**Why It's Bad:**
- Warp executes in SIMT (Single Instruction, Multiple Thread) model
- All threads in warp must execute same instruction
- Divergent branches are serialized
- Effective parallelism reduced

**Example of Divergence:**
```cpp
__global__ void divergent(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx % 2 == 0) {  // ✗ BAD: Half threads diverge
        data[idx] = data[idx] * 2;
    } else {
        data[idx] = data[idx] + 1;
    }
    // Warp must execute both branches serially
}
```

**How to Minimize:**

1. **Predication** (Convert branches to arithmetic):
```cpp
int mask = (idx % 2 == 0) ? 1 : 0;
data[idx] = mask * (data[idx] * 2) + (1 - mask) * (data[idx] + 1);
```

2. **Reorganize Data** (Group similar work together):
```cpp
// Process all even indices first, then odd
```

3. **Warp-Level Functions** (Use warp voting functions):
```cpp
unsigned mask = __ballot_sync(0xffffffff, condition);
if (__popc(mask) == 32) {
    // All threads true - no divergence
}
```

4. **Block-Level Partitioning:**
```cpp
// Have entire blocks process similar data
```

**Performance Impact:**
- No divergence: 100% warp efficiency
- 50/50 divergence: 50% efficiency (2x slower)
- Worst case: 1/32 active: 3% efficiency (32x slower!)

---

### Q4: Implement a parallel reduction in CUDA. Optimize it.

**Answer:**

See `solutions/phase5_cuda/03_parallel_reduction.cu` for complete implementation.

**Key Optimization Techniques:**

1. **Sequential Addressing** (avoid divergence)
2. **First Add During Load** (reduce iterations)
3. **Warp Unrolling** (eliminate last sync)
4. **Complete Unrolling** (compile-time optimization)
5. **Warp Shuffle** (register-based, fastest)

**Progression:**
```
Naive:           100.0 μs  (baseline)
Sequential:       45.0 μs  (2.2x)
First Add:        30.0 μs  (3.3x)
Warp Unroll:      25.0 μs  (4.0x)
Warp Shuffle:     15.0 μs  (6.7x)
```

**Final optimized kernel:**
```cpp
__inline__ __device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce(float* input, float* output, int n) {
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // First add during load
    float sum = 0;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];

    // Warp-level reduction
    sum = warpReduce(sum);

    // Inter-warp reduction
    __shared__ float warpSums[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) warpSums[wid] = sum;
    __syncthreads();

    if (threadIdx.x < 32) {
        sum = warpSums[threadIdx.x];
        sum = warpReduce(sum);
    }

    if (threadIdx.x == 0) output[blockIdx.x] = sum;
}
```

---

## GPU Architecture

### Q5: Explain the GPU architecture at a high level. How does it differ from CPU?

**Answer:**

**GPU Architecture (Nvidia):**

```
GPU
├── Graphics Processing Clusters (GPCs)
│   ├── Texture Processing Clusters (TPCs)
│   │   ├── Streaming Multiprocessors (SMs)  ← Key unit!
│   │   │   ├── CUDA Cores (FP32 units)
│   │   │   ├── Tensor Cores (matrix ops)
│   │   │   ├── Shared Memory / L1 Cache
│   │   │   ├── Register File
│   │   │   ├── Warp Schedulers
│   │   │   └── Special Function Units (SFUs)
│   │   └── ...
├── L2 Cache
├── Global Memory (HBM/GDDR)
└── Memory Controllers
```

**Example: A100 GPU**
- 108 SMs
- 64 FP32 cores per SM = 6,912 CUDA cores
- 4 Tensor Cores per SM
- 164 KB shared memory per SM
- 40 GB HBM2 memory
- 1.5 TB/s memory bandwidth

**GPU vs CPU:**

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Design Philosophy** | Minimize latency (single thread) | Maximize throughput (many threads) |
| **Cores** | 4-64 powerful cores | Thousands of simple cores |
| **Cache** | Large (MB per core) | Small (KB per core) |
| **Clock Speed** | 3-5 GHz | 1-2 GHz |
| **Control Logic** | Complex (branch prediction, OoO) | Simple (in-order) |
| **Best For** | Serial, complex logic | Parallel, regular computation |
| **Latency Hiding** | Cache hierarchy | Massive threading |
| **Programming Model** | Sequential | Parallel (SIMT) |

**Key Insight:**
- CPU: "Make one task run very fast"
- GPU: "Run millions of tasks in parallel"

**When to Use GPU:**
- ✓ Data parallelism (same operation, different data)
- ✓ High arithmetic intensity
- ✓ Regular memory patterns
- ✗ Complex control flow
- ✗ Small problems
- ✗ Irregular memory access

---

### Q6: What is occupancy and why does it matter?

**Answer:**

**Occupancy** is the ratio of active warps to maximum possible warps on an SM.

**Formula:**
```
Occupancy = Active Warps per SM / Maximum Warps per SM
```

**Why It Matters:**

1. **Latency Hiding:**
   - Memory access: ~200-800 cycles
   - Math operation: ~10 cycles
   - Need many warps to switch between while waiting

2. **Resource Utilization:**
   - More active warps = better utilization
   - Can hide memory and instruction latency

3. **Sweet Spot:**
   - 100% occupancy NOT always best!
   - 50-75% often optimal
   - More registers per thread can improve ILP

**Factors Limiting Occupancy:**

1. **Registers per thread:**
   - A100: 65,536 registers per SM
   - 128 registers/thread × 512 threads = 65,536 ✓
   - 256 registers/thread × 256 threads = 65,536 ✓
   - More registers → fewer threads

2. **Shared memory per block:**
   - A100: 164 KB per SM
   - 48 KB per block → max 3 blocks per SM
   - Must balance shared memory usage

3. **Threads per block:**
   - Max 1024 threads per block
   - Must be multiple of 32 (warp size)

4. **Blocks per SM:**
   - Hardware limit (typically 16-32)

**Example:**
```cpp
// Low occupancy (intentionally)
__global__ void low_occupancy() {
    __shared__ float data[48 * 1024 / sizeof(float)]; // 48 KB
    float regs[100];  // Lots of registers
    // Only 1 block per SM fits!
}

// High occupancy
__global__ void high_occupancy() {
    __shared__ float data[1024];  // 4 KB
    float acc = 0.0f;  // Few registers
    // Many blocks per SM can fit
}
```

**How to Optimize:**

1. **Use CUDA Occupancy Calculator:**
```bash
nvcc --ptxas-options=-v kernel.cu
# Shows register and shared memory usage
```

2. **Profile with Nsight Compute:**
```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active
```

3. **Trade-offs:**
   - More shared memory → Better reuse → Lower occupancy
   - More registers → Better ILP → Lower occupancy
   - Find the sweet spot for your kernel!

**Common Misconception:**
"Higher occupancy is always better" ❌

**Reality:**
- Compute-bound kernel: Occupancy less critical
- Memory-bound kernel: Need high occupancy to hide latency
- Balance between occupancy and per-thread resources

---

## Memory & Performance Optimization

### Q7: You have a kernel that's running slow. How do you optimize it?

**Answer:**

**Systematic Optimization Process:**

**Step 1: Profile First!**
```bash
# High-level overview
nsys profile --stats=true ./program

# Detailed kernel metrics
ncu --set full ./program

# Specific metrics
ncu --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum ./program
```

**Step 2: Identify Bottleneck**

Check these metrics:

1. **Memory Bound?**
   - `dram__throughput.avg.pct_of_peak_sustained_elapsed` < 60%
   - Optimize memory access patterns

2. **Compute Bound?**
   - `sm__throughput.avg.pct_of_peak_sustained_elapsed` < 60%
   - Increase arithmetic intensity

3. **Latency Bound?**
   - Low occupancy + memory/compute underutilized
   - Increase parallelism

4. **Launch Bound?**
   - Kernel launch overhead significant
   - Reduce launches, increase work per kernel

**Step 3: Apply Optimizations**

**If Memory Bound:**
```cpp
// Before: Uncoalesced access
float val = data[threadIdx.x * stride];  // ✗

// After: Coalesced access
float val = data[threadIdx.x];  // ✓

// Before: Repeated global memory reads
for (int i = 0; i < N; i++) {
    sum += globalArray[sharedIndex];  // ✗
}

// After: Cache in register
float cached = globalArray[sharedIndex];  // ✓
for (int i = 0; i < N; i++) {
    sum += cached;
}

// Use shared memory for block-level reuse
__shared__ float tile[TILE_SIZE];
tile[threadIdx.x] = globalArray[...];
__syncthreads();
// Now use tile[] instead of globalArray[]
```

**If Compute Bound:**
```cpp
// Increase arithmetic intensity
// Do more work per memory access

// Use faster math functions
__fdividef(a, b)  // Fast reciprocal
__expf(x)         // Fast exp
__sinf(x)         // Fast sin

// Enable fused multiply-add
result = __fmaf_rn(a, b, c);  // a*b + c in one instruction
```

**If Latency Bound:**
```cpp
// Increase occupancy
// - Reduce register usage: __launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)
// - Reduce shared memory usage
// - Increase threads per block

__launch_bounds__(256, 4)  // 256 threads, min 4 blocks per SM
__global__ void kernel() {
    // ...
}
```

**Step 4: Measure Impact**
- Always measure before and after
- One optimization at a time
- Keep notes on what worked

**Optimization Priority:**
1. Algorithm (O(n²) → O(n log n))
2. Memory access patterns
3. Occupancy
4. Instruction optimization
5. Tuning parameters

---

### Q8: Explain bank conflicts in shared memory. How do you avoid them?

**Answer:**

**Bank Conflicts** occur when multiple threads in a warp access the same shared memory bank simultaneously.

**Shared Memory Banking:**
- Shared memory divided into 32 banks (on most GPUs)
- Successive 4-byte words map to successive banks
- Bank 0: addresses 0, 128, 256, ...
- Bank 1: addresses 4, 132, 260, ...
- Bank 31: addresses 124, 252, 380, ...

**Conflict Types:**

1. **No Conflict** (all threads access different banks):
```cpp
__shared__ float data[32];
float val = data[threadIdx.x];  // ✓ Each thread different bank
```

2. **Broadcast** (all threads read same address):
```cpp
__shared__ float data[32];
float val = data[0];  // ✓ Broadcast - no conflict for reads
```

3. **Bank Conflict** (multiple threads → same bank, different addresses):
```cpp
__shared__ float data[32][32];
// Thread i accesses data[i][0]
float val = data[threadIdx.x][0];  // ✗ All access bank 0!
```

**How to Avoid:**

**Technique 1: Padding**
```cpp
// Before: 32-way bank conflicts
__shared__ float data[32][32];
float val = data[threadIdx.x][0];  // ✗

// After: No conflicts
__shared__ float data[32][33];  // +1 padding
float val = data[threadIdx.x][0];  // ✓
```

**Technique 2: Transpose Access Pattern**
```cpp
// Instead of:
data[row][col] where col is constant

// Use:
data[col][row] where row varies by thread
```

**Technique 3: XOR Addressing**
```cpp
int idx = threadIdx.x ^ (threadIdx.x / 32);
float val = data[idx];
```

**Performance Impact:**
- No conflicts: 1 transaction
- N-way conflict: N serialized transactions
- 32-way conflict: 32x slower! (**3200% slower!**)

**How to Detect:**
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./program

# Should be 0 or very low for optimized kernels
```

**Real-World Example (Matrix Transpose):**
```cpp
// Naive - has bank conflicts
__global__ void transposeNaive(float* out, float* in, int n) {
    __shared__ float tile[32][32];

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();

    int x2 = blockIdx.y * 32 + threadIdx.x;
    int y2 = blockIdx.x * 32 + threadIdx.y;

    out[y2 * n + x2] = tile[threadIdx.x][threadIdx.y];  // ✗ Bank conflicts!
}

// Optimized - padding eliminates conflicts
__global__ void transposeOptimized(float* out, float* in, int n) {
    __shared__ float tile[32][33];  // ✓ Padding!

    // ... same code but no conflicts
}

// Speedup: 2-3x!
```

---

## Algorithms & Data Structures

### Q9: How would you implement parallel merge sort on GPU?

**Answer:**

**Parallel Merge Sort Strategy:**

**Phase 1: Local Sort (Within Blocks)**
```cpp
__global__ void localSort(int* data, int n) {
    __shared__ int shared[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    shared[threadIdx.x] = (idx < n) ? data[idx] : INT_MAX;
    __syncthreads();

    // Bitonic sort within block (fully parallel)
    bitonicSort(shared, BLOCK_SIZE);

    if (idx < n) data[idx] = shared[threadIdx.x];
}
```

**Phase 2: Merge Sorted Chunks (Recursive)**
```cpp
// Merge pairs of sorted chunks
// Size: 1 → 2 → 4 → 8 → ... → N
for (int size = BLOCK_SIZE; size < n; size *= 2) {
    mergeKernel<<<...>>>(data, size);
}
```

**Phase 3: Parallel Merge**
```cpp
__global__ void parallelMerge(int* data, int* temp,
                               int chunkSize, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread finds its position in merged output
    int chunk1_start = (tid / chunkSize) * chunkSize * 2;
    int chunk2_start = chunk1_start + chunkSize;

    // Binary search to find merge position
    int pos = findMergePos(data, chunk1_start, chunk2_start,
                           chunkSize, tid);

    temp[pos] = data[tid];
}
```

**Complete Algorithm:**

1. Sort within blocks (bitonic or odd-even merge)
2. Merge pairs iteratively: 1→2, 2→4, 4→8, ...
3. Use parallel merge at each step

**Complexity:**
- Work: O(n log² n) (bitonic) or O(n log n) (merge)
- Span: O(log² n)
- Parallelism: O(n / log n)

**Alternative: Thrust Library**
```cpp
#include <thrust/sort.h>

thrust::device_vector<int> data(n);
thrust::sort(data.begin(), data.end());
// Optimized implementation!
```

**Interview Discussion Points:**
- Compare with radix sort (O(kn) for k-bit integers)
- Memory overhead (need temp array)
- When to use GPU vs CPU sorting
- Stability requirements

---

### Q10: Implement a thread-safe hash table on GPU.

**Answer:**

**Challenges:**
1. Concurrent insertions - race conditions
2. Memory allocation on device
3. Hash collisions
4. Performance (minimize atomics)

**Approach 1: Open Addressing with Atomics**

```cpp
struct HashTable {
    int* keys;
    int* values;
    int capacity;
    int empty_key = -1;
};

__device__ void insert(HashTable* table, int key, int value) {
    int hash = key % table->capacity;

    while (true) {
        int old = atomicCAS(&table->keys[hash],
                           table->empty_key,
                           key);

        if (old == table->empty_key || old == key) {
            // Successfully inserted or key already exists
            table->values[hash] = value;
            return;
        }

        // Collision - linear probing
        hash = (hash + 1) % table->capacity;
    }
}

__device__ int lookup(HashTable* table, int key) {
    int hash = key % table->capacity;

    while (true) {
        int stored_key = table->keys[hash];

        if (stored_key == key) {
            return table->values[hash];
        }

        if (stored_key == table->empty_key) {
            return -1;  // Not found
        }

        hash = (hash + 1) % table->capacity;
    }
}
```

**Approach 2: Chaining (Lock-Free)**

```cpp
struct Node {
    int key;
    int value;
    Node* next;
};

struct HashTable {
    Node** buckets;
    int num_buckets;
};

__device__ void insert(HashTable* table, int key, int value) {
    int bucket = key % table->num_buckets;

    Node* new_node = /* allocate */;
    new_node->key = key;
    new_node->value = value;

    // Atomic pointer swap
    Node* old_head;
    do {
        old_head = table->buckets[bucket];
        new_node->next = old_head;
    } while (atomicCAS(&table->buckets[bucket],
                       old_head,
                       new_node) != old_head);
}
```

**Optimization: Warp-Cooperative Insert**

```cpp
__device__ void warpInsert(HashTable* table, int key, int value) {
    int lane = threadIdx.x % 32;
    int hash = (key + lane) % table->capacity;

    // Try 32 positions in parallel
    int old = atomicCAS(&table->keys[hash], EMPTY, key);

    // Check if anyone succeeded
    int success_mask = __ballot_sync(0xffffffff,
                                     old == EMPTY || old == key);

    if (success_mask != 0) {
        int winner = __ffs(success_mask) - 1;
        if (lane == winner) {
            table->values[hash] = value;
        }
        return;
    }

    // Retry with new offset
    // ...
}
```

**Discussion Points:**
- Trade-offs: Open addressing vs chaining
- Load factor considerations
- Perfect hashing for static data
- Concurrent resizing (very difficult!)
- Using shared memory for block-local hash tables

---

## System Design

### Q11: Design a GPU-accelerated inference system for a deep learning model.

**Answer:**

**Requirements Gathering:**
- Model size and architecture?
- Latency requirements? (real-time vs batch)
- Throughput requirements? (requests/second)
- Single vs multi-GPU?

**High-Level Architecture:**

```
┌─────────────────────────────────────────────────┐
│              Load Balancer / API Gateway         │
└─────────────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌────▼──────┐ ┌────▼──────┐
│ Inference    │ │ Inference  │ │ Inference │
│ Server 1     │ │ Server 2   │ │ Server 3  │
│ ┌──────────┐ │ │            │ │           │
│ │  Model   │ │ │            │ │           │
│ │  Cache   │ │ │            │ │           │
│ └──────────┘ │ │            │ │           │
│              │ │            │ │           │
│ ┌──────────┐ │ │            │ │           │
│ │   GPU    │ │ │            │ │           │
│ │  (A100)  │ │ │            │ │           │
│ └──────────┘ │ │            │ │           │
└──────────────┘ └────────────┘ └───────────┘
```

**Component Design:**

**1. Request Batching:**
```cpp
class RequestBatcher {
    std::vector<Request> batch;
    std::mutex mutex;

public:
    void addRequest(Request req) {
        std::lock_guard<std::mutex> lock(mutex);
        batch.push_back(req);

        if (batch.size() >= MAX_BATCH || timeoutReached()) {
            processBatch();
        }
    }

private:
    void processBatch() {
        // Combine inputs into single GPU transfer
        // Process entire batch in one kernel launch
        // Distribute results back to requests
    }
};
```

**2. Memory Management:**
```cpp
class GPUMemoryPool {
    std::vector<void*> free_buffers;
    size_t buffer_size;

public:
    void* allocate() {
        if (!free_buffers.empty()) {
            void* buf = free_buffers.back();
            free_buffers.pop_back();
            return buf;
        }

        void* buf;
        cudaMalloc(&buf, buffer_size);
        return buf;
    }

    void deallocate(void* buf) {
        free_buffers.push_back(buf);
    }
};
```

**3. Pipeline Stages:**
```
Input → Preprocessing → GPU Transfer → Inference → Transfer Back → Postprocessing
```

Use CUDA streams for overlap:
```cpp
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaMemcpyAsync(..., streams[i]);  // H2D transfer
    inferenceKernel<<<..., streams[i]>>>(...);  // Compute
    cudaMemcpyAsync(..., streams[i]);  // D2H transfer
}
```

**4. Optimization Strategies:**

- **Model Optimization:**
  - Quantization (FP16/INT8)
  - Layer fusion
  - TensorRT optimization
  - Pruning for sparsity

- **Batching:**
  - Dynamic batching (collect requests for T ms)
  - Max batch size based on GPU memory
  - Typical: 32-256 depending on model

- **Caching:**
  - Model weights in GPU memory (persistent)
  - Frequent intermediate results
  - KV cache for transformers

- **Multi-GPU:**
  - Model parallelism (large models)
  - Data parallelism (high throughput)
  - NCCL for inter-GPU communication

**5. Monitoring & Metrics:**

```cpp
class PerformanceMonitor {
    void recordMetrics() {
        // Latency (p50, p95, p99)
        // Throughput (requests/sec)
        // GPU utilization (%)
        // Memory usage
        // Queue depth
    }
};
```

**Trade-offs Discussion:**
- Latency vs throughput (batching tradeoff)
- Memory vs speed (larger batch = more memory)
- Multi-GPU complexity vs scaling benefits
- Cost optimization (right-sizing GPU instances)

---

## C++ Fundamentals

### Q12: Explain move semantics and rvalue references. When should you use them?

**Answer:**

**Rvalue References (`T&&`):**
- Reference to temporary objects
- Can be "moved from" (resources stolen)
- Distinguished from lvalue references (`T&`)

**Example:**
```cpp
class Matrix {
    float* data;
    int size;

public:
    // Copy constructor (expensive)
    Matrix(const Matrix& other) {
        size = other.size;
        data = new float[size];
        std::copy(other.data, other.data + size, data);  // Deep copy
    }

    // Move constructor (cheap)
    Matrix(Matrix&& other) noexcept {
        data = other.data;     // Steal pointer
        size = other.size;
        other.data = nullptr;  // Leave in valid state
        other.size = 0;
    }

    // Copy assignment
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            delete[] data;
            size = other.size;
            data = new float[size];
            std::copy(other.data, other.data + size, data);
        }
        return *this;
    }

    // Move assignment
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }

    ~Matrix() {
        delete[] data;
    }
};

// Usage:
Matrix createMatrix() {
    Matrix m(1000, 1000);
    return m;  // Move, not copy!
}

Matrix m1 = createMatrix();  // Move constructor called
Matrix m2 = std::move(m1);   // Explicit move
```

**When to Use:**

1. **Return large objects from functions** (automatic)
2. **Transfer ownership** (std::unique_ptr)
3. **Optimize containers** (vector reallocation)
4. **Perfect forwarding** (templates)

**Perfect Forwarding:**
```cpp
template<typename T>
void wrapper(T&& arg) {
    // Forward arg preserving lvalue/rvalue-ness
    actualFunction(std::forward<T>(arg));
}
```

**Common Pitfall:**
```cpp
void process(Matrix&& m) {
    // m is an lvalue inside this function!
    useMatrix(m);              // Passes as lvalue
    useMatrix(std::move(m));   // Passes as rvalue
}
```

**Interview Discussion:**
- Rule of Five (destructor, copy/move constructor, copy/move assignment)
- `noexcept` importance for move operations
- Return Value Optimization (RVO) vs move
- std::move doesn't move - it casts!

---

### Q13: What are common memory issues in C++ and how do you prevent them?

**Answer:**

**1. Memory Leaks**

```cpp
// Problem:
void leak() {
    int* data = new int[1000];
    // Forgot to delete!
}

// Solution 1: Smart pointers
void noLeak() {
    std::unique_ptr<int[]> data(new int[1000]);
    // Auto-deleted
}

// Solution 2: RAII
class Buffer {
    int* data;
public:
    Buffer(int n) : data(new int[n]) {}
    ~Buffer() { delete[] data; }
};
```

**2. Use-After-Free**

```cpp
// Problem:
int* ptr = new int(42);
delete ptr;
int x = *ptr;  // ✗ Undefined behavior!

// Solution:
int* ptr = new int(42);
delete ptr;
ptr = nullptr;  // ✓
if (ptr) {
    int x = *ptr;
}
```

**3. Double Free**

```cpp
// Problem:
int* ptr = new int(42);
delete ptr;
delete ptr;  // ✗ Crash!

// Solution: Smart pointers handle this automatically
std::unique_ptr<int> ptr(new int(42));
// Can't double-delete
```

**4. Dangling Pointers**

```cpp
// Problem:
int* getDanglingPointer() {
    int local = 42;
    return &local;  // ✗ Returns address of stack variable
}

// Solution: Return by value or allocate on heap
int getValue() {
    return 42;  // ✓
}

std::unique_ptr<int> getHeapValue() {
    return std::make_unique<int>(42);  // ✓
}
```

**5. Buffer Overflows**

```cpp
// Problem:
int arr[10];
arr[15] = 42;  // ✗ Out of bounds!

// Solution: Use bounds-checked containers
std::vector<int> vec(10);
vec.at(15) = 42;  // Throws exception

// Or use span (C++20)
std::span<int> s(arr, 10);
```

**6. Race Conditions (Multi-threaded)**

```cpp
// Problem:
int counter = 0;

void increment() {
    counter++;  // ✗ Not atomic!
}

// Solution 1: Mutex
std::mutex mtx;
void increment() {
    std::lock_guard<std::mutex> lock(mtx);
    counter++;
}

// Solution 2: Atomic
std::atomic<int> counter{0};
void increment() {
    counter.fetch_add(1);  // ✓
}
```

**Prevention Tools:**

1. **Sanitizers:**
```bash
g++ -fsanitize=address -g program.cpp      # AddressSanitizer
g++ -fsanitize=thread -g program.cpp       # ThreadSanitizer
g++ -fsanitize=undefined -g program.cpp    # UBSanitizer
```

2. **Valgrind:**
```bash
valgrind --leak-check=full ./program
```

3. **Static Analysis:**
```bash
clang-tidy program.cpp
cppcheck program.cpp
```

**Best Practices:**
- Use smart pointers by default
- RAII for all resources
- const-correctness
- Prefer std::vector over raw arrays
- Enable compiler warnings (-Wall -Wextra -Werror)
- Regular code reviews
- Automated testing with sanitizers

---

## Behavioral Questions

### Q14: Tell me about a time you optimized code for performance.

**STAR Framework Answer:**

**Situation:**
"In my previous project, we had a real-time image processing pipeline that was running at only 15 FPS on a 4K video stream, but we needed 30 FPS for the application requirements."

**Task:**
"I was tasked with identifying the bottleneck and improving performance to meet the 30 FPS target without compromising image quality."

**Action:**
"I took a systematic approach:

1. **Profiling:** Used Nsight Systems to identify that 70% of time was spent in the Gaussian blur kernel

2. **Analysis:** The kernel was memory-bound with poor cache utilization

3. **Optimization:**
   - Implemented separable convolution (2D → two 1D passes)
   - Added shared memory tiling to reuse data
   - Optimized memory access patterns for coalescing

4. **Validation:** Used Nsight Compute to verify:
   - Memory bandwidth utilization: 45% → 82%
   - Bank conflicts: 125K → 0
   - Occupancy: 33% → 67%

5. **Results:**
   - Gaussian blur: 12ms → 3ms (4x faster)
   - Overall pipeline: 66ms → 28ms (33 → 35 FPS)
   - Exceeded target while maintaining quality"

**Result:**
"The optimization not only met the 30 FPS requirement but exceeded it, providing headroom for future features. I documented the techniques and presented them to the team, which led to applying similar optimizations to other kernels in the pipeline."

**Key Learnings:**
- Always profile before optimizing
- Measure impact quantitatively
- Share knowledge with team

---

### Q15: How do you handle disagreements with team members?

**Answer:**

"I believe technical disagreements are healthy and lead to better solutions when handled constructively. Here's my approach:

**1. Listen First:**
- Understand their perspective fully
- Ask clarifying questions
- Assume good intent

**2. Data-Driven Discussion:**
- Focus on objective metrics (performance, maintainability)
- Prototype both approaches if feasible
- Benchmark and compare

**3. Find Common Ground:**
- Identify shared goals
- Explore hybrid solutions
- Be willing to compromise

**Example:**
'In a recent project, a colleague wanted to use a complex lock-free data structure, while I preferred a simpler mutex-based approach. Instead of arguing:

- We defined success criteria (throughput, latency, complexity)
- Each implemented a proof-of-concept
- Benchmarked both under realistic workloads
- Results showed lock-free was 30% faster but 3x more complex
- We chose simple approach for MVP, noted optimization for future
- Both learned from the exercise'

**When to Escalate:**
- If deadlock after good-faith discussion
- Impact on project timeline
- Need architectural decision from senior eng/architect

**Key Principle:**
Strong opinions, weakly held. Be confident but flexible with data."

---

## Additional Resources for Interview Prep

**Technical Deep Dives:**
1. Read Nvidia's CUDA C++ Programming Guide (cover to cover)
2. Study Nsight Compute metrics and what they mean
3. Implement classic parallel algorithms from scratch
4. Profile real applications and optimize them

**Architecture Knowledge:**
1. Read GPU architecture whitepapers (Ampere, Hopper)
2. Understand memory hierarchy deeply
3. Learn about Tensor Cores and their usage
4. Study NVLink and multi-GPU programming

**Practical Skills:**
1. Contribute to open-source GPU projects
2. Implement papers from NVIDIA Research
3. Attend GTC (GPU Technology Conference) talks
4. Practice on real GPUs (Google Colab free tier)

**Interview Simulation:**
1. Practice explaining concepts to non-experts
2. Time yourself solving coding problems
3. Record yourself and review communication
4. Do mock interviews with peers

---

**Remember:**
- **Communication is key** - explain your thought process
- **Show depth** - understand the "why" not just the "how"
- **Be honest** - say "I don't know" and explain how you'd find out
- **Ask questions** - interviews are two-way conversations
- **Stay calm** - take your time to think before answering

Good luck with your Nvidia interview! 🚀
