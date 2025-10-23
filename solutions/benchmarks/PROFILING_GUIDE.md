# CUDA Profiling and Performance Analysis Guide

This guide covers essential profiling tools and techniques for optimizing CUDA applications. Master these for Nvidia interviews!

---

## Table of Contents
1. [Profiling Tools Overview](#profiling-tools-overview)
2. [Nsight Systems - Timeline Profiling](#nsight-systems)
3. [Nsight Compute - Kernel Analysis](#nsight-compute)
4. [Key Metrics to Monitor](#key-metrics)
5. [Performance Optimization Workflow](#optimization-workflow)
6. [Common Bottlenecks and Solutions](#common-bottlenecks)
7. [Interview-Relevant Metrics](#interview-metrics)

---

## Profiling Tools Overview

### Essential Nvidia Tools

| Tool | Purpose | Use When |
|------|---------|----------|
| **Nsight Systems** | Application-level timeline | Finding high-level bottlenecks, understanding data flow |
| **Nsight Compute** | Detailed kernel profiling | Optimizing specific kernels |
| **nvprof** | Legacy profiler | Quick checks (being phased out) |
| **cuda-memcheck** | Memory error detection | Debugging memory issues |
| **compute-sanitizer** | Modern memory checker | Replacing cuda-memcheck |

### Installation
```bash
# Usually included with CUDA Toolkit
# If not:
# Download from: https://developer.nvidia.com/nsight-systems
# And: https://developer.nvidia.com/nsight-compute
```

---

## Nsight Systems - Timeline Profiling

### Basic Usage

```bash
# Generate profiling report
nsys profile --stats=true -o report ./program

# With specific duration
nsys profile --duration=10 --stats=true ./program

# Only CUDA APIs
nsys profile --trace=cuda --stats=true ./program

# Include CPU sampling
nsys profile --sample=cpu --stats=true ./program
```

### What to Look For

**1. Kernel Launch Overhead**
```
If you see: Many small kernels with gaps between them
Problem: Launch overhead dominates
Solution: Kernel fusion, persistent kernels
```

**2. Host-Device Transfer Bottlenecks**
```
If you see: Long cudaMemcpy bars
Problem: Too much data movement
Solution: Unified Memory, keep data on device, overlapping with streams
```

**3. CPU-GPU Synchronization**
```
If you see: GPU idle between kernels
Problem: Implicit synchronization (cudaMemcpy, cudaDeviceSynchronize)
Solution: Async operations, streams, events
```

**4. Pipeline Stalls**
```
If you see: Gaps in timeline
Problem: Dependencies preventing overlap
Solution: Multiple streams, reorder operations
```

### Example Analysis

```bash
# Profile the application
nsys profile --stats=true -o matmul_profile ./matmul

# View in GUI
nsys-ui matmul_profile.qdrep

# Command-line stats
nsys stats matmul_profile.qdrep
```

**Good Timeline (Optimized):**
```
[====CPU Compute====]
                    [====Memcpy H2D====]
                                       [====Kernel1====][====Kernel2====]
                                                                        [====Memcpy D2H====]
```

**Bad Timeline (Unoptimized):**
```
[==CPU==]  (idle)  [==CPU==]  (idle)  [==CPU==]
          [MemH2D]          [Kernel]          [MemD2H]
```

---

## Nsight Compute - Kernel Analysis

### Basic Usage

```bash
# Full profiling (slow but comprehensive)
ncu --set full -o kernel_profile ./program

# Specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./program

# Profile specific kernel
ncu --kernel-name "myKernel" ./program

# First N kernel launches only
ncu --launch-count 5 ./program

# Interactive mode
ncu --mode=launch-and-attach ./program
```

### Critical Metrics

#### 1. Memory Bound Analysis

```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./program
```

**What it means:**
- < 60%: Memory bandwidth underutilized → optimize access patterns
- 60-90%: Good utilization
- > 90%: Excellent! Memory bound kernel

**Related metrics:**
```bash
# Memory bandwidth breakdown
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./program

# L2 cache hit rate
ncu --metrics lts__t_sectors_op_read_hit_rate.pct ./program

# Coalescing efficiency
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./program
```

#### 2. Compute Bound Analysis

```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./program
```

**What it means:**
- < 60%: Compute units underutilized
- 60-90%: Good compute utilization
- > 90%: Compute bound kernel

**Related metrics:**
```bash
# FP32 operations
ncu --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum ./program

# Tensor Core utilization (if applicable)
ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed ./program
```

#### 3. Occupancy Analysis

```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./program
```

**What it means:**
- < 30%: Very low occupancy → check resource usage
- 30-50%: Moderate (may be OK if compute-bound)
- 50-100%: Good to excellent

**What limits occupancy:**
```bash
# Register usage per thread
ncu --metrics launch__registers_per_thread ./program

# Shared memory per block
ncu --metrics launch__shared_mem_per_block_allocated ./program

# Threads per block
ncu --metrics launch__threads_per_block ./program

# Theoretical occupancy
ncu --metrics sm__maximum_warps_per_active_cycle_pct ./program
```

#### 4. Warp Execution Efficiency

```bash
# Warp divergence
ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./program
```

**What it means:**
- Ideal: 32 (all threads active)
- < 16: Significant divergence

```bash
# Stall reasons
ncu --metrics smsp__average_warps_issue_stalled_short_scoreboard.pct,smsp__average_warps_issue_stalled_long_scoreboard.pct,smsp__average_warps_issue_stalled_membar.pct ./program
```

#### 5. Bank Conflicts

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./program
```

**What it means:**
- 0: Perfect! No bank conflicts
- > 0: Bank conflicts present → add padding or change access pattern

---

## Key Metrics to Monitor

### Memory Metrics

| Metric | What It Measures | Goal |
|--------|------------------|------|
| `dram__throughput` | Global memory bandwidth utilization | > 60% |
| `l2_cache_hit_rate` | L2 cache effectiveness | > 80% |
| `global_load_efficiency` | Coalescing efficiency | > 80% |
| `shared_memory_throughput` | Shared memory bandwidth | Context-dependent |

### Compute Metrics

| Metric | What It Measures | Goal |
|--------|------------------|------|
| `sm__throughput` | SM compute utilization | > 60% |
| `flop_count_sp` | Single-precision FLOPs | Maximize |
| `flop_sp_efficiency` | How efficiently using FP units | > 60% |
| `ipc` | Instructions per cycle | Maximize |

### Occupancy Metrics

| Metric | What It Measures | Goal |
|--------|------------------|------|
| `achieved_occupancy` | Actual occupancy | 50-100% |
| `registers_per_thread` | Register usage | Minimize (if low occ) |
| `shared_memory_per_block` | Shared mem usage | Balance with occupancy |

### Execution Metrics

| Metric | What It Measures | Goal |
|--------|------------------|------|
| `warp_execution_efficiency` | Thread utilization in warps | > 90% |
| `branch_efficiency` | Control flow efficiency | > 90% |
| `stall_reasons` | Why warps are stalling | Minimize |

---

## Performance Optimization Workflow

### Step 1: Baseline Measurement

```bash
# Run with full profiling
ncu --set full -o baseline ./program

# Note key metrics:
# - Kernel execution time
# - Memory throughput
# - Compute throughput
# - Occupancy
```

### Step 2: Identify Bottleneck

**Decision Tree:**

```
Is memory throughput > 60%?
├─ YES → Memory Bound
│  ├─ Is global load efficiency < 80%? → Fix coalescing
│  ├─ Is L2 hit rate < 70%? → Improve locality
│  └─ Otherwise → Memory inherently limited
│
└─ NO → Check compute throughput
   ├─ Is compute throughput > 60%? → Compute Bound
   │  ├─ Is occupancy < 50%? → Increase parallelism
   │  └─ Otherwise → Compute inherently limited
   │
   └─ Both low? → Latency Bound
      ├─ Check stall reasons
      └─ Increase occupancy to hide latency
```

### Step 3: Apply Targeted Optimizations

**If Memory Bound:**
```cpp
// 1. Improve coalescing
// Before:
float val = data[threadIdx.x * stride];  // Strided

// After:
float val = data[threadIdx.x];  // Sequential

// 2. Use shared memory
__shared__ float tile[TILE_SIZE];
tile[threadIdx.x] = globalData[...];
__syncthreads();
// Now use tile[] for multiple accesses

// 3. Increase data reuse
// Prefetch data into registers
float reg0 = data[idx];
float reg1 = data[idx + offset];
// Use both values multiple times
```

**If Compute Bound:**
```cpp
// 1. Use faster math
__fdividef(a, b);  // Instead of a / b
__expf(x);         // Instead of exp(x)

// 2. Enable FMA
c = __fmaf_rn(a, b, c);  // a*b + c

// 3. Unroll loops
#pragma unroll
for (int i = 0; i < N; i++) {
    // Loop body
}

// 4. Use Tensor Cores (if applicable)
// For matrix multiply: use WMMA API or cuBLAS
```

**If Latency Bound:**
```cpp
// 1. Increase occupancy
__launch_bounds__(256, 4)  // Min 4 blocks per SM

// 2. Reduce register usage
// Spill less-used values to shared memory

// 3. ILP (Instruction-Level Parallelism)
// Process multiple elements per thread
float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
for (int i = 0; i < N; i += 4) {
    sum0 += data[i + 0];
    sum1 += data[i + 1];
    sum2 += data[i + 2];
    sum3 += data[i + 3];
}
```

### Step 4: Measure Again

```bash
ncu --set full -o optimized ./program

# Compare metrics
ncu --import baseline.ncu-rep --import optimized.ncu-rep
```

### Step 5: Iterate

Keep optimizing until:
- Performance meets requirements, OR
- Hitting theoretical limits (> 90% of peak), OR
- Diminishing returns (< 5% improvement per iteration)

---

## Common Bottlenecks and Solutions

### Bottleneck 1: Uncoalesced Memory Access

**Symptoms:**
- Low `global_memory_load_efficiency` (< 50%)
- High `dram__bytes_read` relative to actual data needed

**Diagnosis:**
```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg ./program
```

**Solution:**
```cpp
// BAD: Strided access
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    result += data[i * STRIDE];
}

// GOOD: Sequential access
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    result += data[i];
}
```

### Bottleneck 2: Bank Conflicts

**Symptoms:**
- `shared_memory_bank_conflicts` > 0
- Shared memory operations slower than expected

**Diagnosis:**
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./program
```

**Solution:**
```cpp
// BAD: 32-way conflicts
__shared__ float data[32][32];
float val = data[threadIdx.x][0];  // All threads access column 0

// GOOD: Padding eliminates conflicts
__shared__ float data[32][33];  // +1 padding
float val = data[threadIdx.x][0];  // No conflicts!
```

### Bottleneck 3: Low Occupancy

**Symptoms:**
- `achieved_occupancy` < 30%
- Both memory and compute underutilized

**Diagnosis:**
```bash
ncu --metrics launch__registers_per_thread,launch__shared_mem_per_block_allocated ./program
```

**Solutions:**

**Too many registers:**
```cpp
// Option 1: Compiler directive
__launch_bounds__(256, 4)  // 256 threads, 4 blocks/SM
__global__ void kernel() { ... }

// Option 2: Compiler flag
nvcc -maxrregcount=64 kernel.cu

// Option 3: Reduce local variables
// Move some to shared memory
```

**Too much shared memory:**
```cpp
// Reduce tile size
#define TILE_SIZE 16  // Instead of 32

// Or use dynamic shared memory
extern __shared__ float dynamic_mem[];
```

### Bottleneck 4: Warp Divergence

**Symptoms:**
- `warp_execution_efficiency` < 80%
- `branch_efficiency` < 80%

**Diagnosis:**
```bash
ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./program
```

**Solution:**
```cpp
// BAD: Divergent branches
if (threadIdx.x % 2 == 0) {
    // Half threads do this
} else {
    // Other half do this
}

// GOOD: Predicated execution
int mask = (threadIdx.x % 2 == 0);
result = mask * path_a + (1 - mask) * path_b;
```

### Bottleneck 5: Excessive Synchronization

**Symptoms:**
- `__syncthreads()` appears frequently
- High `stall_membar` percentage

**Solution:**
```cpp
// Reduce synchronization points
// Reorganize algorithm to need fewer syncs

// Use warp-level primitives (no sync needed)
float val = __shfl_down_sync(0xffffffff, value, 1);

// Double buffering to overlap sync with compute
__shared__ float bufferA[SIZE], bufferB[SIZE];
// Load to bufferA, sync, compute
// Meanwhile, load next data to bufferB
```

---

## Interview-Relevant Metrics

### Questions You'll Be Asked

**Q: "How do you identify if a kernel is memory-bound or compute-bound?"**

**A:** "I use Nsight Compute to check:
1. `dram__throughput.avg.pct_of_peak`: Memory utilization
2. `sm__throughput.avg.pct_of_peak`: Compute utilization

If memory > 60% and compute < 40% → Memory bound
If compute > 60% and memory < 40% → Compute bound
If both low → Latency bound (increase occupancy)
If both high → Well balanced (rare)"

---

**Q: "What's a good occupancy target?"**

**A:** "It depends on the kernel characteristics:
- Memory-bound kernels: Need high occupancy (70-100%) to hide memory latency
- Compute-bound kernels: Can tolerate lower occupancy (30-50%) if registers enable better ILP
- Generally aim for 50%+ as baseline
- Profile to find sweet spot - sometimes more registers/shared mem with lower occupancy performs better"

---

**Q: "How do you detect and fix bank conflicts?"**

**A:** "Use Nsight Compute:
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
```

If non-zero, fix by:
1. Padding shared memory arrays: `float data[N][N+1]` instead of `[N][N]`
2. Changing access pattern to avoid same-bank access
3. Using XOR-based indexing: `idx = i ^ (i / 32)`

Verify fix by confirming metric goes to 0."

---

**Q: "What tools would you use to profile a slow CUDA application?"**

**A:** "I follow this hierarchy:

1. **Nsight Systems** (High-level):
   - Identify which kernels take most time
   - Find data transfer bottlenecks
   - Check for CPU-GPU synchronization issues

2. **Nsight Compute** (Kernel-level):
   - Deep dive into expensive kernels
   - Analyze memory/compute balance
   - Check occupancy and efficiency metrics

3. **Code-level Timing**:
   - CUDA events for precise kernel timing
   - Validate profiling observations

4. **Compute-sanitizer** (If crashes/errors):
   - Memory errors, race conditions"

---

## Practical Example: Complete Optimization Session

Let's profile and optimize a real kernel:

```cpp
// Initial kernel (unoptimized)
__global__ void matmul_naive(float* A, float* B, float* C, int N) {
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
```

### Profiling Session

```bash
# Step 1: Baseline
$ ncu --set full -o baseline ./matmul_naive

# Key findings:
# - Kernel time: 45.2 ms
# - dram__throughput: 78% (memory bound!)
# - global_load_efficiency: 35% (BAD coalescing)
# - achieved_occupancy: 48%
```

### Optimization 1: Tiling with Shared Memory

```cpp
__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N/TILE; t++) {
        // Load tiles
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();

        // Compute
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}
```

```bash
$ ncu --set full -o opt1 ./matmul_tiled

# Results:
# - Kernel time: 8.7 ms (5.2x faster!)
# - dram__throughput: 45% (reduced global mem traffic)
# - achieved_occupancy: 52%
# - bank_conflicts: 8192 (need to fix!)
```

### Optimization 2: Fix Bank Conflicts

```cpp
// Add padding to shared memory
__shared__ float As[TILE][TILE+1];  // +1 padding
__shared__ float Bs[TILE][TILE+1];
```

```bash
$ ncu --set full -o opt2 ./matmul_padded

# Results:
# - Kernel time: 7.1 ms (6.4x faster than baseline!)
# - bank_conflicts: 0 (fixed!)
# - dram__throughput: 44%
```

### Final Comparison

| Version | Time | Speedup | Key Improvement |
|---------|------|---------|-----------------|
| Naive | 45.2 ms | 1.0x | Baseline |
| Tiled | 8.7 ms | 5.2x | Shared memory reuse |
| Padded | 7.1 ms | 6.4x | Eliminated bank conflicts |

---

## Quick Reference: Essential Commands

```bash
# Timeline profiling
nsys profile --stats=true ./program

# Kernel profiling (common metrics)
ncu --metrics \
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
  ./program

# Check memory errors
compute-sanitizer --tool memcheck ./program

# Race condition detection
compute-sanitizer --tool racecheck ./program

# Check for sync errors
compute-sanitizer --tool synccheck ./program
```

---

## Resources

- [Nvidia Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Nvidia Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [CUDA Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [Nvidia Developer Blog: Profiling](https://developer.nvidia.com/blog/tag/profiling/)

---

**Remember**: Profile first, optimize second. Premature optimization is the root of all evil, but informed optimization is the path to glory!
