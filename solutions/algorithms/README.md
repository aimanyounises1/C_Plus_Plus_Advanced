# GPU-Accelerated Algorithm Problems

This directory contains LeetCode-style algorithm problems with both CPU and GPU implementations. These are designed to prepare you for technical interviews at Nvidia and other GPU computing companies.

## Why GPU Algorithms Matter for Interviews

While traditional algorithm interviews focus on sequential solutions, Nvidia interviewers often ask:
- "How would you parallelize this algorithm?"
- "What's the parallel complexity?"
- "Can this benefit from GPU acceleration?"

Understanding parallel algorithm design demonstrates deeper systems thinking.

## Problem Categories

### Array & String Problems
Classic problems with parallel implementations:
- Two Sum (parallel hash table)
- Maximum Subarray (parallel prefix sum)
- Longest Substring Without Repeating Characters
- Container With Most Water

### Sorting & Searching
GPU-optimized algorithms:
- Parallel Merge Sort
- Bitonic Sort (GPU-native)
- Parallel Binary Search
- Quick Select on GPU

### Dynamic Programming
Converting sequential DP to parallel:
- Longest Increasing Subsequence
- Edit Distance (diagonal parallelization)
- Matrix Chain Multiplication

### Graph Algorithms
Essential for parallel computing:
- BFS (level-synchronous)
- DFS (work-stealing approach)
- Shortest Path (parallel Bellman-Ford)
- Connected Components (label propagation)

### Tree Problems
Parallel tree traversal:
- Parallel Tree Traversal (in/pre/post-order)
- Lowest Common Ancestor (parallel preprocessing)
- Diameter of Binary Tree

## How to Use These Problems

### For Self-Study
1. Read the problem description
2. Try implementing sequential solution first (20-30 min)
3. Think about parallelization opportunities
4. Implement parallel version
5. Compare with reference solution
6. Analyze complexity and performance

### For Interview Prep
1. Practice explaining your thought process out loud
2. Start with brute force, then optimize
3. Discuss trade-offs (CPU vs GPU)
4. Mention memory constraints
5. Consider when GPU acceleration makes sense

### Interview Strategy

**When asked an algorithm question:**

1. **Clarify requirements** (5 min)
   - Input size and range
   - Memory constraints
   - Expected latency
   - Single vs batch processing

2. **Propose sequential solution** (10-15 min)
   - Brute force first
   - Optimize time/space complexity
   - Explain big-O analysis

3. **Discuss parallelization** (10-15 min)
   - Identify independent operations
   - Data dependencies
   - Parallel complexity (work/span)
   - GPU suitability

4. **Implement and test** (15-20 min)
   - Write clean, compilable code
   - Handle edge cases
   - Verify correctness

5. **Optimize if time permits** (5-10 min)
   - Memory coalescing
   - Shared memory usage
   - Occupancy optimization

## Compilation

Each problem includes compilation instructions. General format:

```bash
# CPU version
g++ -o problem problem_cpu.cpp -O3 -std=c++17

# GPU version
nvcc -o problem problem_gpu.cu -O3 -std=c++17 -arch=sm_70

# Both (with unified source)
nvcc -o problem problem.cu -O3 -std=c++17 -arch=sm_70
```

## Performance Metrics to Track

- **Speedup**: GPU time / CPU time
- **Efficiency**: Speedup / (# of CUDA cores)
- **Bandwidth Utilization**: Actual / Peak bandwidth
- **Compute Utilization**: % of peak FLOPS
- **Scalability**: Performance vs input size

## Common Parallel Patterns

These patterns appear repeatedly:

1. **Map**: Apply function to each element independently
2. **Reduce**: Combine elements (sum, max, min)
3. **Scan**: Prefix sum and variants
4. **Scatter**: Write to irregular locations
5. **Gather**: Read from irregular locations
6. **Stencil**: Compute from neighbors
7. **Pack/Filter**: Compact sparse data
8. **Expand**: Replicate elements

## GPU Algorithm Design Principles

### When GPU Acceleration Helps
✓ Large data parallel operations (N > 10K)
✓ Regular memory access patterns
✓ High arithmetic intensity
✓ Embarrassingly parallel problems
✓ Repeated operations (amortize transfer cost)

### When GPU May Not Help
✗ Small problem sizes (N < 1K)
✗ Highly irregular/recursive algorithms
✗ Lots of branching/divergence
✗ Sequential dependencies
✗ One-time operations with large transfer overhead

## Complexity Analysis

### Sequential Complexity
- Time: T(n)
- Space: S(n)

### Parallel Complexity
- **Work**: Total operations across all processors
- **Span (Depth)**: Longest dependency chain
- **Parallelism**: Work / Span (max theoretical speedup)

**Example: Parallel Reduction**
- Sequential: T(n) = O(n), W(n) = O(n)
- Parallel: T_p(n) = O(log n), W(n) = O(n)
- Parallelism: O(n / log n)

## Interview Tips

**Common Mistakes to Avoid:**
1. Not considering memory transfer overhead
2. Ignoring thread divergence
3. Poor memory access patterns
4. Not handling edge cases (N < block size, etc.)
5. Forgetting error checking

**What Interviewers Look For:**
1. Problem decomposition skills
2. Understanding of parallel primitives
3. Awareness of GPU architecture constraints
4. Trade-off analysis
5. Clean, correct code

**Key Points to Mention:**
- Memory coalescing importance
- Bank conflict avoidance
- Occupancy considerations
- When to use shared memory
- Synchronization overhead

## Resources

- **Parallel Algorithms**: "Introduction to Parallel Computing" by Grama et al.
- **CUDA Patterns**: Nvidia CUDA C++ Programming Guide (Appendix B)
- **CUB Library**: Nvidia's collective primitives (reference implementations)
- **Thrust**: High-level parallel algorithms library

## Next Steps

After mastering these problems:
1. Implement your own parallel algorithm library
2. Contribute to open-source GPU projects
3. Profile real applications (PyTorch, TensorFlow)
4. Read Nvidia's GTC presentations
5. Practice on actual GPU hardware

---

**Remember**: In interviews, communication is as important as the solution. Explain your reasoning, discuss trade-offs, and show you understand when and why to use GPUs.
