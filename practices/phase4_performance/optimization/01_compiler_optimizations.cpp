/*
 * ==================================================================================================
 * Exercise: Compiler Optimizations
 * ==================================================================================================
 * Difficulty: Intermediate/Advanced | Time: 45-55 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Understand compiler optimization levels (-O0, -O2, -O3)
 * 2. Learn advanced optimization flags (LTO, PGO)
 * 3. Master inline, constexpr, and compiler hints
 * 4. Practice reading assembly output
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - NVCC optimization flags
 * - Understanding CUDA compiler behavior
 * - PTX/SASS assembly analysis
 * - Kernel optimization strategies
 * ==================================================================================================
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
using namespace std;
using namespace chrono;

/*
 * EXERCISE 1: Optimization Levels (15 min)
 * Compile this file with different flags to see the impact:
 * g++ -O0 01_compiler_optimizations.cpp -o opt0  # No optimization
 * g++ -O2 01_compiler_optimizations.cpp -o opt2  # Standard optimization
 * g++ -O3 01_compiler_optimizations.cpp -o opt3  # Aggressive optimization
 */

double sumSquares(const vector<double>& data) {
    double sum = 0.0;
    for (size_t i = 0; i < data.size(); i++) {
        sum += data[i] * data[i];
    }
    return sum;
}

void optimizationLevelsDemo() {
    const int N = 100000000;
    vector<double> data(N, 1.5);

    auto start = high_resolution_clock::now();
    double result = sumSquares(data);
    auto end = high_resolution_clock::now();
    auto time = duration_cast<milliseconds>(end - start).count();

    cout << "Sum of squares: " << result << endl;
    cout << "Time: " << time << " ms" << endl;
    cout << "Run with -O0, -O2, -O3 to compare!" << endl;
}

/*
 * EXERCISE 2: Inline Functions (10 min)
 */

// Small function - good candidate for inlining
inline int square(int x) {
    return x * x;
}

// Force inline (compiler hint)
__attribute__((always_inline)) inline int cube(int x) {
    return x * x * x;
}

// Prevent inlining
__attribute__((noinline)) int expensiveFunction(int x) {
    int result = 0;
    for (int i = 0; i < 1000; i++) {
        result += x * i;
    }
    return result;
}

void inliningDemo() {
    int sum = 0;
    auto start = high_resolution_clock::now();

    for (int i = 0; i < 1000000; i++) {
        sum += square(i % 100);  // Likely inlined
    }

    auto end = high_resolution_clock::now();
    cout << "Inline test sum: " << sum << endl;
    cout << "Time: " << duration_cast<microseconds>(end - start).count() << " µs" << endl;
}

/*
 * EXERCISE 3: Constexpr for Compile-Time Computation (10 min)
 */

constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

constexpr int fibonacci(int n) {
    return (n <= 1) ? n : fibonacci(n - 1) + fibonacci(n - 2);
}

void constexprDemo() {
    // Computed at compile time!
    constexpr int fact10 = factorial(10);
    constexpr int fib10 = fibonacci(10);

    cout << "Factorial(10): " << fact10 << " (computed at compile time)" << endl;
    cout << "Fibonacci(10): " << fib10 << " (computed at compile time)" << endl;

    // Check assembly: no function calls, just constants!
}

/*
 * EXERCISE 4: Branch Hints (likely/unlikely) (10 min)
 */

int processWithHints(int x) {
    // Tell compiler this branch is likely
    if (__builtin_expect(x > 0, 1)) {  // Likely true
        return x * 2;
    } else {
        return x * 3;  // Unlikely path
    }
}

void branchHintsDemo() {
    int sum = 0;
    auto start = high_resolution_clock::now();

    for (int i = 0; i < 10000000; i++) {
        sum += processWithHints(i);  // Almost always positive
    }

    auto end = high_resolution_clock::now();
    cout << "Branch hints sum: " << sum << endl;
    cout << "Time: " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: Compiler optimization levels?
 * A: -O0: No optimization (debugging)
 *    -O1: Basic optimization
 *    -O2: Standard (inlining, dead code elimination, CSE)
 *    -O3: Aggressive (vectorization, more inlining)
 *    -Ofast: -O3 + fast math (breaks IEEE compliance)
 *
 * Q2: What is LTO (Link-Time Optimization)?
 * A: Optimization across translation units
 *    Enables inlining across files
 *    Flag: -flto
 *
 * Q3: What is PGO (Profile-Guided Optimization)?
 * A: Use runtime profiling to guide optimization
 *    1. Compile with -fprofile-generate
 *    2. Run program to collect profile
 *    3. Recompile with -fprofile-use
 *
 * Q4: When to use inline?
 * A: - Small, frequently called functions
 *    - Header-only templates
 *    - Modern compilers inline automatically
 *    - inline keyword is mostly a hint
 *
 * Q5: Inline vs macro?
 * A: Inline: Type-safe, debuggable, respects scope
 *    Macro: Text substitution, no type checking
 *    Always prefer inline
 *
 * Q6: What is constexpr?
 * A: Compute at compile time if possible
 *    Zero runtime cost
 *    Enables constant folding
 *
 * Q7: __builtin_expect(expr, value)?
 * A: Branch prediction hint
 *    value: expected result (1=likely, 0=unlikely)
 *    Helps CPU branch predictor
 *
 * Q8: How to read compiler output?
 * A: - g++ -S generates assembly
 *    - g++ -fopt-info shows optimizations
 *    - objdump -d for disassembly
 *    - Compiler Explorer (godbolt.org)
 *
 * Q9: What optimizations does -O3 enable?
 * A: - Aggressive inlining
 *    - Loop vectorization (SIMD)
 *    - Loop unrolling
 *    - Function specialization
 *    - May increase code size
 *
 * Q10: Fast math (-ffast-math)?
 * A: Breaks IEEE 754 compliance
 *    - No NaN/infinity checks
 *    - Reorder FP operations
 *    - Use for graphics, avoid for finance
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 *
 * NVCC compiler optimization parallels:
 *
 * 1. Optimization flags:
 *    nvcc -O0  # Debug mode
 *    nvcc -O3  # Maximum optimization (default)
 *    nvcc --use_fast_math  # Like -ffast-math
 *
 * 2. Inline in CUDA:
 *    __forceinline__ __device__ float square(float x) {
 *        return x * x;
 *    }
 *    Critical for device functions to avoid overhead
 *
 * 3. Constexpr in CUDA:
 *    constexpr int BLOCK_SIZE = 256;
 *    kernel<<<grid, BLOCK_SIZE>>>(...);
 *    Compile-time constants for kernel config
 *
 * 4. Branch hints:
 *    if (__builtin_expect(threadIdx.x < WARP_SIZE, 1)) {
 *        // Likely path
 *    }
 *
 * 5. PTX/SASS analysis:
 *    nvcc -ptx kernel.cu          # Generate PTX
 *    nvcc -cubin kernel.cu        # Generate SASS
 *    cuobjdump -sass kernel.cubin # Disassemble
 *
 * 6. Profile-Guided Optimization:
 *    Not directly supported, but:
 *    - Use Nsight Compute for profiling
 *    - Manually optimize based on metrics
 *    - Adjust kernel launch parameters
 *
 * Real NVIDIA interview question:
 * "Why is __forceinline__ important for device functions?"
 * Answer: Device function calls have significant overhead (register
 * pressure, instruction cache). Inlining eliminates call overhead and
 * enables further optimizations like constant propagation.
 *
 * COMPILATION:
 * g++ -O0 01_compiler_optimizations.cpp -o opt0  # ~1000ms
 * g++ -O2 01_compiler_optimizations.cpp -o opt2  # ~100ms
 * g++ -O3 01_compiler_optimizations.cpp -o opt3  # ~80ms (vectorized)
 * ==================================================================================================
 */

int main() {
    cout << "=== Compiler Optimizations Practice ===" << endl;

    cout << "\n1. Optimization Levels:" << endl;
    optimizationLevelsDemo();

    cout << "\n2. Inline Functions:" << endl;
    inliningDemo();

    cout << "\n3. Constexpr (Compile-Time):" << endl;
    constexprDemo();

    cout << "\n4. Branch Hints:" << endl;
    branchHintsDemo();

    cout << "\nTip: Compile with -O0, -O2, -O3 and compare times!" << endl;
    cout << "Also try: g++ -S to see assembly output" << endl;

    return 0;
}
