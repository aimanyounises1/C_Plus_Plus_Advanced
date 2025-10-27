/*
 * ==================================================================================================
 * Exercise: Ranges and Coroutines (C++20)
 * ==================================================================================================
 * Difficulty: Advanced | Time: 50-60 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master std::ranges for data processing
 * 2. Understand range adaptors
 * 3. Learn coroutines basics (co_yield, co_await)
 * 4. Practice modern async patterns
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Modern data processing patterns
 * - Async GPU operations
 * - Pipeline-style algorithms
 * - C++20 features
 * ==================================================================================================
 */

#include <iostream>
#include <vector>
#include <algorithm>

#if __cplusplus >= 202002L
#include <ranges>
namespace views = std::ranges::views;
#endif

using namespace std;

/*
 * EXERCISE 1: Basic Ranges (15 min)
 */

void rangesBasics() {
#if __cplusplus >= 202002L
    vector<int> nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Filter and transform in one pipeline
    auto result = nums
        | views::filter([](int n) { return n % 2 == 0; })
        | views::transform([](int n) { return n * n; });

    cout << "Even numbers squared: ";
    for (int n : result) {
        cout << n << " ";
    }
    cout << endl;

    // Take first N
    auto first5 = nums | views::take(5);
    for (int n : first5) {
        cout << n << " ";
    }
    cout << endl;
#else
    cout << "C++20 ranges not available" << endl;
#endif
}

/*
 * EXERCISE 2: Range Adaptors (10 min)
 */

void rangeAdaptors() {
#if __cplusplus >= 202002L
    vector<int> nums = {1, 2, 3, 4, 5};

    // Multiple adaptors chained
    auto result = nums
        | views::transform([](int n) { return n * 2; })
        | views::filter([](int n) { return n > 5; })
        | views::take(3);

    for (int n : result) {
        cout << n << " ";
    }
    cout << endl;
#else
    cout << "C++20 ranges not available" << endl;
#endif
}

/*
 * EXERCISE 3: Coroutines Basics (20 min)
 *
 * Note: Full coroutine support requires more boilerplate
 * This is a simplified demonstration
 */

// Simplified generator (requires more setup in real code)
void coroutinesIntro() {
    cout << "Coroutines: co_yield, co_await, co_return" << endl;
    cout << "Full implementation requires custom promise_type" << endl;
    cout << "Used for: async operations, generators, lazy evaluation" << endl;
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What are ranges?
 * A: C++20: Composable, lazy-evaluated views over sequences
 *
 * Q2: Benefits of ranges?
 * A: - Lazy evaluation
 *    - Composable pipelines
 *    - More readable than iterators
 *    - Zero-overhead abstractions
 *
 * Q3: What are coroutines?
 * A: Functions that can suspend and resume execution
 *    Keywords: co_yield, co_await, co_return
 *
 * Q4: Use cases for coroutines?
 * A: - Async I/O
 *    - Generators (lazy sequences)
 *    - State machines
 *    - Event-driven code
 *
 * Q5: Coroutines vs threads?
 * A: Coroutines: Cooperative, lightweight, single-threaded
 *    Threads: Preemptive, heavyweight, parallel
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Ranges for processing GPU results
 * - Coroutines for async GPU operations
 * - Pipeline-style data processing
 * - Modern async patterns with CUDA streams
 *
 * Example: auto results = gpuResults | views::filter(valid) | views::transform(process);
 *
 * COMPILATION: g++ -std=c++20 -fcoroutines 05_ranges_coroutines.cpp -o ranges
 * ==================================================================================================
 */

int main() {
    cout << "=== Ranges and Coroutines Practice ===" << endl;

    rangesBasics();
    rangeAdaptors();
    coroutinesIntro();

    return 0;
}
