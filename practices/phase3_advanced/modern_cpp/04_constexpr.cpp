/*
 * ==================================================================================================
 * Exercise: Constexpr and Compile-Time Programming
 * ==================================================================================================
 * Difficulty: Intermediate/Advanced | Time: 40-50 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master constexpr functions
 * 2. Understand compile-time computation
 * 3. Learn consteval (C++20)
 * 4. Practice constexpr containers (C++20)
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Zero-runtime-cost abstractions
 * - Compile-time GPU configuration
 * - Template metaprogramming
 * - Performance optimization
 * ==================================================================================================
 */

#include <iostream>
#include <array>
using namespace std;

/*
 * EXERCISE 1: Basic constexpr (10 min)
 */

constexpr int square(int x) {
    return x * x;
}

constexpr int fibonacci(int n) {
    return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2);
}

void constexprBasics() {
    constexpr int x = square(5);  // Computed at compile time
    constexpr int fib10 = fibonacci(10);

    cout << "square(5) = " << x << endl;
    cout << "fib(10) = " << fib10 << endl;

    // Can also call at runtime
    int n = 6;
    cout << "square(6) = " << square(n) << endl;
}

/*
 * EXERCISE 2: constexpr Classes (10 min)
 */

class Point {
    int x_, y_;
public:
    constexpr Point(int x, int y) : x_(x), y_(y) {}

    constexpr int x() const { return x_; }
    constexpr int y() const { return y_; }

    constexpr int distanceSquared() const {
        return x_*x_ + y_*y_;
    }
};

void constexprClasses() {
    constexpr Point p(3, 4);
    constexpr int dist = p.distanceSquared();
    cout << "Distance squared: " << dist << endl;
}

/*
 * EXERCISE 3: Constexpr Arrays (10 min)
 */

template<size_t N>
constexpr auto generateSquares() {
    array<int, N> result{};
    for (size_t i = 0; i < N; i++) {
        result[i] = i * i;
    }
    return result;
}

void constexprArrays() {
    constexpr auto squares = generateSquares<10>();
    for (int val : squares) {
        cout << val << " ";
    }
    cout << endl;
}

/*
 * EXERCISE 4: consteval (C++20) (10 min)
 */

#if __cplusplus >= 202002L

// consteval: MUST be evaluated at compile time
consteval int compileTimeOnly(int x) {
    return x * x;
}

void constevalExample() {
    constexpr int x = compileTimeOnly(5);  // OK
    // int n = 5;
    // int y = compileTimeOnly(n);  // ERROR: n not constexpr
}

#endif

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is constexpr?
 * A: Function/variable that CAN be evaluated at compile time
 *
 * Q2: constexpr vs const?
 * A: const: Runtime constant
 *    constexpr: Compile-time constant (can also be runtime)
 *
 * Q3: What is consteval? (C++20)
 * A: MUST be evaluated at compile time (no runtime evaluation)
 *
 * Q4: Benefits of constexpr?
 * A: - Zero runtime cost
 *    - Can use in constant expressions
 *    - Enables template metaprogramming
 *
 * Q5: Limitations of constexpr?
 * A: - No dynamic allocation (until C++20)
 *    - No virtual functions
 *    - Limited standard library support
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Compile-time grid/block size calculations
 * - Constexpr device properties
 * - Zero-cost abstractions for GPU code
 * - Template-based kernel configuration
 *
 * COMPILATION: g++ -std=c++20 04_constexpr.cpp -o constexpr
 * ==================================================================================================
 */

int main() {
    cout << "=== Constexpr Practice ===" << endl;

    constexprBasics();
    constexprClasses();
    constexprArrays();

#if __cplusplus >= 202002L
    constevalExample();
#else
    cout << "C++20 consteval not available" << endl;
#endif

    return 0;
}
