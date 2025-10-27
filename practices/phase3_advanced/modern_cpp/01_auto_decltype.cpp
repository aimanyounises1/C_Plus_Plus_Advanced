/*
 * ==================================================================================================
 * Exercise: Auto and Decltype in Modern C++
 * ==================================================================================================
 * Difficulty: Intermediate | Time: 35-45 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master auto keyword for type deduction
 * 2. Understand decltype for type extraction
 * 3. Learn decltype(auto) for perfect forwarding
 * 4. Practice AAA (Almost Always Auto) idiom
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Cleaner generic GPU code
 * - Type-safe kernel wrappers
 * - Template return type deduction
 * - Modern C++ best practices
 * ==================================================================================================
 */

#include <iostream>
#include <vector>
#include <map>
using namespace std;

/*
 * EXERCISE 1: Basic auto (10 min)
 */

void basicAuto() {
    auto x = 5;                    // int
    auto y = 3.14;                 // double
    auto s = string("hello");      // string
    auto v = vector<int>{1,2,3};   // vector<int>

    cout << "x: " << x << ", y: " << y << endl;
}

/*
 * EXERCISE 2: auto with References and Pointers (10 min)
 */

void autoReferencesPointers() {
    int x = 42;

    auto a = x;        // int (copy)
    auto& b = x;       // int& (reference)
    auto* c = &x;      // int* (pointer)
    const auto& d = x; // const int&

    b = 100;  // Modifies x
    cout << "x after modification: " << x << endl;
}

/*
 * EXERCISE 3: decltype (10 min)
 */

void decltypeExample() {
    int x = 10;
    decltype(x) y = 20;  // y has same type as x (int)

    vector<int> v = {1, 2, 3};
    decltype(v.size()) count = v.size();  // size_t

    cout << "y: " << y << ", count: " << count << endl;
}

/*
 * EXERCISE 4: Trailing Return Types (10 min)
 */

// C++11: trailing return type
template<typename T, typename U>
auto add(T t, U u) -> decltype(t + u) {
    return t + u;
}

// C++14: automatic deduction
template<typename T, typename U>
auto multiply(T t, U u) {
    return t * u;
}

/*
 * EXERCISE 5: decltype(auto) (10 min)
 */

int globalValue = 42;

// Returns reference (decltype(auto) preserves reference)
decltype(auto) getGlobal() {
    return (globalValue);  // Returns int&
}

// Returns value (auto would also return value)
auto getGlobalCopy() {
    return globalValue;  // Returns int
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is auto?
 * A: Type deduction from initializer at compile time
 *
 * Q2: When does auto deduce reference?
 * A: Never! auto drops references. Use auto& or auto&& explicitly
 *
 * Q3: What is decltype?
 * A: Extracts type of expression, preserving references/const
 *
 * Q4: decltype(auto) vs auto?
 * A: decltype(auto): Preserves references/const exactly
 *    auto: Drops references, must add & explicitly
 *
 * Q5: AAA idiom?
 * A: Almost Always Auto - use auto for most variable declarations
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Auto with thrust::device_vector iterators
 * - Decltype for kernel return types
 * - Generic lambda parameters on GPU
 *
 * COMPILATION: g++ -std=c++17 01_auto_decltype.cpp -o auto
 * ==================================================================================================
 */

int main() {
    cout << "=== Auto and Decltype Practice ===" << endl;

    basicAuto();
    autoReferencesPointers();
    decltypeExample();

    cout << "add(5, 3.14): " << add(5, 3.14) << endl;
    cout << "multiply(2, 3.5): " << multiply(2, 3.5) << endl;

    getGlobal() = 100;  // Can modify because it's a reference
    cout << "Modified global: " << globalValue << endl;

    return 0;
}
