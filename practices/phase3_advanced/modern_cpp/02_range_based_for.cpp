/*
 * ==================================================================================================
 * Exercise: Range-Based For Loops in Modern C++
 * ==================================================================================================
 * Difficulty: Beginner/Intermediate | Time: 30-40 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master range-based for loop syntax
 * 2. Understand structured bindings (C++17)
 * 3. Learn when to use const& vs &
 * 4. Practice with different container types
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Cleaner iteration over GPU data structures
 * - Modern C++ in CUDA host code
 * - Safer, more readable code
 * ==================================================================================================
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>
using namespace std;

/*
 * EXERCISE 1: Basic Range-Based For (10 min)
 */

void basicRangeFor() {
    vector<int> numbers = {1, 2, 3, 4, 5};

    // Read-only (copy)
    for (int num : numbers) {
        cout << num << " ";
    }
    cout << endl;

    // Read-only (const reference - efficient)
    for (const auto& num : numbers) {
        cout << num << " ";
    }
    cout << endl;

    // Modify elements
    for (auto& num : numbers) {
        num *= 2;
    }
}

/*
 * EXERCISE 2: Structured Bindings (C++17) (10 min)
 */

void structuredBindings() {
    map<string, int> ages = {{"Alice", 30}, {"Bob", 25}, {"Charlie", 35}};

    // C++17: structured bindings
    for (const auto& [name, age] : ages) {
        cout << name << " is " << age << " years old" << endl;
    }

    // Before C++17:
    for (const auto& pair : ages) {
        cout << pair.first << " is " << pair.second << " years old" << endl;
    }
}

/*
 * EXERCISE 3: Range-Based For with Arrays (5 min)
 */

void arrayIteration() {
    int arr[] = {10, 20, 30, 40, 50};

    for (int val : arr) {
        cout << val << " ";
    }
    cout << endl;
}

/*
 * EXERCISE 4: Custom Types (10 min)
 */

struct Point {
    int x, y;
};

void customTypeIteration() {
    vector<Point> points = {{1,2}, {3,4}, {5,6}};

    for (const auto& p : points) {
        cout << "(" << p.x << ", " << p.y << ") ";
    }
    cout << endl;

    // C++17: Structured bindings with custom types
    for (const auto& [x, y] : points) {
        cout << "(" << x << ", " << y << ") ";
    }
    cout << endl;
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is range-based for loop?
 * A: for (element : container) - iterates over all elements
 *
 * Q2: When to use const auto& vs auto&?
 * A: const auto&: Read-only (most common)
 *    auto&: Need to modify elements
 *    auto: Simple types or when need copy
 *
 * Q3: What are structured bindings?
 * A: C++17: Unpack tuple/pair/struct into separate variables
 *    for (auto [x, y] : pairs)
 *
 * Q4: Can range-for work with C arrays?
 * A: Yes! But only with arrays of known size, not pointers
 *
 * Q5: Performance of range-for vs traditional?
 * A: Same! Compiler generates equivalent code
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Iterate over thrust::host_vector
 * - Process GPU query results
 * - Modern C++ in CUDA host code
 *
 * COMPILATION: g++ -std=c++17 02_range_based_for.cpp -o rangefor
 * ==================================================================================================
 */

int main() {
    cout << "=== Range-Based For Loops Practice ===" << endl;

    basicRangeFor();
    structuredBindings();
    arrayIteration();
    customTypeIteration();

    return 0;
}
