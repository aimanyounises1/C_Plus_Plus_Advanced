/*
 * ==================================================================================================
 * Exercise: Template Specialization in C++
 * ==================================================================================================
 * Difficulty: Advanced | Time: 40-50 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master full template specialization
 * 2. Understand partial specialization
 * 3. Learn when to specialize
 * 4. Practice specialization best practices
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Optimize for specific types (float vs double on GPU)
 * - Special handling for pointer types
 * - Architecture-specific optimizations
 * - Type traits implementation
 * ==================================================================================================
 */

#include <iostream>
#include <cstring>
using namespace std;

/*
 * EXERCISE 1: Full Specialization (15 min)
 */

// Primary template
template<typename T>
class Container {
public:
    void print(const T& val) {
        cout << "Generic: " << val << endl;
    }
};

// Full specialization for const char*
template<>
class Container<const char*> {
public:
    void print(const char* val) {
        cout << "String specialization: " << val << endl;
    }
};

// Full specialization for bool
template<>
class Container<bool> {
public:
    void print(const bool& val) {
        cout << "Bool specialization: " << (val ? "true" : "false") << endl;
    }
};

/*
 * EXERCISE 2: Partial Specialization (15 min)
 */

// Primary template
template<typename T, typename U>
class Pair {
public:
    T first;
    U second;
    void info() { cout << "Generic pair" << endl; }
};

// Partial specialization: both same type
template<typename T>
class Pair<T, T> {
public:
    T first, second;
    void info() { cout << "Same type pair" << endl; }
};

// Partial specialization: pointer types
template<typename T, typename U>
class Pair<T*, U*> {
public:
    T* first;
    U* second;
    void info() { cout << "Pointer pair" << endl; }
};

/*
 * EXERCISE 3: Function Template Specialization (10 min)
 */

template<typename T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

// Specialization for const char*
template<>
const char* maximum<const char*>(const char* a, const char* b) {
    return (strcmp(a, b) > 0) ? a : b;
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is template specialization?
 * A: Providing custom implementation for specific type(s)
 *
 * Q2: Full vs partial specialization?
 * A: Full: All parameters specified. Partial: Some parameters still generic
 *
 * Q3: Can function templates be partially specialized?
 * A: No! Only full specialization or overloading
 *
 * Q4: When to use specialization?
 * A: - Optimize for specific types
 *    - Handle special cases (pointers, bool)
 *    - Provide different behavior for certain types
 *
 * Q5: Specialization vs overloading?
 * A: Specialization: Customize template. Overloading: Separate function
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Specialize for __half vs float vs double
 * - Optimize atomics for specific types
 * - Special handling for texture/surface types
 * - Architecture-specific kernel selection
 *
 * COMPILATION: g++ -std=c++17 03_template_specialization.cpp -o tspec
 * ==================================================================================================
 */

int main() {
    cout << "=== Template Specialization Practice ===" << endl;

    Container<int> c1;
    c1.print(42);

    Container<const char*> c2;
    c2.print("Hello");

    Container<bool> c3;
    c3.print(true);

    Pair<int, double> p1;
    p1.info();

    Pair<int, int> p2;
    p2.info();

    cout << "Max string: " << maximum("apple", "banana") << endl;

    return 0;
}
