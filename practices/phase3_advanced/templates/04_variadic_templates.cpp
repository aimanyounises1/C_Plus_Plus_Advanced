/*
 * ==================================================================================================
 * Exercise: Variadic Templates in C++
 * ==================================================================================================
 * Difficulty: Advanced | Time: 45-55 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master parameter packs
 * 2. Understand template recursion
 * 3. Learn fold expressions (C++17)
 * 4. Practice variadic template patterns
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Variable argument kernel launchers
 * - Generic tuple-like structures
 * - Perfect forwarding for GPU functions
 * - Type-safe printf alternatives
 * ==================================================================================================
 */

#include <iostream>
using namespace std;

/*
 * EXERCISE 1: Basic Variadic Template (15 min)
 */

// Base case: no arguments
void print() {
    cout << endl;
}

// Recursive case: at least one argument
template<typename T, typename... Args>
void print(T first, Args... rest) {
    cout << first << " ";
    print(rest...);  // Recursive call with remaining args
}

/*
 * EXERCISE 2: Sizeof... Operator (10 min)
 */

template<typename... Args>
void printCount(Args... args) {
    cout << "Number of arguments: " << sizeof...(Args) << endl;
    cout << "Values: ";
    print(args...);
}

/*
 * EXERCISE 3: Fold Expressions (C++17) (15 min)
 */

// Sum using fold expression
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);  // Unary right fold: (a + (b + (c + d)))
}

// Logical AND using fold
template<typename... Args>
bool all(Args... args) {
    return (args && ...);
}

// Print with comma using fold
template<typename... Args>
void printWithComma(Args... args) {
    ((cout << args << ", "), ...);
    cout << endl;
}

/*
 * EXERCISE 4: Variadic Class Template (15 min)
 */

// Simple tuple implementation
template<typename... Types>
class Tuple;

// Empty tuple
template<>
class Tuple<> {
public:
    void print() const { cout << "()"; }
};

// Non-empty tuple
template<typename T, typename... Rest>
class Tuple<T, Rest...> {
private:
    T value;
    Tuple<Rest...> rest;

public:
    Tuple(T v, Rest... r) : value(v), rest(r...) {}

    void print() const {
        cout << "(" << value;
        if constexpr(sizeof...(Rest) > 0) {
            cout << ", ";
            rest.print();
        } else {
            cout << ")";
        }
    }
};

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is a parameter pack?
 * A: Template parameter representing zero or more parameters
 *
 * Q2: How to expand parameter pack?
 * A: Use ... operator: func(args...)
 *
 * Q3: What is fold expression?
 * A: C++17 feature: Reduces parameter pack using binary operator
 *    Forms: (... op pack), (pack op ...), (init op ... op pack)
 *
 * Q4: How does variadic template recursion work?
 * A: Base case for zero args, recursive case processes first arg and recurses
 *
 * Q5: Difference between sizeof...(Args) and sizeof(Args...)?
 * A: sizeof...: Number of arguments. sizeof: Size of pack expansion (error!)
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Kernel launcher with variable arguments
 * - Generic tuple for kernel parameters
 * - Perfect forwarding to device functions
 * - Type-safe logging/debugging utilities
 *
 * Example: launch_kernel(grid, block, kernel, arg1, arg2, arg3, ...)
 *
 * COMPILATION: g++ -std=c++17 04_variadic_templates.cpp -o variadic
 * ==================================================================================================
 */

int main() {
    cout << "=== Variadic Templates Practice ===" << endl;

    print(1, 2.5, "hello", 'c');

    printCount(10, 20, 30, 40, 50);

    cout << "Sum: " << sum(1, 2, 3, 4, 5) << endl;
    cout << "All true: " << all(true, true, true) << endl;
    cout << "All true: " << all(true, false, true) << endl;

    printWithComma(1, 2, 3, 4, 5);

    Tuple<int, double, const char*> t(42, 3.14, "test");
    t.print();
    cout << endl;

    return 0;
}
