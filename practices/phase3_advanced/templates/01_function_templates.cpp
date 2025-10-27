/*
 * ==================================================================================================
 * Exercise: Function Templates in C++
 * ==================================================================================================
 * Difficulty: Intermediate/Advanced | Time: 45-55 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master function template syntax
 * 2. Understand template type deduction
 * 3. Learn explicit specialization
 * 4. Practice template parameters
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Generic GPU algorithms
 * - Type-safe kernel wrappers
 * - Zero-cost abstractions
 * - Compile-time optimizations
 * ==================================================================================================
 */

#include <iostream>
#include <vector>
#include <string>
using namespace std;

/*
 * THEORY: Function Templates
 *
 * TEMPLATE: Blueprint for creating functions/classes with different types
 * - Compile-time code generation
 * - No runtime overhead (zero-cost abstraction)
 * - Type-safe generic programming
 *
 * SYNTAX: template<typename T> ReturnType func(T param) { }
 *
 * TYPE DEDUCTION: Compiler infers template arguments
 * - func(5) → T = int
 * - func(3.14) → T = double
 * - Can explicitly specify: func<int>(3.14)
 * ==================================================================================================
 */

/*
 * EXERCISE 1: Basic Function Templates (15 min)
 */

// TODO 1.1: Create generic max function
template<typename T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

// Usage: maximum(5, 10) → int, maximum(3.14, 2.71) → double

// TODO 1.2: Generic swap function
template<typename T>
void swapValues(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

// TODO 1.3: Test with different types
void testBasicTemplates() {
    int x = 5, y = 10;
    swapValues(x, y);
    cout << "Swapped: " << x << ", " << y << endl;

    string s1 = "Hello", s2 = "World";
    swapValues(s1, s2);
    cout << "Swapped: " << s1 << ", " << s2 << endl;
}

/*
 * EXERCISE 2: Multiple Template Parameters (10 min)
 */

// TODO 2.1: Function with two type parameters
template<typename T, typename U>
void printPair(T first, U second) {
    cout << "(" << first << ", " << second << ")" << endl;
}

// TODO 2.2: Return type different from parameters
template<typename T1, typename T2>
auto add(T1 a, T2 b) -> decltype(a + b) {  // Return type deduction
    return a + b;
}

/*
 * EXERCISE 3: Template Specialization (15 min)
 */

// TODO 3.1: Generic print function
template<typename T>
void print(T value) {
    cout << value << endl;
}

// TODO 3.2: Specialized version for const char*
template<>
void print<const char*>(const char* value) {
    cout << "String: " << value << endl;
}

// TODO 3.3: Specialized for bool
template<>
void print<bool>(bool value) {
    cout << (value ? "true" : "false") << endl;
}

/*
 * EXERCISE 4: Function Template Overloading (10 min)
 */

// Generic version
template<typename T>
T sum(T a, T b) {
    return a + b;
}

// Overload for three parameters
template<typename T>
T sum(T a, T b, T c) {
    return a + b + c;
}

// Non-template overload (preferred if exact match)
int sum(int a, int b) {
    cout << "Non-template version called" << endl;
    return a + b;
}

/*
 * EXERCISE 5: Constrained Templates (10 min)
 */

// TODO 5.1: Template with static_assert
template<typename T>
T divide(T a, T b) {
    static_assert(is_arithmetic<T>::value, "T must be numeric");
    return a / b;
}

// TODO 5.2: SFINAE example (advanced)
template<typename T>
typename enable_if<is_integral<T>::value, T>::type
doubleValue(T x) {
    return x * 2;
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is a template?
 * A: Blueprint for generating code at compile time with different types
 *
 * Q2: Template vs function overloading?
 * A: Template: One definition, many types. Overloading: Multiple definitions
 *
 * Q3: What is template specialization?
 * A: Providing specific implementation for particular type(s)
 *
 * Q4: How does type deduction work?
 * A: Compiler infers template arguments from function call arguments
 *
 * Q5: Can templates have default arguments?
 * A: Yes! template<typename T = int> void func()
 *
 * Q6: What is template instantiation?
 * A: Compiler generates actual code for specific type(s)
 *
 * Q7: Runtime cost of templates?
 * A: Zero! All resolved at compile time. May increase binary size
 *
 * Q8: Can template functions be virtual?
 * A: No! Virtual requires runtime dispatch, templates are compile-time
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Generic kernel launchers for any type
 * - Type-safe host-device transfers
 * - Template-based vectorization
 * - Zero-overhead GPU abstractions
 *
 * Example: template<typename T> __global__ void kernel(T* data, int n)
 *
 * COMPILATION: g++ -std=c++17 01_function_templates.cpp -o ftemplates
 * ==================================================================================================
 */

int main() {
    cout << "=== Function Templates Practice ===" << endl;

    testBasicTemplates();

    cout << "Max: " << maximum(10, 20) << endl;
    cout << "Max: " << maximum(3.14, 2.71) << endl;

    printPair(42, "Answer");
    cout << "Sum: " << add(5, 3.14) << endl;

    return 0;
}
