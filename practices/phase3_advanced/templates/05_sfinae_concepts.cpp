/*
 * ==================================================================================================
 * Exercise: SFINAE and Concepts in C++
 * ==================================================================================================
 * Difficulty: Advanced | Time: 50-60 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master SFINAE (Substitution Failure Is Not An Error)
 * 2. Understand enable_if and type traits
 * 3. Learn C++20 concepts
 * 4. Practice template constraints
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Type-safe generic GPU code
 * - Compile-time dispatch
 * - Constrained templates for correctness
 * - Modern C++ GPU libraries
 * ==================================================================================================
 */

#include <iostream>
#include <type_traits>
using namespace std;

/*
 * EXERCISE 1: Basic SFINAE with enable_if (15 min)
 */

// Only enable for integral types
template<typename T>
typename enable_if<is_integral<T>::value, T>::type
doubleValue(T x) {
    return x * 2;
}

// Only enable for floating point types
template<typename T>
typename enable_if<is_floating_point<T>::value, T>::type
halfValue(T x) {
    return x / 2;
}

/*
 * EXERCISE 2: SFINAE with Function Overloading (15 min)
 */

// Overload for pointer types
template<typename T>
enable_if_t<is_pointer<T>::value, void>
printType(T) {
    cout << "Pointer type" << endl;
}

// Overload for integral types
template<typename T>
enable_if_t<is_integral<T>::value, void>
printType(T) {
    cout << "Integral type" << endl;
}

// Overload for floating point
template<typename T>
enable_if_t<is_floating_point<T>::value, void>
printType(T) {
    cout << "Floating point type" << endl;
}

/*
 * EXERCISE 3: Tag Dispatch (10 min)
 */

// Algorithm for random access iterators
template<typename Iter>
void advanceImpl(Iter& it, int n, random_access_iterator_tag) {
    it += n;  // Efficient for random access
    cout << "Random access advance" << endl;
}

// Algorithm for other iterators
template<typename Iter>
void advanceImpl(Iter& it, int n, input_iterator_tag) {
    while (n--) ++it;  // Step by step
    cout << "Input iterator advance" << endl;
}

template<typename Iter>
void advance(Iter& it, int n) {
    advanceImpl(it, n, typename iterator_traits<Iter>::iterator_category());
}

/*
 * EXERCISE 4: C++20 Concepts (15 min)
 */

#if __cplusplus >= 202002L

// Define concept
template<typename T>
concept Numeric = is_arithmetic<T>::value;

// Use concept
template<Numeric T>
T multiply(T a, T b) {
    return a * b;
}

// Concept for containers
template<typename T>
concept Container = requires(T t) {
    { t.size() } -> same_as<size_t>;
    { t.begin() };
    { t.end() };
};

template<Container C>
void printSize(const C& container) {
    cout << "Size: " << container.size() << endl;
}

#endif

/*
 * EXERCISE 5: Type Traits (10 min)
 */

template<typename T>
void analyzeType() {
    cout << "Type analysis:" << endl;
    cout << "  Is integral: " << is_integral<T>::value << endl;
    cout << "  Is floating: " << is_floating_point<T>::value << endl;
    cout << "  Is pointer: " << is_pointer<T>::value << endl;
    cout << "  Is const: " << is_const<T>::value << endl;
    cout << "  Size: " << sizeof(T) << " bytes" << endl;
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is SFINAE?
 * A: Substitution Failure Is Not An Error - invalid template substitution
 *    causes deduction failure (not compile error), tries other overloads
 *
 * Q2: What does enable_if do?
 * A: Conditionally enable/disable template based on compile-time condition
 *
 * Q3: What are C++20 concepts?
 * A: Named requirements on template parameters, cleaner than SFINAE
 *
 * Q4: Why use SFINAE?
 * A: - Constrain templates to valid types
 *    - Provide different implementations based on type properties
 *    - Better error messages
 *
 * Q5: Tag dispatch vs SFINAE?
 * A: Tag dispatch: Runtime selection using type tags
 *    SFINAE: Compile-time selection through substitution failure
 *
 * Q6: What are type traits?
 * A: Compile-time type information: is_integral, is_pointer, etc.
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Constrain kernels to valid types (float, double, __half)
 * - Different code paths for coalesced vs non-coalesced access
 * - Compile-time architecture detection
 * - Type-safe kernel parameter validation
 *
 * Example:
 * template<typename T>
 * requires std::is_arithmetic_v<T>
 * __global__ void kernel(T* data, int n) { ... }
 *
 * COMPILATION: g++ -std=c++20 05_sfinae_concepts.cpp -o sfinae
 * ==================================================================================================
 */

int main() {
    cout << "=== SFINAE and Concepts Practice ===" << endl;

    cout << "Double: " << doubleValue(5) << endl;
    cout << "Half: " << halfValue(10.0) << endl;

    printType(42);
    printType(3.14);
    int* ptr = nullptr;
    printType(ptr);

    analyzeType<int>();
    analyzeType<const double*>();

#if __cplusplus >= 202002L
    cout << "Multiply: " << multiply(5, 3) << endl;
#else
    cout << "C++20 concepts not available" << endl;
#endif

    return 0;
}
