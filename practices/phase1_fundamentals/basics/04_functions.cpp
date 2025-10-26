/*
 * Exercise: Functions
 * Difficulty: Beginner
 * Time: 40-50 minutes
 * Topics: Declaration, definition, parameters, return values, overloading, recursion
 *
 * LEARNING OBJECTIVES:
 * - Understand function declaration vs definition
 * - Master parameter passing (value, reference, pointer)
 * - Learn function overloading
 * - Practice default parameters
 * - Understand inline functions
 * - Learn basic recursion
 *
 * INTERVIEW RELEVANCE:
 * - Function design is fundamental to clean code
 * - Pass-by-reference vs pass-by-value is frequently asked
 * - Recursion problems are common in technical interviews
 * - Understanding stack frames is critical for debugging
 * - Inline functions relate to GPU device functions (__device__)
 */

#include <iostream>
#include <string>
#include <cmath>
#include <vector>

// Forward declarations (tells compiler these functions exist)
// TODO 1.1: Declare functions here


// ========================================================================
// EXERCISE 1: Basic Function Declaration and Definition (5 min)
// ========================================================================

// TODO 1.2: Define a simple function that prints "Hello, Functions!"
// void printHello() { ... }


// TODO 1.3: Define a function that takes an integer and returns its square
// int square(int n) { ... }


// TODO 1.4: Define a function that takes two integers and returns their sum
// int add(int a, int b) { ... }


// ========================================================================
// EXERCISE 2: Return Values (5 min)
// ========================================================================

// TODO 2.1: Function that returns a boolean
// Check if a number is even
// bool isEven(int n) { ... }


// TODO 2.2: Function that returns a double
// Calculate circle area given radius (π * r²)
// double circleArea(double radius) { ... }


// TODO 2.3: Function that returns a string
// Get greeting message for a name
// std::string getGreeting(std::string name) { ... }


// TODO 2.4: Function with early return
// Get absolute value of a number
// int absoluteValue(int n) { ... }


// ========================================================================
// EXERCISE 3: Pass by Value vs Pass by Reference (10 min) - IMPORTANT!
// ========================================================================

// TODO 3.1: Pass by value (makes a copy - changes don't affect original)
// void incrementByValue(int x) {
//     x++;  // Only changes the local copy
// }


// TODO 3.2: Pass by reference (no copy - changes affect original)
// void incrementByReference(int& x) {
//     x++;  // Changes the original variable
// }


// TODO 3.3: Pass by const reference (efficient for large objects, but read-only)
// void printVector(const std::vector<int>& vec) {
//     // Can read but not modify
// }


// TODO 3.4: Pass by pointer (similar to reference, but can be null)
// void incrementByPointer(int* x) {
//     if (x != nullptr) {
//         (*x)++;
//     }
// }


// ========================================================================
// EXERCISE 4: Function Overloading (5 min)
// ========================================================================

// TODO 4.1: Overload a function to work with different types
// Same function name, different parameter types

// int max(int a, int b) { ... }
// double max(double a, double b) { ... }
// long max(long a, long b) { ... }


// TODO 4.2: Overload based on number of parameters
// int multiply(int a, int b) { ... }
// int multiply(int a, int b, int c) { ... }


// ========================================================================
// EXERCISE 5: Default Parameters (5 min)
// ========================================================================

// TODO 5.1: Function with default parameter
// void printMessage(std::string message, int times = 1) {
//     // If times not provided, defaults to 1
// }


// TODO 5.2: Multiple default parameters (default from right to left)
// double calculatePrice(double base, double tax = 0.1, double discount = 0.0) {
//     // tax defaults to 10%, discount to 0%
// }


// ========================================================================
// EXERCISE 6: Inline Functions (5 min)
// ========================================================================

// TODO 6.1: Inline function (suggestion to compiler to inline the code)
// Useful for small, frequently called functions
// inline int min(int a, int b) {
//     return (a < b) ? a : b;
// }


// TODO 6.2: When to use inline
// - Small functions (1-3 lines)
// - Called frequently
// - Performance-critical code
// NOTE: In CUDA, __device__ __forceinline__ is used for device functions


// ========================================================================
// EXERCISE 7: Recursion Basics (10 min)
// ========================================================================

// TODO 7.1: Factorial using recursion
// factorial(5) = 5 * 4 * 3 * 2 * 1 = 120
// Base case: factorial(0) = 1
// Recursive case: factorial(n) = n * factorial(n-1)
// unsigned long long factorial(int n) { ... }


// TODO 7.2: Fibonacci using recursion
// fib(0) = 0, fib(1) = 1
// fib(n) = fib(n-1) + fib(n-2)
// int fibonacci(int n) { ... }


// TODO 7.3: Sum of array using recursion
// int sumArray(int arr[], int size) {
//     if (size == 0) return 0;
//     return arr[size-1] + sumArray(arr, size-1);
// }


// ========================================================================
// EXERCISE 8: Practical Functions (10 min)
// ========================================================================

// TODO 8.1: Temperature conversion
// double celsiusToFahrenheit(double celsius) { ... }
// double fahrenheitToCelsius(double fahrenheit) { ... }


// TODO 8.2: Array operations
// int findMax(int arr[], int size) { ... }
// int findMin(int arr[], int size) { ... }
// double findAverage(int arr[], int size) { ... }


// TODO 8.3: String operations
// bool isPalindrome(std::string str) { ... }
// std::string reverseString(std::string str) { ... }


// TODO 8.4: Math utilities
// bool isPrime(int n) { ... }
// int gcd(int a, int b) { ... } // Greatest common divisor using Euclidean algorithm


// ========================================================================
// MAIN FUNCTION
// ========================================================================

int main() {
    std::cout << "=== Functions Exercises ===\n\n";

    // ========================================================================
    // EXERCISE 1: Basic Functions
    // ========================================================================
    std::cout << "Exercise 1: Basic Functions\n";
    std::cout << "----------------------------\n";

    // TODO: Call printHello()


    // TODO: Call square(5) and print result


    // TODO: Call add(10, 20) and print result


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 2: Return Values
    // ========================================================================
    std::cout << "Exercise 2: Return Values\n";
    std::cout << "-------------------------\n";

    // TODO: Test isEven with several numbers


    // TODO: Calculate and print area of circle with radius 5.0


    // TODO: Get greeting for "Alice"


    // TODO: Test absoluteValue with positive and negative numbers


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 3: Pass by Value vs Reference
    // ========================================================================
    std::cout << "Exercise 3: Pass by Value vs Reference\n";
    std::cout << "---------------------------------------\n";

    // TODO: Demonstrate pass by value
    int val = 10;
    std::cout << "Before incrementByValue: " << val << "\n";
    // Call incrementByValue(val)
    std::cout << "After incrementByValue: " << val << "\n";  // Still 10!

    // TODO: Demonstrate pass by reference
    int ref = 10;
    std::cout << "Before incrementByReference: " << ref << "\n";
    // Call incrementByReference(ref)
    std::cout << "After incrementByReference: " << ref << "\n";  // Now 11!

    // TODO: Demonstrate pass by pointer
    int ptr = 10;
    std::cout << "Before incrementByPointer: " << ptr << "\n";
    // Call incrementByPointer(&ptr)
    std::cout << "After incrementByPointer: " << ptr << "\n";  // Now 11!

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 4: Function Overloading
    // ========================================================================
    std::cout << "Exercise 4: Function Overloading\n";
    std::cout << "---------------------------------\n";

    // TODO: Call overloaded max functions
    // std::cout << "max(5, 10) = " << max(5, 10) << "\n";
    // std::cout << "max(5.5, 10.2) = " << max(5.5, 10.2) << "\n";

    // TODO: Call overloaded multiply functions
    // std::cout << "multiply(3, 4) = " << multiply(3, 4) << "\n";
    // std::cout << "multiply(3, 4, 5) = " << multiply(3, 4, 5) << "\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 5: Default Parameters
    // ========================================================================
    std::cout << "Exercise 5: Default Parameters\n";
    std::cout << "-------------------------------\n";

    // TODO: Call printMessage with and without second parameter
    // printMessage("Hello");        // Uses default times=1
    // printMessage("Hello", 3);     // Prints 3 times

    // TODO: Call calculatePrice with different number of arguments
    // calculatePrice(100.0);              // Uses default tax and discount
    // calculatePrice(100.0, 0.15);        // Custom tax, default discount
    // calculatePrice(100.0, 0.15, 0.1);   // Custom tax and discount

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 6: Inline Functions
    // ========================================================================
    std::cout << "Exercise 6: Inline Functions\n";
    std::cout << "-----------------------------\n";

    // TODO: Test min function
    // std::cout << "min(5, 10) = " << min(5, 10) << "\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 7: Recursion
    // ========================================================================
    std::cout << "Exercise 7: Recursion\n";
    std::cout << "---------------------\n";

    // TODO: Test factorial
    // std::cout << "factorial(5) = " << factorial(5) << "\n";
    // std::cout << "factorial(10) = " << factorial(10) << "\n";

    // TODO: Test fibonacci
    // std::cout << "fibonacci(10) = " << fibonacci(10) << "\n";

    // TODO: Test sumArray
    // int numbers[] = {1, 2, 3, 4, 5};
    // std::cout << "Sum of array = " << sumArray(numbers, 5) << "\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 8: Practical Functions
    // ========================================================================
    std::cout << "Exercise 8: Practical Functions\n";
    std::cout << "--------------------------------\n";

    // TODO: Temperature conversions
    // std::cout << "25°C = " << celsiusToFahrenheit(25) << "°F\n";

    // TODO: Array operations
    // int data[] = {23, 45, 12, 67, 34};
    // std::cout << "Max: " << findMax(data, 5) << "\n";
    // std::cout << "Min: " << findMin(data, 5) << "\n";
    // std::cout << "Avg: " << findAverage(data, 5) << "\n";

    // TODO: String operations
    // std::cout << "Is 'racecar' a palindrome? " << isPalindrome("racecar") << "\n";

    // TODO: Math utilities
    // std::cout << "Is 17 prime? " << isPrime(17) << "\n";
    // std::cout << "GCD of 48 and 18: " << gcd(48, 18) << "\n";

    std::cout << "\n";

    // ========================================================================
    // CHALLENGE EXERCISES
    // ========================================================================
    std::cout << "Challenge Exercises\n";
    std::cout << "-------------------\n";

    // CHALLENGE 1: Implement power function using recursion
    // power(2, 3) = 8 (2 * 2 * 2)
    // double power(double base, int exponent) { ... }


    // CHALLENGE 2: Implement binary search (iterative, not recursive)
    // Return index of target, or -1 if not found
    // int binarySearch(int arr[], int size, int target) { ... }


    // CHALLENGE 3: Swap two integers using a function
    // void swap(int& a, int& b) { ... }


    // CHALLENGE 4: Function that returns multiple values using references
    // void getMinMax(int arr[], int size, int& min, int& max) { ... }


    // CHALLENGE 5: Implement Euclidean algorithm for GCD
    // int gcd(int a, int b) {
    //     if (b == 0) return a;
    //     return gcd(b, a % b);
    // }


    std::cout << "\n";

    return 0;
}

// ========================================================================
// COMMON INTERVIEW QUESTIONS
// ========================================================================
/*
 * Q1: What's the difference between pass by value and pass by reference?
 * A: Pass by value creates a copy of the argument. Changes to the parameter
 *    don't affect the original. This is safe but can be slow for large objects.
 *
 *    Pass by reference creates an alias to the original. Changes to the
 *    parameter affect the original. More efficient for large objects.
 *
 *    Example:
 *    void func1(int x) { x++; }      // Pass by value
 *    void func2(int& x) { x++; }     // Pass by reference
 *
 *    int val = 5;
 *    func1(val);  // val is still 5
 *    func2(val);  // val is now 6
 *
 * Q2: When should you use const with function parameters?
 * A: Use const when:
 *    - You want to pass by reference for efficiency
 *    - But don't want to allow modifications
 *    - Communicates intent: "this function won't change this parameter"
 *
 *    void print(const std::string& str) {
 *        // Can read str, but can't modify it
 *        // Efficient (no copy) and safe (can't change)
 *    }
 *
 * Q3: What is function overloading?
 * A: Multiple functions with the same name but different parameter types
 *    or number of parameters. The compiler chooses the right one based on
 *    the arguments you pass.
 *
 *    int add(int a, int b);           // For integers
 *    double add(double a, double b);  // For doubles
 *
 *    Overloading is NOT allowed based on return type alone!
 *
 * Q4: What is the difference between inline and regular functions?
 * A: inline is a suggestion to the compiler to insert the function's code
 *    directly at the call site instead of making a function call.
 *
 *    Benefits:
 *    - Eliminates function call overhead
 *    - Enables better optimization
 *
 *    Drawbacks:
 *    - Increases code size (code is duplicated at each call site)
 *    - Compiler may ignore the inline suggestion
 *
 *    Use for: Small, frequently called functions
 *    Don't use for: Large functions, recursive functions
 *
 * Q5: What is recursion and when should you use it?
 * A: Recursion is when a function calls itself.
 *
 *    Requirements:
 *    - Base case (when to stop)
 *    - Recursive case (call itself with simpler input)
 *
 *    Good for:
 *    - Tree/graph traversal
 *    - Divide and conquer algorithms
 *    - Problems with recursive structure (factorial, Fibonacci)
 *
 *    Bad for:
 *    - Simple iterations (use loops instead)
 *    - Deep recursion (stack overflow risk)
 *    - Performance-critical code (function call overhead)
 *
 * Q6: What is a stack overflow?
 * A: When recursion is too deep or a function uses too much stack space.
 *    Each function call adds a frame to the call stack. If the stack
 *    runs out of memory, the program crashes.
 *
 *    Example that causes stack overflow:
 *    int factorial(int n) {
 *        return n * factorial(n-1);  // No base case!
 *    }
 *
 *    Fix: Add base case
 *    int factorial(int n) {
 *        if (n <= 1) return 1;       // Base case
 *        return n * factorial(n-1);
 *    }
 *
 * Q7: What's the difference between parameters and arguments?
 * A: Parameters are the variables in the function declaration.
 *    Arguments are the actual values passed when calling the function.
 *
 *    void greet(std::string name) {  // 'name' is a parameter
 *        cout << "Hello, " << name;
 *    }
 *
 *    greet("Alice");  // "Alice" is an argument
 *
 * Q8: Can you have a function with no return type?
 * A: Yes, use void:
 *    void printMessage() {
 *        cout << "Hello\n";
 *        // No return statement needed
 *    }
 *
 *    But every function technically returns something (even if void).
 */

/*
 * FUNCTIONS IN GPU PROGRAMMING:
 * ==============================
 *
 * 1. Device Functions (__device__):
 *    __device__ float square(float x) {
 *        return x * x;
 *    }
 *    - Called from GPU, runs on GPU
 *    - Often inlined for performance
 *
 * 2. Host Functions (__host__):
 *    __host__ void setupData() { ... }
 *    - Called from CPU, runs on CPU
 *    - Default if no qualifier specified
 *
 * 3. Host + Device (__host__ __device__):
 *    __host__ __device__ float add(float a, float b) {
 *        return a + b;
 *    }
 *    - Can be called from both CPU and GPU
 *    - Compiled for both architectures
 *
 * 4. Inline Device Functions:
 *    __device__ __forceinline__ float fastMul(float a, float b) {
 *        return a * b;
 *    }
 *    - Forces inlining for performance
 *    - Eliminates function call overhead
 *
 * 5. No Recursion in Old CUDA:
 *    - Recursion not supported in CUDA compute capability < 2.0
 *    - Now supported but use with caution (limited stack)
 *
 * 6. Pass by Reference:
 *    - Can't use C++ references in device functions in older CUDA
 *    - Use pointers instead
 *    - Modern CUDA (11+) supports references
 *
 * COMPILATION:
 * ============
 * g++ -std=c++17 -Wall -Wextra 04_functions.cpp -o functions
 * ./functions
 *
 * LEARNING CHECKLIST:
 * ===================
 * ☐ Can declare and define functions
 * ☐ Understand pass by value vs pass by reference
 * ☐ Know when to use const references
 * ☐ Can implement function overloading
 * ☐ Understand default parameters
 * ☐ Know when to use inline functions
 * ☐ Can write recursive functions with base cases
 * ☐ Understand stack frames and stack overflow
 * ☐ Can design clean function interfaces
 *
 * NEXT STEPS:
 * ===========
 * - Move to 05_arrays_strings.cpp
 * - Practice recursive problems on LeetCode
 * - Study function optimization techniques
 * - Learn about tail recursion
 * - Understand CUDA device function qualifiers
 */
