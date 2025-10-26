/*
 * Exercise: Control Flow
 * Difficulty: Beginner
 * Time: 40-50 minutes
 * Topics: if-else, switch, for, while, do-while, break, continue, nested loops
 *
 * LEARNING OBJECTIVES:
 * - Master conditional statements (if-else, switch)
 * - Understand all loop types and when to use each
 * - Learn break and continue keywords
 * - Practice nested loops and loop optimization
 * - Understand control flow impact on performance
 *
 * INTERVIEW RELEVANCE:
 * - Control flow questions are fundamental in interviews
 * - Loop optimization is critical for high-performance code
 * - Branch divergence is a major GPU performance issue (Nvidia cares!)
 * - Understanding time complexity comes from analyzing loops
 * - Nested loop problems are common in technical interviews
 */

#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    std::cout << "=== Control Flow Exercises ===\n\n";

    // ========================================================================
    // EXERCISE 1: If-Else Statements (5 min)
    // ========================================================================
    std::cout << "Exercise 1: If-Else Statements\n";
    std::cout << "-------------------------------\n";

    // TODO 1.1: Simple if statement
    // Check if a number is positive
    int number = 10;


    // TODO 1.2: if-else statement
    // Check if a number is even or odd
    // Hint: Use modulo operator (%)


    // TODO 1.3: if-else if-else chain
    // Grade classification: A (90+), B (80-89), C (70-79), D (60-69), F (<60)
    int score = 85;




    // TODO 1.4: Nested if statements
    // Check if a year is a leap year
    // Rules: divisible by 4, but not by 100, unless also divisible by 400
    int year = 2024;




    std::cout << "\n";

    // ========================================================================
    // EXERCISE 2: Ternary Operator (5 min)
    // ========================================================================
    std::cout << "Exercise 2: Ternary Operator\n";
    std::cout << "-----------------------------\n";

    // TODO 2.1: Basic ternary operator
    // Find the maximum of two numbers
    int a = 15, b = 20;
    // int max = condition ? value_if_true : value_if_false;


    // TODO 2.2: Nested ternary (be careful - can get hard to read!)
    // Find the maximum of three numbers
    int x = 10, y = 25, z = 15;


    // TODO 2.3: Ternary for string output
    bool isLoggedIn = true;
    // Print "Welcome back" if logged in, else "Please log in"


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 3: Switch Statements (5 min)
    // ========================================================================
    std::cout << "Exercise 3: Switch Statements\n";
    std::cout << "------------------------------\n";

    // TODO 3.1: Basic switch statement
    // Print the day of the week (1 = Monday, 7 = Sunday)
    int day = 3;




    // TODO 3.2: Switch with fall-through
    // Classify months by number of days
    int month = 2;  // February
    // Months with 31 days: 1,3,5,7,8,10,12
    // Months with 30 days: 4,6,9,11
    // February: 28 (or 29 in leap year)




    // TODO 3.3: Switch vs if-else
    // When to use switch: discrete values, better readability
    // When to use if-else: ranges, complex conditions
    char grade = 'B';


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 4: For Loops (5 min)
    // ========================================================================
    std::cout << "Exercise 4: For Loops\n";
    std::cout << "----------------------\n";

    // TODO 4.1: Basic for loop
    // Print numbers 1 to 10


    // TODO 4.2: For loop with step
    // Print even numbers from 2 to 20


    // TODO 4.3: Reverse for loop
    // Countdown from 10 to 1


    // TODO 4.4: For loop to calculate sum
    // Sum of numbers 1 to 100
    int sum = 0;


    // TODO 4.5: For loop with array
    int arr[] = {5, 10, 15, 20, 25};
    int size = sizeof(arr) / sizeof(arr[0]);
    // Print all elements


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 5: While Loops (5 min)
    // ========================================================================
    std::cout << "Exercise 5: While Loops\n";
    std::cout << "-----------------------\n";

    // TODO 5.1: Basic while loop
    // Print numbers 1 to 5
    int i = 1;


    // TODO 5.2: While loop with condition
    // Keep dividing by 2 until result is less than 1
    double value = 100.0;


    // TODO 5.3: While loop for input validation
    // Simulate: keep asking for positive number (use a preset value to avoid actual input)
    int userInput = -5;  // Pretend user entered this
    // In real code, you'd use cin


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 6: Do-While Loops (5 min)
    // ========================================================================
    std::cout << "Exercise 6: Do-While Loops\n";
    std::cout << "--------------------------\n";

    // TODO 6.1: Basic do-while
    // Print numbers 1 to 5
    int j = 1;


    // TODO 6.2: Difference between while and do-while
    // while checks condition first, do-while checks after
    int k = 10;
    while (k < 5) {  // This won't execute
        std::cout << "While: " << k << "\n";
    }
    // Now try do-while with same condition


    // TODO 6.3: Do-while for menu systems
    // Common use case: run at least once
    int choice = 0;
    // Simulate a menu (do-while ensures it runs at least once)


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 7: Break and Continue (5 min)
    // ========================================================================
    std::cout << "Exercise 7: Break and Continue\n";
    std::cout << "-------------------------------\n";

    // TODO 7.1: Break statement
    // Find the first number divisible by 7 between 1 and 100


    // TODO 7.2: Continue statement
    // Print numbers 1 to 10, but skip 5
    std::cout << "Numbers (skipping 5): ";


    // TODO 7.3: Break in nested loop
    // Search for a value in 2D array
    int matrix[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int target = 5;
    bool found = false;


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 8: Nested Loops (10 min)
    // ========================================================================
    std::cout << "Exercise 8: Nested Loops\n";
    std::cout << "------------------------\n";

    // TODO 8.1: Print a multiplication table (1-5)
    std::cout << "Multiplication Table:\n";


    // TODO 8.2: Print a triangle pattern
    // *
    // **
    // ***
    // ****
    // *****
    std::cout << "\nTriangle Pattern:\n";


    // TODO 8.3: Print a number pyramid
    //     1
    //    121
    //   12321
    //  1234321
    std::cout << "\nNumber Pyramid:\n";


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 9: Loop Performance (10 min) - IMPORTANT FOR NVIDIA!
    // ========================================================================
    std::cout << "Exercise 9: Loop Performance\n";
    std::cout << "----------------------------\n";

    // TODO 9.1: Loop unrolling concept
    // Compare: regular loop vs partially unrolled loop
    // Unrolling reduces loop overhead
    int data[100];
    for (int i = 0; i < 100; i++) data[i] = i;

    // Regular loop
    sum = 0;
    for (int i = 0; i < 100; i++) {
        sum += data[i];
    }
    std::cout << "Regular loop sum: " << sum << "\n";

    // Unrolled loop (process 4 elements per iteration)
    // TODO: Implement unrolled version


    // TODO 9.2: Loop invariant code motion
    // Move calculations that don't change out of the loop
    int n = 100;
    // Bad: recalculates n*2 every iteration
    // for (int i = 0; i < n*2; i++) { ... }

    // Good: calculate once
    // int limit = n*2;
    // for (int i = 0; i < limit; i++) { ... }


    // TODO 9.3: Branch prediction
    // Sorted data is faster to process (predictable branches)
    // This matters for CPU and especially for GPU (branch divergence!)


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 10: Practical Applications (10 min)
    // ========================================================================
    std::cout << "Exercise 10: Practical Applications\n";
    std::cout << "------------------------------------\n";

    // TODO 10.1: Find the largest element in an array
    int numbers[] = {23, 45, 12, 67, 34, 89, 11};
    int arraySize = sizeof(numbers) / sizeof(numbers[0]);


    // TODO 10.2: Count occurrences of a value
    int values[] = {1, 2, 3, 2, 4, 2, 5, 2};
    int searchValue = 2;


    // TODO 10.3: Reverse an array
    int original[] = {1, 2, 3, 4, 5};
    int len = sizeof(original) / sizeof(original[0]);


    // TODO 10.4: Check if array is sorted
    int sorted[] = {1, 2, 3, 4, 5};
    int sortedSize = sizeof(sorted) / sizeof(sorted[0]);


    // TODO 10.5: Find prime numbers up to N
    int N = 50;
    std::cout << "Prime numbers up to " << N << ": ";
    // Hint: A number is prime if it's only divisible by 1 and itself


    std::cout << "\n\n";

    // ========================================================================
    // CHALLENGE EXERCISES (Optional - 15 min)
    // ========================================================================
    std::cout << "Challenge Exercises\n";
    std::cout << "-------------------\n";

    // CHALLENGE 1: FizzBuzz
    // Print numbers 1 to 100, but:
    // - Print "Fizz" for multiples of 3
    // - Print "Buzz" for multiples of 5
    // - Print "FizzBuzz" for multiples of both
    std::cout << "FizzBuzz (1-30):\n";


    // CHALLENGE 2: Find all pairs in array that sum to target
    // Example: arr = [1, 2, 3, 4, 5], target = 5
    // Pairs: (1,4), (2,3)
    int pairArr[] = {1, 2, 3, 4, 5};
    int pairSize = sizeof(pairArr) / sizeof(pairArr[0]);
    int targetSum = 6;
    std::cout << "\nPairs that sum to " << targetSum << ":\n";


    // CHALLENGE 3: Print a diamond pattern
    // Example for n=5:
    //     *
    //    ***
    //   *****
    //    ***
    //     *
    std::cout << "\nDiamond Pattern:\n";


    // CHALLENGE 4: Implement binary search
    // Given a sorted array, find an element using binary search
    int sortedArr[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    int searchSize = sizeof(sortedArr) / sizeof(sortedArr[0]);
    int searchTarget = 11;
    // Binary search: O(log n) - much faster than linear O(n)


    // CHALLENGE 5: Generate Fibonacci sequence
    // 0, 1, 1, 2, 3, 5, 8, 13, 21, ...
    std::cout << "\nFibonacci sequence (first 15 numbers):\n";


    std::cout << "\n";

    // ========================================================================
    // COMMON INTERVIEW QUESTIONS
    // ========================================================================
    /*
     * Q1: What's the difference between while and do-while?
     * A: while checks the condition before executing the loop body,
     *    so it may never execute. do-while checks after, so it always
     *    executes at least once. Use do-while for menus or input validation.
     *
     * Q2: When should you use a switch vs if-else?
     * A: Use switch for:
     *    - Checking discrete values (not ranges)
     *    - Better readability with many conditions
     *    - Compiler can optimize better (jump tables)
     *    Use if-else for:
     *    - Range checks (if x > 10 && x < 20)
     *    - Complex boolean conditions
     *    - Different types of comparisons
     *
     * Q3: What is loop unrolling and why does it help?
     * A: Loop unrolling processes multiple iterations in a single loop iteration.
     *    Benefits:
     *    - Reduces loop overhead (fewer increments, comparisons, jumps)
     *    - Enables better CPU pipelining and ILP (instruction-level parallelism)
     *    - Critical for GPU programming to reduce control flow overhead
     *    Example: Instead of processing 1 element per iteration, process 4
     *
     * Q4: What is branch divergence and why is it bad for GPUs?
     * A: Branch divergence occurs when threads in a warp take different
     *    execution paths (e.g., some threads do if, others do else).
     *    On GPUs:
     *    - All threads in a warp execute in lockstep (SIMT model)
     *    - When paths diverge, both paths must be executed serially
     *    - Threads not on the active path are masked out (wasted cycles)
     *    - Can cut performance in half or worse!
     *    Solution: Reorganize data so threads in same warp take same path
     *
     * Q5: What's the time complexity of nested loops?
     * A: Depends on how many times each loop runs:
     *    - Two loops, both run n times: O(n²)
     *    - Three nested loops of n: O(n³)
     *    - Outer loop n times, inner loop m times: O(n*m)
     *    Always analyze the worst case!
     *
     * Q6: How do you optimize a loop?
     * A: Several techniques:
     *    1. Loop invariant code motion (move constant calculations outside)
     *    2. Loop unrolling (process multiple elements per iteration)
     *    3. Reduce memory access (cache frequently used values)
     *    4. Minimize branching (especially in inner loops)
     *    5. Use const/restrict to help compiler optimize
     *    6. Consider vectorization (SIMD on CPU, parallel on GPU)
     *
     * Q7: What's the difference between break and continue?
     * A: break exits the loop entirely
     *    continue skips the rest of the current iteration and moves to the next
     *    Example:
     *    for (int i = 0; i < 10; i++) {
     *        if (i == 3) continue; // Skip 3, but keep going
     *        if (i == 7) break;    // Stop at 7
     *        cout << i; // Prints: 0 1 2 4 5 6
     *    }
     *
     * Q8: How do you avoid infinite loops?
     * A: 1. Always ensure loop condition will eventually become false
     *    2. Make sure loop variable is modified inside the loop
     *    3. Be careful with floating-point comparisons (use epsilon)
     *    4. Use for loops when iteration count is known
     *    5. Add safety counters for complex conditions
     *    6. Use debugger to step through if loop seems stuck
     */

    return 0;
}

/*
 * CONTROL FLOW IN GPU PROGRAMMING:
 * =================================
 * Control flow has MAJOR performance implications in CUDA:
 *
 * 1. Branch Divergence:
 *    if (threadIdx.x % 2 == 0) {
 *        // Half the threads do this
 *    } else {
 *        // Other half do this
 *    }
 *    This is SLOW! Both paths execute serially.
 *
 * 2. Better: Reorganize to avoid divergence:
 *    int value = (threadIdx.x % 2 == 0) ? A : B;
 *    // All threads execute same code with different data
 *
 * 3. Loop Unrolling in CUDA:
 *    #pragma unroll
 *    for (int i = 0; i < 4; i++) {
 *        sum += data[i];
 *    }
 *    // Compiler fully unrolls, eliminating loop overhead
 *
 * 4. Warp-Uniform Control Flow:
 *    if (__all_sync(mask, condition)) {
 *        // All threads in warp have same condition - NO divergence!
 *    }
 *
 * COMPILATION:
 * ============
 * g++ -std=c++17 -Wall -Wextra 03_control_flow.cpp -o control_flow
 * ./control_flow
 *
 * LEARNING CHECKLIST:
 * ===================
 * ☐ Can use if-else for conditional logic
 * ☐ Know when to use switch vs if-else
 * ☐ Master all loop types (for, while, do-while)
 * ☐ Understand break and continue
 * ☐ Can write and optimize nested loops
 * ☐ Understand loop performance implications
 * ☐ Know about branch divergence (GPU-specific)
 * ☐ Can analyze time complexity of loops
 * ☐ Can implement common algorithms using loops
 *
 * NEXT STEPS:
 * ===========
 * - Move to 04_functions.cpp
 * - Practice loop-based algorithms on LeetCode
 * - Study branch divergence in CUDA programming
 * - Learn about loop vectorization and SIMD
 * - Understand how compilers optimize loops
 */
