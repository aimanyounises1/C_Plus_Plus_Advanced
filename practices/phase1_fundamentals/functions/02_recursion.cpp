/*
 * ============================================================================
 * Exercise: Recursion in C++
 * ============================================================================
 * Difficulty: Intermediate
 * Time: 50-60 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Understand recursive function design and base cases
 * 2. Learn call stack mechanics and memory usage
 * 3. Practice tail recursion and optimization
 * 4. Compare iterative vs recursive solutions
 * 5. Recognize when recursion is appropriate
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Recursion is commonly tested in technical interviews
 * - Understanding call stacks is critical for debugging
 * - Tail recursion relates to compiler optimizations
 * - Tree/graph traversals often use recursion
 * - Stack overflow awareness important for GPU programming
 *
 * PREREQUISITES:
 * - Function definition and calling
 * - Control flow (if/else)
 * - Basic understanding of the call stack
 * ============================================================================
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

using namespace std;

/*
 * ============================================================================
 * EXERCISE 1: Basic Recursion - Factorial (10 minutes)
 * ============================================================================
 * Understand the fundamental structure of recursive functions
 */

// TODO 1.1: Implement recursive factorial function
// Requirements:
// - Base case: factorial(0) = 1
// - Recursive case: factorial(n) = n * factorial(n-1)
// - Return type: unsigned long long
// Hint: Think about when to stop recursing (base case!)


// TODO 1.2: Implement iterative factorial for comparison
// Requirements:
// - Use a loop instead of recursion
// - Should produce same results as recursive version


// TODO 1.3: Test both implementations
// Print factorials of 0, 1, 5, 10, 15, 20
// Compare execution and readability


/*
 * ============================================================================
 * EXERCISE 2: Fibonacci Sequence (10 minutes)
 * ============================================================================
 * Learn about the performance pitfalls of naive recursion
 */

// TODO 2.1: Implement naive recursive Fibonacci
// Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34...
// Formula: fib(n) = fib(n-1) + fib(n-2)
// Base cases: fib(0) = 0, fib(1) = 1


// TODO 2.2: Add a call counter to visualize redundant calls
// Hint: Use a static variable or pass a reference parameter
// Calculate fib(10) and count how many recursive calls are made


// TODO 2.3: Implement iterative Fibonacci
// Use a loop to calculate Fibonacci numbers efficiently


// TODO 2.4: Compare performance
// Time both implementations for fib(30) or fib(35)
// Notice the dramatic difference!


/*
 * ============================================================================
 * EXERCISE 3: Tail Recursion (10 minutes)
 * ============================================================================
 * Learn about tail-call optimization and efficient recursion
 */

// TODO 3.1: Implement tail-recursive factorial
// A tail-recursive function has the recursive call as the last operation
// Use an accumulator parameter to maintain state
// Example signature: unsigned long long factorial_tail(int n, unsigned long long acc = 1)


// TODO 3.2: Implement tail-recursive Fibonacci
// Use two accumulator parameters (a and b) to track previous values
// Example signature: long long fib_tail(int n, long long a = 0, long long b = 1)


// TODO 3.3: Test tail-recursive implementations
// Verify they produce correct results
// Note: C++ compilers may optimize tail recursion to iteration


/*
 * ============================================================================
 * EXERCISE 4: Recursive Array Operations (10 minutes)
 * ============================================================================
 * Apply recursion to array/vector processing
 */

// TODO 4.1: Implement recursive array sum
// Calculate sum of all elements in an array using recursion
// Base case: empty array (size 0) returns 0
// Recursive case: first element + sum of rest
// Hint: Use array index or vector iterators


// TODO 4.2: Implement recursive array maximum
// Find the maximum element in an array using recursion
// Base case: single element array returns that element
// Recursive case: max(first element, max of rest)


// TODO 4.3: Implement recursive binary search
// Search for a target value in a sorted array
// Return the index if found, -1 if not found
// Parameters: array, target, low index, high index


/*
 * ============================================================================
 * EXERCISE 5: String Recursion (10 minutes)
 * ============================================================================
 * Practice recursion with strings
 */

// TODO 5.1: Implement recursive string reversal
// Reverse a string using recursion
// Base case: empty or single-character string
// Recursive case: last character + reverse(rest of string)


// TODO 5.2: Implement recursive palindrome checker
// Check if a string is a palindrome using recursion
// Base case: strings of length 0 or 1 are palindromes
// Recursive case: first == last && isPalindrome(middle substring)


// TODO 5.3: Implement recursive vowel counter
// Count the number of vowels in a string recursively
// Base case: empty string has 0 vowels
// Recursive case: (1 if first is vowel else 0) + count(rest)


/*
 * ============================================================================
 * EXERCISE 6: Advanced Recursion - Permutations (15 minutes)
 * ============================================================================
 * Solve a complex problem using recursion
 */

// TODO 6.1: Generate all permutations of a string
// Example: "ABC" -> ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"]
// Use backtracking approach:
// 1. For each character, swap it to the front
// 2. Recursively generate permutations of the rest
// 3. Backtrack (swap back)


// TODO 6.2: Print all permutations
// Call your permutation function and display results


/*
 * ============================================================================
 * CHALLENGE EXERCISES (Optional - 20 minutes)
 * ============================================================================
 */

// CHALLENGE 1: Tower of Hanoi
// Solve the classic Tower of Hanoi puzzle recursively
// Given 3 pegs and n disks, move all disks from source to destination
// Rules: Only one disk at a time, larger disk can't be on smaller disk
// Print each move
// Hint: Move n-1 disks to auxiliary, move largest to dest, move n-1 to dest


// CHALLENGE 2: Recursive Maze Solver
// Given a 2D grid representing a maze:
// - 0 = path, 1 = wall, 2 = destination
// - Start at position (0, 0)
// Find a path to the destination using recursion (backtracking)
// Return true if path exists, false otherwise


// CHALLENGE 3: Memoized Fibonacci
// Implement Fibonacci with memoization to avoid redundant calculations
// Use a static map or vector to cache previously computed values
// This should be as fast as iterative but keep recursive elegance


/*
 * ============================================================================
 * PRACTICAL APPLICATION: Recursive Directory Tree (15 minutes)
 * ============================================================================
 */

// APPLICATION 1: Print a simulated directory tree recursively
// Given a simple tree structure (vector of files and subdirectories),
// print the tree with proper indentation
// Example output:
// root/
//   file1.txt
//   subdir1/
//     file2.txt
//     file3.txt
//   subdir2/
//     file4.txt


/*
 * ============================================================================
 * COMMON INTERVIEW QUESTIONS & ANSWERS
 * ============================================================================
 *
 * Q1: What is recursion and what are its key components?
 * A: Recursion is when a function calls itself to solve a problem by breaking
 *    it into smaller subproblems. Key components:
 *    1. Base case: Condition where recursion stops (prevents infinite loop)
 *    2. Recursive case: Function calls itself with modified parameters
 *    3. Progress toward base case: Each call must move closer to base case
 *    Example: factorial(n) = n * factorial(n-1), base case factorial(0) = 1
 *
 * Q2: What are the advantages and disadvantages of recursion?
 * A: Advantages:
 *    - More elegant and readable for naturally recursive problems
 *    - Simplifies complex problems (tree traversal, backtracking)
 *    - Matches mathematical definitions
 *    Disadvantages:
 *    - Uses more memory (each call adds stack frame)
 *    - Can be slower than iterative solutions
 *    - Risk of stack overflow for deep recursion
 *    - Harder to debug for beginners
 *
 * Q3: What is tail recursion and why is it important?
 * A: Tail recursion occurs when the recursive call is the last operation in
 *    the function (no pending operations after the call returns). Importance:
 *    - Compilers can optimize tail recursion into iteration (tail call optimization)
 *    - Eliminates stack growth, preventing overflow
 *    - Same performance as loops with recursive elegance
 *    Example: factorial_tail(n, acc) = factorial_tail(n-1, n*acc)
 *    Non-tail: factorial(n) = n * factorial(n-1)  // multiplication pending
 *
 * Q4: How do you decide between recursion and iteration?
 * A: Use recursion when:
 *    - Problem is naturally recursive (trees, graphs, divide-and-conquer)
 *    - Recursive solution is significantly simpler and clearer
 *    - Depth is bounded and reasonable
 *    Use iteration when:
 *    - Performance is critical
 *    - Recursion depth could be very large (risk of stack overflow)
 *    - Iterative solution is simple enough
 *    - Working with limited stack space (embedded systems, GPU kernels)
 *
 * Q5: What is the call stack and how does recursion use it?
 * A: The call stack is a region of memory that stores information about active
 *    function calls. For each function call, a stack frame is pushed containing:
 *    - Return address (where to continue after function returns)
 *    - Parameters and local variables
 *    - Saved registers
 *    In recursion, each recursive call adds a new frame. When base case is
 *    reached, frames are popped in reverse order (LIFO). Example with factorial(3):
 *    Stack grows: main -> factorial(3) -> factorial(2) -> factorial(1) -> factorial(0)
 *    Then unwinds: factorial(0) returns 1, factorial(1) returns 1*1=1,
 *    factorial(2) returns 2*1=2, factorial(3) returns 3*2=6
 *
 * Q6: What causes stack overflow in recursion and how to prevent it?
 * A: Stack overflow occurs when recursion depth exceeds available stack space.
 *    Common causes:
 *    - Missing or incorrect base case (infinite recursion)
 *    - Deep recursion on large inputs
 *    - Large local variables in each recursive call
 *    Prevention:
 *    - Ensure correct base case
 *    - Use iteration for deep recursion
 *    - Use tail recursion (compiler may optimize)
 *    - Increase stack size (compiler/system settings)
 *    - Use explicit stack data structure instead of call stack
 *
 * Q7: What is memoization and when should it be used with recursion?
 * A: Memoization is caching the results of expensive function calls and
 *    returning cached result when same inputs occur again. Use when:
 *    - Same subproblems are solved multiple times (overlapping subproblems)
 *    - Example: Naive Fibonacci has exponential time due to redundant calls
 *    - fib(5) calls fib(4) and fib(3), fib(4) also calls fib(3) (duplicate!)
 *    With memoization:
 *    - Store fib(n) in a cache (map/array) after computing it
 *    - Check cache before recursing
 *    - Reduces time complexity from O(2^n) to O(n) for Fibonacci
 *
 * Q8: How is recursion different from iteration in terms of performance?
 * A: Performance differences:
 *    Time complexity:
 *    - Often the same algorithmic complexity
 *    - Recursion has function call overhead (parameter passing, stack manipulation)
 *    - Iteration typically faster by constant factor
 *    Space complexity:
 *    - Recursion: O(depth) stack space for call frames
 *    - Iteration: O(1) stack space (only loop variables)
 *    - Exception: Tail recursion can be optimized to O(1)
 *    Practical impact:
 *    - Recursion depth of 1000s can cause stack overflow
 *    - GPU programming: Limited stack, prefer iteration
 *    - Modern CPUs: Branch prediction may favor iteration
 *
 * Q9: Explain the concept of "divide and conquer" in recursive algorithms.
 * A: Divide and conquer is a strategy where a problem is:
 *    1. Divide: Break problem into smaller subproblems
 *    2. Conquer: Solve subproblems recursively
 *    3. Combine: Merge solutions to get final answer
 *    Examples:
 *    - Merge sort: Divide array in half, sort each half, merge
 *    - Quick sort: Partition around pivot, sort partitions
 *    - Binary search: Check middle, search left or right half
 *    - Strassen's matrix multiplication
 *    Benefits: Often achieves better time complexity than naive approaches
 *    (e.g., O(n log n) sorting vs O(n²))
 *
 * Q10: How would you convert a recursive algorithm to an iterative one?
 * A: General approaches:
 *    1. Use explicit stack: Replace call stack with your own stack data structure
 *       - Push state before "recursive call"
 *       - Pop and process in loop
 *       - Example: Iterative tree traversal using stack
 *    2. Tail recursion: Convert to simple loop
 *       - Replace recursive call with assignment and continue
 *       - factorial_tail becomes while loop updating accumulator
 *    3. Dynamic programming: For problems with overlapping subproblems
 *       - Compute solutions bottom-up instead of top-down
 *       - Example: Iterative Fibonacci using two variables
 *    4. State machine: Track state explicitly instead of implicitly on call stack
 *    Trade-off: Iterative version may be less intuitive but more efficient
 *
 * ============================================================================
 * GPU/CUDA RELEVANCE FOR NVIDIA INTERVIEW:
 * ============================================================================
 *
 * 1. Limited Stack Space: GPU kernels have very limited stack space (typically
 *    only a few KB per thread). Deep recursion can easily overflow. Prefer
 *    iteration or ensure bounded recursion depth.
 *
 * 2. CUDA Dynamic Parallelism: CUDA allows kernel to launch kernels (recursive
 *    kernels possible), but it's expensive. Better to express parallelism
 *    differently.
 *
 * 3. Tree Traversal: Common in GPU algorithms (BVH for ray tracing, kd-trees).
 *    Usually implemented iteratively with explicit stack to avoid overflow.
 *
 * 4. Divide and Conquer: Parallel algorithms often use recursive decomposition
 *    conceptually (merge sort, quick sort) but implement with iterative
 *    coordination.
 *
 * 5. Work Distribution: Understanding recursion helps with parallel algorithm
 *    design where work is recursively subdivided among threads/blocks.
 *
 * ============================================================================
 * COMPILATION & EXECUTION:
 * ============================================================================
 *
 * Compile with:
 *   g++ -std=c++17 -Wall -O2 02_recursion.cpp -o recursion
 *
 * Run:
 *   ./recursion
 *
 * For deeper recursion (if needed), increase stack size:
 *   ulimit -s unlimited    # Linux/Mac
 *
 * ============================================================================
 * EXPECTED OUTPUT (after completing exercises):
 * ============================================================================
 *
 * Should demonstrate:
 * - Factorial calculations (recursive vs iterative)
 * - Fibonacci numbers showing performance difference
 * - Tail-recursive implementations
 * - Recursive array operations (sum, max, binary search)
 * - String operations (reverse, palindrome, vowel count)
 * - Permutations of a string
 * - Optional: Tower of Hanoi moves, maze solution, directory tree
 *
 * ============================================================================
 * LEARNING CHECKLIST:
 * ============================================================================
 *
 * After completing these exercises, you should be able to:
 * ☐ Write recursive functions with correct base cases
 * ☐ Understand and explain call stack behavior
 * ☐ Identify when recursion is appropriate vs iteration
 * ☐ Implement tail-recursive functions
 * ☐ Recognize and solve overlapping subproblems with memoization
 * ☐ Convert between recursive and iterative implementations
 * ☐ Analyze space complexity of recursive algorithms
 * ☐ Implement backtracking algorithms
 * ☐ Debug recursive functions effectively
 * ☐ Explain recursion trade-offs in interviews
 *
 * ============================================================================
 * NEXT STEPS:
 * ============================================================================
 *
 * 1. Complete 03_header_files.cpp to finish Phase 1 functions
 * 2. Move to Phase 2: Object-Oriented Programming
 * 3. Practice explaining your recursive solutions out loud
 * 4. Try converting your recursive solutions to iterative ones
 * 5. Time different approaches to understand performance implications
 * 6. Study tree and graph traversal algorithms (heavily use recursion)
 *
 * ============================================================================
 */

int main() {
    cout << "=== Recursion Practice ===" << endl;
    cout << "Complete the TODOs above and test your implementations here!" << endl;

    // Test your implementations here

    return 0;
}
