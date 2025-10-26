/*
 * Exercise: Arrays and Strings
 * Difficulty: Beginner
 * Time: 45-55 minutes
 * Topics: Static arrays, C-strings, std::string, array operations, memory layout
 *
 * LEARNING OBJECTIVES:
 * - Understand static arrays and their limitations
 * - Learn C-style strings (char arrays)
 * - Master std::string class operations
 * - Practice common array algorithms
 * - Understand memory layout (crucial for GPU!)
 * - Learn Array of Structures (AoS) vs Structure of Arrays (SoA)
 *
 * INTERVIEW RELEVANCE:
 * - Array manipulation is fundamental
 * - String problems are very common in interviews
 * - Understanding memory layout is critical for optimization
 * - AoS vs SoA is important for GPU programming (memory coalescing)
 * - Many algorithmic problems use arrays as the primary data structure
 */

#include <iostream>
#include <string>
#include <cstring>  // For C-string functions
#include <algorithm> // For std::sort, std::reverse
#include <cctype>    // For character functions

int main() {
    std::cout << "=== Arrays and Strings Exercises ===\n\n";

    // ========================================================================
    // EXERCISE 1: Static Arrays - Declaration and Initialization (5 min)
    // ========================================================================
    std::cout << "Exercise 1: Static Arrays\n";
    std::cout << "-------------------------\n";

    // TODO 1.1: Declare an integer array of size 5
    // int numbers[5];


    // TODO 1.2: Initialize array with values
    int values[5] = {10, 20, 30, 40, 50};
    // Print all elements


    // TODO 1.3: Partial initialization (rest are zero)
    int partial[10] = {1, 2, 3};  // Rest are 0
    // Print to verify


    // TODO 1.4: Array size using sizeof
    // sizeof(array) / sizeof(array[0]) gives number of elements
    int data[] = {5, 10, 15, 20, 25};
    int size = sizeof(data) / sizeof(data[0]);
    std::cout << "Array size: " << size << "\n";

    // TODO 1.5: Access and modify elements
    // Remember: array indices start at 0!


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 2: Array Operations (10 min)
    // ========================================================================
    std::cout << "Exercise 2: Array Operations\n";
    std::cout << "-----------------------------\n";

    int nums[] = {23, 45, 12, 67, 34, 89, 11};
    int n = sizeof(nums) / sizeof(nums[0]);

    // TODO 2.1: Find the maximum element
    int max = nums[0];
    // Loop through array to find max


    // TODO 2.2: Find the minimum element


    // TODO 2.3: Calculate the sum and average
    int sum = 0;


    // TODO 2.4: Linear search - find index of a value
    int target = 67;
    int index = -1;  // -1 means not found


    // TODO 2.5: Count occurrences of a value
    int arr[] = {1, 2, 3, 2, 4, 2, 5, 2};
    int arrSize = sizeof(arr) / sizeof(arr[0]);
    int searchValue = 2;
    int count = 0;


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 3: Array Algorithms (10 min)
    // ========================================================================
    std::cout << "Exercise 3: Array Algorithms\n";
    std::cout << "----------------------------\n";

    // TODO 3.1: Reverse an array
    int original[] = {1, 2, 3, 4, 5};
    int len = sizeof(original) / sizeof(original[0]);
    std::cout << "Original: ";
    // Print array
    // Reverse it (swap first with last, second with second-to-last, etc.)
    std::cout << "Reversed: ";
    // Print reversed array


    // TODO 3.2: Rotate array left by k positions
    // [1, 2, 3, 4, 5] rotated left by 2 -> [3, 4, 5, 1, 2]
    int rotate[] = {1, 2, 3, 4, 5};
    int k = 2;


    // TODO 3.3: Sort an array (use std::sort)
    int unsorted[] = {64, 34, 25, 12, 22, 11, 90};
    int sortSize = sizeof(unsorted) / sizeof(unsorted[0]);
    std::cout << "Before sort: ";
    // Print
    // std::sort(unsorted, unsorted + sortSize);
    std::cout << "After sort: ";
    // Print


    // TODO 3.4: Remove duplicates from sorted array
    // Return new length
    int sorted[] = {1, 1, 2, 2, 2, 3, 4, 4, 5};


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 4: Multi-dimensional Arrays (5 min)
    // ========================================================================
    std::cout << "Exercise 4: Multi-dimensional Arrays\n";
    std::cout << "-------------------------------------\n";

    // TODO 4.1: Declare and initialize a 2D array
    int matrix[3][4] = {
        {1,  2,  3,  4},
        {5,  6,  7,  8},
        {9, 10, 11, 12}
    };

    // TODO 4.2: Print 2D array
    std::cout << "Matrix:\n";
    // Use nested loops


    // TODO 4.3: Sum of all elements in 2D array


    // TODO 4.4: Find element in 2D array
    int searchTarget = 7;


    // TODO 4.5: Transpose matrix (swap rows and columns)


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 5: C-style Strings (10 min)
    // ========================================================================
    std::cout << "Exercise 5: C-style Strings\n";
    std::cout << "----------------------------\n";

    // TODO 5.1: C-string declaration and initialization
    char cstr1[] = "Hello";  // Automatically adds '\0'
    char cstr2[20] = "World";
    std::cout << "cstr1: " << cstr1 << "\n";

    // TODO 5.2: String length using strlen
    // #include <cstring>
    // int length = strlen(cstr1);


    // TODO 5.3: String copy using strcpy
    char dest[20];
    // strcpy(dest, cstr1);


    // TODO 5.4: String concatenation using strcat
    char combined[50] = "Hello ";
    // strcat(combined, "World");


    // TODO 5.5: String comparison using strcmp
    // Returns 0 if equal, <0 if first < second, >0 if first > second
    char str1[] = "apple";
    char str2[] = "banana";
    // int cmp = strcmp(str1, str2);


    // TODO 5.6: Character array iteration
    char name[] = "NVIDIA";
    std::cout << "Characters: ";
    // Loop through each character


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 6: std::string Class (10 min)
    // ========================================================================
    std::cout << "Exercise 6: std::string Class\n";
    std::cout << "------------------------------\n";

    // TODO 6.1: String declaration and initialization
    std::string str = "Hello, World!";


    // TODO 6.2: String length
    // Use str.length() or str.size()


    // TODO 6.3: String concatenation
    std::string first = "Hello";
    std::string second = " World";
    // std::string result = first + second;


    // TODO 6.4: Substring extraction
    // str.substr(start, length)
    std::string phrase = "GPU Programming";
    // Extract "GPU" (first 3 characters)


    // TODO 6.5: Find substring
    // str.find("substring") returns position or std::string::npos
    std::string text = "CUDA is for parallel computing";
    // Find position of "parallel"


    // TODO 6.6: Replace substring
    // str.replace(pos, length, newString)


    // TODO 6.7: Character access
    // Use str[i] or str.at(i)
    std::string word = "NVIDIA";
    // Print each character


    // TODO 6.8: String comparison
    // Use ==, !=, <, >, <=, >=
    std::string s1 = "apple";
    std::string s2 = "banana";


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 7: String Manipulation (10 min)
    // ========================================================================
    std::cout << "Exercise 7: String Manipulation\n";
    std::cout << "--------------------------------\n";

    // TODO 7.1: Convert to uppercase
    std::string lower = "hello world";
    std::string upper = lower;
    // Use std::toupper with std::transform
    // std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);


    // TODO 7.2: Convert to lowercase
    std::string mixed = "HELLO World";


    // TODO 7.3: Reverse a string
    std::string orig = "NVIDIA";
    std::string reversed = orig;
    // Use std::reverse
    // std::reverse(reversed.begin(), reversed.end());


    // TODO 7.4: Check if palindrome
    std::string palindrome1 = "racecar";
    std::string palindrome2 = "hello";
    // Compare string with its reverse


    // TODO 7.5: Count vowels in a string
    std::string sentence = "GPU acceleration";
    int vowelCount = 0;
    // Check each character if it's a, e, i, o, u (case insensitive)


    // TODO 7.6: Remove whitespace
    std::string spaced = "  Hello   World  ";
    // Use std::remove_if with isspace


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 8: Memory Layout - AoS vs SoA (10 min) - GPU CRITICAL!
    // ========================================================================
    std::cout << "Exercise 8: Memory Layout (AoS vs SoA)\n";
    std::cout << "---------------------------------------\n";

    // Array of Structures (AoS) - Common in CPU programming
    struct Point_AoS {
        float x;
        float y;
        float z;
    };

    Point_AoS points_aos[4] = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f},
        {10.0f, 11.0f, 12.0f}
    };

    // Memory layout: [x1,y1,z1][x2,y2,z2][x3,y3,z3][x4,y4,z4]
    // Problem for GPU: Adjacent threads access non-adjacent memory!

    std::cout << "AoS Layout:\n";
    for (int i = 0; i < 4; i++) {
        std::cout << "Point " << i << ": ("
                  << points_aos[i].x << ", "
                  << points_aos[i].y << ", "
                  << points_aos[i].z << ")\n";
    }

    // Structure of Arrays (SoA) - Better for GPU!
    struct Points_SoA {
        float x[4];
        float y[4];
        float z[4];
    };

    Points_SoA points_soa = {
        {1.0f, 4.0f, 7.0f, 10.0f},  // All x coordinates
        {2.0f, 5.0f, 8.0f, 11.0f},  // All y coordinates
        {3.0f, 6.0f, 9.0f, 12.0f}   // All z coordinates
    };

    // Memory layout: [x1,x2,x3,x4][y1,y2,y3,y4][z1,z2,z3,z4]
    // Better for GPU: Adjacent threads access adjacent memory (coalesced!)

    std::cout << "\nSoA Layout:\n";
    for (int i = 0; i < 4; i++) {
        std::cout << "Point " << i << ": ("
                  << points_soa.x[i] << ", "
                  << points_soa.y[i] << ", "
                  << points_soa.z[i] << ")\n";
    }

    // TODO 8.1: Calculate sum of all x coordinates in both layouts
    float sum_x_aos = 0.0f;
    float sum_x_soa = 0.0f;


    // TODO 8.2: Explain why SoA is better for GPU
    /*
     * SoA is better for GPU because:
     * - Memory coalescing: Adjacent threads access adjacent memory
     * - Better cache utilization
     * - Vectorization-friendly
     * - Can process one component at a time efficiently
     */

    std::cout << "\n";

    // ========================================================================
    // CHALLENGE EXERCISES (Optional - 15 min)
    // ========================================================================
    std::cout << "Challenge Exercises\n";
    std::cout << "-------------------\n";

    // CHALLENGE 1: Two Sum Problem
    // Given an array and a target, find two numbers that add up to target
    // Return their indices
    int twoSumArr[] = {2, 7, 11, 15};
    int twoSumTarget = 9;
    // Answer: indices 0 and 1 (2 + 7 = 9)


    // CHALLENGE 2: Longest Common Prefix
    // Find the longest common prefix string amongst an array of strings
    // Example: ["flower", "flow", "flight"] -> "fl"


    // CHALLENGE 3: Anagram Check
    // Check if two strings are anagrams (same letters, different order)
    // Example: "listen" and "silent" are anagrams
    std::string anagram1 = "listen";
    std::string anagram2 = "silent";


    // CHALLENGE 4: Move Zeros
    // Move all zeros in an array to the end while maintaining relative order
    // Example: [0, 1, 0, 3, 12] -> [1, 3, 12, 0, 0]
    int zeros[] = {0, 1, 0, 3, 12};


    // CHALLENGE 5: First Non-Repeating Character
    // Find the first character in a string that doesn't repeat
    std::string unique = "leetcode";  // Answer: 'l'


    // CHALLENGE 6: Matrix Spiral Traversal
    // Print a 2D matrix in spiral order
    // [1  2  3]
    // [4  5  6]  ->  1 2 3 6 9 8 7 4 5
    // [7  8  9]


    std::cout << "\n";

    return 0;
}

// ========================================================================
// COMMON INTERVIEW QUESTIONS
// ========================================================================
/*
 * Q1: What's the difference between an array and a pointer?
 * A: Arrays:
 *    - Fixed size known at compile time
 *    - Array name is a constant pointer to first element
 *    - sizeof(array) gives total size in bytes
 *    - Stored on stack (for local arrays)
 *
 *    Pointers:
 *    - Can point to any memory location
 *    - Can be reassigned
 *    - sizeof(pointer) gives size of pointer itself (usually 8 bytes on 64-bit)
 *    - Can point to heap memory
 *
 *    However, when passed to a function, arrays decay to pointers!
 *
 * Q2: What is array decay?
 * A: When you pass an array to a function, it "decays" to a pointer to
 *    the first element. You lose size information.
 *
 *    void func(int arr[]) {
 *        // arr is actually a pointer
 *        // sizeof(arr) gives sizeof(int*), not array size!
 *    }
 *
 *    Solution: Pass size as a separate parameter
 *
 * Q3: What's the difference between char[] and char*?
 * A: char arr[] = "hello";    // Mutable array on stack
 *    arr[0] = 'H';           // OK
 *
 *    char* ptr = "hello";    // Pointer to string literal (const)
 *    ptr[0] = 'H';          // UNDEFINED BEHAVIOR! May crash
 *
 *    Better: const char* ptr = "hello";
 *
 * Q4: What is the null terminator in C-strings?
 * A: The '\0' character (value 0) that marks the end of a C-string.
 *    Without it, functions like strlen won't know where the string ends!
 *
 *    char str1[6] = "hello";     // Automatically adds '\0'
 *    char str2[5] = {'h','e','l','l','o'};  // NO '\0' - DANGEROUS!
 *
 * Q5: Why use std::string over C-strings?
 * A: std::string is safer and more convenient:
 *    - Automatic memory management (no buffer overflows)
 *    - Knows its own length (no need for strlen)
 *    - Rich API (find, replace, substr, etc.)
 *    - Can grow dynamically
 *    - Works with C++ features (<<, ==, +, etc.)
 *
 *    Use C-strings only when:
 *    - Interfacing with C libraries
 *    - Performance-critical embedded systems
 *    - Working with legacy code
 *
 * Q6: What is Array of Structures (AoS) vs Structure of Arrays (SoA)?
 * A: AoS: struct Point { float x, y, z; }; Point pts[1000];
 *    Memory: [x1,y1,z1][x2,y2,z2][x3,y3,z3]...
 *    - Natural OOP style
 *    - Poor for SIMD/GPU (strided access)
 *
 *    SoA: struct Points { float x[1000], y[1000], z[1000]; };
 *    Memory: [x1,x2,x3,...][y1,y2,y3,...][z1,z2,z3,...]
 *    - Better for vectorization
 *    - Better GPU memory coalescing
 *    - Cache-friendly for operations on one component
 *
 * Q7: How do you copy an array?
 * A: Several ways:
 *
 *    1. Manual loop:
 *       for (int i = 0; i < size; i++) dest[i] = src[i];
 *
 *    2. std::copy:
 *       std::copy(src, src + size, dest);
 *
 *    3. memcpy (for POD types):
 *       memcpy(dest, src, size * sizeof(int));
 *
 *    Note: Assignment doesn't work!
 *    int a[5] = {1,2,3,4,5};
 *    int b[5] = a;  // ERROR!
 *
 * Q8: What happens if you access an array out of bounds?
 * A: UNDEFINED BEHAVIOR!
 *    - Might crash
 *    - Might read/write wrong memory
 *    - Might appear to work (worst case - silent corruption)
 *
 *    C++ doesn't check array bounds at runtime (for performance).
 *    Use std::vector or array.at() for bounds checking.
 */

/*
 * ARRAYS IN GPU PROGRAMMING:
 * ===========================
 *
 * 1. Global Memory Arrays:
 *    float* d_array;
 *    cudaMalloc(&d_array, N * sizeof(float));
 *    - Large arrays stored in global memory
 *    - High latency, coalesced access is critical
 *
 * 2. Shared Memory Arrays:
 *    __shared__ float shared_array[256];
 *    - Fast on-chip memory
 *    - Limited size (48KB typical)
 *    - Used for thread cooperation
 *
 * 3. Constant Memory:
 *    __constant__ float const_array[1024];
 *    - Read-only
 *    - Cached, fast for broadcast access
 *    - Limited to 64KB
 *
 * 4. Memory Coalescing:
 *    Good: threads 0-31 access indices 0-31
 *    Bad: threads 0-31 access strided or random indices
 *    SoA layout enables coalescing!
 *
 * 5. Bank Conflicts (Shared Memory):
 *    Avoid multiple threads accessing same bank
 *    Use padding or rearrange access patterns
 *
 * COMPILATION:
 * ============
 * g++ -std=c++17 -Wall -Wextra 05_arrays_strings.cpp -o arrays
 * ./arrays
 *
 * LEARNING CHECKLIST:
 * ===================
 * ☐ Understand static array declaration and initialization
 * ☐ Can perform common array operations (search, sort, reverse)
 * ☐ Know multi-dimensional arrays
 * ☐ Understand C-strings and their limitations
 * ☐ Master std::string class operations
 * ☐ Can manipulate strings (uppercase, reverse, palindrome)
 * ☐ Understand AoS vs SoA memory layouts
 * ☐ Know why SoA is better for GPU programming
 * ☐ Aware of array bounds and memory safety
 *
 * NEXT STEPS:
 * ===========
 * - Move to phase1_fundamentals/data_structures/01_structures.cpp
 * - Practice array/string problems on LeetCode
 * - Study memory coalescing in CUDA
 * - Learn about std::vector (dynamic arrays)
 * - Understand cache-friendly data structures
 */
