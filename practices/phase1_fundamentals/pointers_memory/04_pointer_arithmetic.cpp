/*
 * Exercise: Pointer Arithmetic
 * Difficulty: Intermediate
 * Time: 40-50 minutes
 * Topics: Pointer increment/decrement, pointer differences, array traversal, void pointers
 *
 * LEARNING OBJECTIVES:
 * - Master pointer increment and decrement
 * - Understand pointer arithmetic with different types
 * - Learn pointer subtraction and comparison
 * - Practice array traversal using pointers
 * - Understand memory addresses and offsets
 * - Learn void pointer arithmetic limitations
 *
 * INTERVIEW RELEVANCE:
 * - Pointer arithmetic is frequently tested
 * - Understanding memory layout is critical
 * - Array/pointer relationship is fundamental
 * - GPU programming heavily uses pointer arithmetic
 * - Kernel indexing uses pointer arithmetic concepts
 */

#include <iostream>
#include <cstring>

int main() {
    std::cout << "=== Pointer Arithmetic Exercises ===\n\n";

    // ========================================================================
    // EXERCISE 1: Basic Pointer Increment/Decrement (10 min)
    // ========================================================================
    std::cout << "Exercise 1: Pointer Increment/Decrement\n";
    std::cout << "----------------------------------------\n";

    int arr[] = {10, 20, 30, 40, 50};
    int* p = arr;  // Points to arr[0]

    std::cout << "Initial: p = " << p << ", *p = " << *p << "\n";

    // TODO 1.1: Increment pointer
    p++;  // Now points to arr[1]
    std::cout << "After p++: p = " << p << ", *p = " << *p << "\n";

    p++;  // Now points to arr[2]
    std::cout << "After p++: p = " << p << ", *p = " << *p << "\n";

    // TODO 1.2: Decrement pointer
    p--;  // Back to arr[1]
    std::cout << "After p--: p = " << p << ", *p = " << *p << "\n";

    // TODO 1.3: Add to pointer
    p = arr;  // Reset
    p += 3;   // Jump to arr[3]
    std::cout << "After p += 3: p = " << p << ", *p = " << *p << "\n";

    // TODO 1.4: Subtract from pointer
    p -= 2;   // Back to arr[1]
    std::cout << "After p -= 2: p = " << p << ", *p = " << *p << "\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 2: Pointer Arithmetic with Different Types (10 min)
    // ========================================================================
    std::cout << "Exercise 2: Different Types\n";
    std::cout << "---------------------------\n";

    // TODO 2.1: int pointer arithmetic
    int ints[] = {1, 2, 3};
    int* pi = ints;

    std::cout << "int array:\n";
    std::cout << "  pi:     " << (void*)pi << "\n";
    std::cout << "  pi + 1: " << (void*)(pi + 1) << "\n";
    std::cout << "  Difference: " << (char*)(pi + 1) - (char*)pi << " bytes\n";
    std::cout << "  sizeof(int): " << sizeof(int) << " bytes\n";

    // TODO 2.2: double pointer arithmetic
    double doubles[] = {1.1, 2.2, 3.3};
    double* pd = doubles;

    std::cout << "\ndouble array:\n";
    std::cout << "  pd:     " << (void*)pd << "\n";
    std::cout << "  pd + 1: " << (void*)(pd + 1) << "\n";
    std::cout << "  Difference: " << (char*)(pd + 1) - (char*)pd << " bytes\n";
    std::cout << "  sizeof(double): " << sizeof(double) << " bytes\n";

    // TODO 2.3: char pointer arithmetic
    char chars[] = "ABC";
    char* pc = chars;

    std::cout << "\nchar array:\n";
    std::cout << "  pc:     " << (void*)pc << "\n";
    std::cout << "  pc + 1: " << (void*)(pc + 1) << "\n";
    std::cout << "  Difference: " << (pc + 1) - pc << " bytes\n";

    // Key point: p + 1 advances by sizeof(*p) bytes!

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 3: Pointer Subtraction (10 min)
    // ========================================================================
    std::cout << "Exercise 3: Pointer Subtraction\n";
    std::cout << "--------------------------------\n";

    int numbers[] = {10, 20, 30, 40, 50, 60};

    // TODO 3.1: Distance between pointers
    int* first = &numbers[0];
    int* last = &numbers[5];

    std::cout << "first points to: " << *first << "\n";
    std::cout << "last points to: " << *last << "\n";
    std::cout << "Distance (last - first): " << (last - first) << " elements\n";
    std::cout << "Distance in bytes: " << (char*)last - (char*)first << " bytes\n";

    // TODO 3.2: Pointer subtraction gives number of elements
    int* mid = &numbers[3];
    std::cout << "\nmid - first = " << (mid - first) << " elements\n";
    std::cout << "last - mid = " << (last - mid) << " elements\n";

    // TODO 3.3: Can't subtract pointers to different arrays
    // int arr1[5], arr2[5];
    // int* p1 = arr1;
    // int* p2 = arr2;
    // int diff = p1 - p2;  // UNDEFINED BEHAVIOR!

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 4: Pointer Comparison (5 min)
    // ========================================================================
    std::cout << "Exercise 4: Pointer Comparison\n";
    std::cout << "-------------------------------\n";

    int data[] = {1, 2, 3, 4, 5};
    int* start = data;
    int* end = data + 5;  // One past last element

    // TODO 4.1: Compare pointers
    std::cout << "start < end: " << (start < end) << "\n";
    std::cout << "start == data: " << (start == data) << "\n";
    std::cout << "end > start: " << (end > start) << "\n";

    // TODO 4.2: Iterate using pointer comparison
    std::cout << "\nArray elements: ";
    for (int* ptr = start; ptr < end; ptr++) {
        std::cout << *ptr << " ";
    }
    std::cout << "\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 5: Array Traversal Using Pointers (10 min)
    // ========================================================================
    std::cout << "Exercise 5: Array Traversal\n";
    std::cout << "---------------------------\n";

    int values[] = {5, 10, 15, 20, 25};
    int size = sizeof(values) / sizeof(values[0]);

    // TODO 5.1: Traverse with pointer increment
    std::cout << "Method 1 (pointer increment):\n";
    int* ptr = values;
    for (int i = 0; i < size; i++) {
        std::cout << "  values[" << i << "] = " << *ptr << "\n";
        ptr++;
    }

    // TODO 5.2: Traverse with pointer offset
    std::cout << "\nMethod 2 (pointer offset):\n";
    ptr = values;
    for (int i = 0; i < size; i++) {
        std::cout << "  values[" << i << "] = " << *(ptr + i) << "\n";
    }

    // TODO 5.3: Traverse with array indexing on pointer
    std::cout << "\nMethod 3 (array notation on pointer):\n";
    ptr = values;
    for (int i = 0; i < size; i++) {
        std::cout << "  values[" << i << "] = " << ptr[i] << "\n";
    }
    // Note: ptr[i] is exactly the same as *(ptr + i)

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 6: Multidimensional Arrays (10 min)
    // ========================================================================
    std::cout << "Exercise 6: Multidimensional Arrays\n";
    std::cout << "------------------------------------\n";

    int matrix[3][4] = {
        {1,  2,  3,  4},
        {5,  6,  7,  8},
        {9, 10, 11, 12}
    };

    // TODO 6.1: Pointer to first row
    int (*rowPtr)[4] = matrix;  // Pointer to array of 4 ints

    std::cout << "First row, first element: " << rowPtr[0][0] << "\n";
    std::cout << "Second row, third element: " << rowPtr[1][2] << "\n";

    // TODO 6.2: Flatten 2D array with pointer
    int* flatPtr = &matrix[0][0];

    std::cout << "\nFlattened matrix:\n";
    for (int i = 0; i < 12; i++) {
        std::cout << flatPtr[i] << " ";
        if ((i + 1) % 4 == 0) std::cout << "\n";
    }

    // TODO 6.3: Pointer arithmetic in 2D
    std::cout << "\nUsing pointer arithmetic:\n";
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 4; col++) {
            std::cout << *(flatPtr + row * 4 + col) << " ";
        }
        std::cout << "\n";
    }
    // Formula: matrix[row][col] = *(base + row * num_cols + col)

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 7: Void Pointers (5 min)
    // ========================================================================
    std::cout << "Exercise 7: Void Pointers\n";
    std::cout << "-------------------------\n";

    int x = 42;
    double d = 3.14;

    // TODO 7.1: void pointer can point to any type
    void* vp = &x;
    std::cout << "void* pointing to int: " << *(int*)vp << "\n";

    vp = &d;
    std::cout << "void* pointing to double: " << *(double*)vp << "\n";

    // TODO 7.2: Cannot do pointer arithmetic on void*
    // void* vp2 = vp + 1;  // ERROR! Size unknown

    // TODO 7.3: Must cast to do arithmetic
    char* cp = (char*)vp;
    cp += sizeof(double);  // OK after cast

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 8: Practical Applications (10 min)
    // ========================================================================
    std::cout << "Exercise 8: Practical Applications\n";
    std::cout << "-----------------------------------\n";

    // TODO 8.1: Reverse array using pointers
    int toReverse[] = {1, 2, 3, 4, 5};
    int* left = toReverse;
    int* right = toReverse + 4;  // Last element

    std::cout << "Original: ";
    for (int i = 0; i < 5; i++) std::cout << toReverse[i] << " ";

    while (left < right) {
        int temp = *left;
        *left = *right;
        *right = temp;
        left++;
        right--;
    }

    std::cout << "\nReversed: ";
    for (int i = 0; i < 5; i++) std::cout << toReverse[i] << " ";
    std::cout << "\n";

    // TODO 8.2: Find element using pointer
    int search[] = {10, 20, 30, 40, 50};
    int targetVal = 30;
    int* found = nullptr;

    for (int* p = search; p < search + 5; p++) {
        if (*p == targetVal) {
            found = p;
            break;
        }
    }

    if (found) {
        std::cout << "\nFound " << targetVal << " at index " << (found - search) << "\n";
    }

    // TODO 8.3: Copy array using pointers
    int src[] = {1, 2, 3, 4, 5};
    int dest[5];

    int* s = src;
    int* d = dest;
    int* sEnd = src + 5;

    while (s < sEnd) {
        *d++ = *s++;  // Copy and increment
    }

    std::cout << "\nCopied array: ";
    for (int i = 0; i < 5; i++) std::cout << dest[i] << " ";
    std::cout << "\n";

    std::cout << "\n";

    return 0;
}

// ========================================================================
// COMMON INTERVIEW QUESTIONS
// ========================================================================
/*
 * Q1: What does pointer arithmetic do?
 * A: When you add or subtract from a pointer, it moves by the size
 *    of the type it points to.
 *
 *    int* p = ...;
 *    p + 1  →  moves by sizeof(int) bytes (typically 4)
 *    p + n  →  moves by n * sizeof(int) bytes
 *
 *    This is why:
 *    - int* p + 1 moves 4 bytes
 *    - double* p + 1 moves 8 bytes
 *    - char* p + 1 moves 1 byte
 *
 * Q2: What's the relationship between arrays and pointers?
 * A: An array name is a constant pointer to the first element.
 *
 *    int arr[5];
 *    arr is equivalent to &arr[0]
 *    arr[i] is equivalent to *(arr + i)
 *
 *    Key differences:
 *    - arr is constant (can't do arr++)
 *    - sizeof(arr) gives total array size
 *    - sizeof(pointer) gives pointer size
 *    - Arrays decay to pointers when passed to functions
 *
 * Q3: What does subtracting two pointers give you?
 * A: The number of elements between them (not bytes!).
 *
 *    int arr[10];
 *    int* p1 = &arr[2];
 *    int* p2 = &arr[7];
 *    p2 - p1 = 5  (elements, not 20 bytes)
 *
 *    Valid only for pointers to the same array!
 *
 * Q4: Can you do arithmetic on void pointers?
 * A: NO! The compiler doesn't know the size.
 *
 *    void* p = ...;
 *    p + 1;  // ERROR! Size unknown
 *
 *    Must cast first:
 *    char* cp = (char*)p;
 *    cp + 1;  // OK, moves 1 byte
 *
 * Q5: What's the difference between *p++ and (*p)++?
 * A: *p++  →  *(p++)  →  Dereference p, then increment p
 *    - Returns current value, then moves pointer
 *
 *    (*p)++  →  Increment the value pointed to
 *    - Keeps pointer same, increments value
 *
 *    Example:
 *    int arr[] = {10, 20, 30};
 *    int* p = arr;
 *
 *    *p++ = 100;  // arr[0] = 100, p now points to arr[1]
 *    (*p)++ = 200;  // arr[1]++, p still points to arr[1]
 *
 * Q6: How do you access a 2D array using pointer arithmetic?
 * A: matrix[row][col] = *(base + row * num_cols + col)
 *
 *    int matrix[3][4];
 *    int* p = &matrix[0][0];
 *
 *    Access matrix[1][2]:
 *    *(p + 1 * 4 + 2) = *(p + 6)
 *
 *    Or use pointer to array:
 *    int (*rowPtr)[4] = matrix;
 *    rowPtr[1][2]  // Cleaner
 *
 * Q7: What's a pointer to an array vs array of pointers?
 * A: Pointer to array:
 *    int (*p)[5];     // Pointer to array of 5 ints
 *    int arr[5];
 *    p = &arr;        // Points to entire array
 *
 *    Array of pointers:
 *    int* p[5];       // Array of 5 int pointers
 *    int a, b, c, d, e;
 *    p[0] = &a;
 *    p[1] = &b;
 *    // etc.
 *
 * Q8: Can you compare pointers from different arrays?
 * A: Technically yes, but the result is UNDEFINED unless:
 *    - Comparing for equality/inequality
 *    - Both are null
 *    - Comparing with nullptr
 *
 *    For <, >, <=, >= comparisons, pointers must be:
 *    - To elements of the same array, OR
 *    - One past the end of the same array
 *
 *    Otherwise: undefined behavior!
 */

/*
 * POINTER ARITHMETIC IN GPU PROGRAMMING:
 * =======================================
 *
 * 1. Thread Indexing (most common use):
 *    __global__ void kernel(float* data) {
 *        int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *        float* ptr = data + idx;  // Pointer arithmetic!
 *        *ptr = ...;
 *    }
 *
 * 2. Strided Access:
 *    __global__ void strided(float* data, int stride) {
 *        int idx = threadIdx.x;
 *        float* ptr = data + idx * stride;
 *        // Process every stride-th element
 *    }
 *
 * 3. 2D Array Access:
 *    __global__ void matrix(float* mat, int width, int height) {
 *        int row = blockIdx.y * blockDim.y + threadIdx.y;
 *        int col = blockIdx.x * blockDim.x + threadIdx.x;
 *        float* elem = mat + row * width + col;
 *        *elem = ...;
 *    }
 *
 * 4. Shared Memory Indexing:
 *    __shared__ float shared[256];
 *    int tid = threadIdx.x;
 *    float* ptr = shared + tid;
 *    *ptr = data[tid];
 *
 * 5. Pointer Offset vs Array Indexing:
 *    // Same thing, different syntax
 *    data[idx] = ...;      // Array syntax (clearer)
 *    *(data + idx) = ...;  // Pointer syntax
 *
 * 6. Memory Coalescing Example:
 *    // Good: Sequential access
 *    __global__ void good(float* data) {
 *        int idx = threadIdx.x;  // 0, 1, 2, 3, ...
 *        data[idx] = ...;        // Coalesced!
 *    }
 *
 *    // Bad: Strided access
 *    __global__ void bad(float* data) {
 *        int idx = threadIdx.x * 32;  // 0, 32, 64, 96, ...
 *        data[idx] = ...;              // Not coalesced!
 *    }
 *
 * COMPILATION:
 * ============
 * g++ -std=c++17 -Wall -Wextra 04_pointer_arithmetic.cpp -o ptrarith
 * ./ptrarith
 *
 * LEARNING CHECKLIST:
 * ===================
 * ☐ Understand pointer increment/decrement
 * ☐ Know arithmetic works with type size
 * ☐ Can subtract pointers to get distance
 * ☐ Can compare pointers
 * ☐ Master array traversal with pointers
 * ☐ Understand 2D array pointer arithmetic
 * ☐ Know void pointer limitations
 * ☐ Can use pointer arithmetic in practice
 *
 * NEXT STEPS:
 * ===========
 * - Move to phase1_fundamentals/functions/
 * - Study function pointers
 * - Learn about pointer aliasing
 * - Understand restrict keyword
 * - Practice GPU thread indexing patterns
 */
