/*
 * Exercise: Pointer Basics
 * Difficulty: Beginner to Intermediate
 * Time: 50-60 minutes
 * Topics: Pointer declaration, dereferencing, address-of, null pointers, pointer arithmetic
 *
 * LEARNING OBJECTIVES:
 * - Understand what pointers are and why they exist
 * - Master pointer declaration and initialization
 * - Learn the address-of (&) and dereference (*) operators
 * - Understand null pointers and nullptr
 * - Practice pointer-to-pointer concepts
 * - Learn the relationship between pointers and arrays
 * - Understand pointer pitfalls and safety
 *
 * INTERVIEW RELEVANCE:
 * - Pointers are fundamental to C++ and very common in interviews
 * - Understanding memory addresses is critical
 * - Pointer arithmetic is used in GPU programming
 * - Device pointers (CUDA) work similarly to host pointers
 * - Memory management requires deep pointer understanding
 */

#include <iostream>
#include <cstring>

int main() {
    std::cout << "=== Pointer Basics Exercises ===\n\n";

    // ========================================================================
    // EXERCISE 1: Basic Pointer Declaration (5 min)
    // ========================================================================
    std::cout << "Exercise 1: Pointer Declaration\n";
    std::cout << "--------------------------------\n";

    // TODO 1.1: Declare an integer and a pointer to it
    int x = 42;
    int* ptr;  // Pointer to int (uninitialized - dangerous!)

    std::cout << "Value of x: " << x << "\n";
    std::cout << "Address of x: " << &x << "\n";  // & = address-of operator

    // TODO 1.2: Initialize pointer with address
    ptr = &x;  // ptr now points to x

    std::cout << "Value of ptr (address it holds): " << ptr << "\n";
    std::cout << "Value pointed to by ptr: " << *ptr << "\n";  // * = dereference operator

    // TODO 1.3: Pointer syntax variations (all equivalent)
    // int* p1;   // Recommended: * with type
    // int *p2;   // * with variable
    // int * p3;  // * in middle

    // TODO 1.4: Multiple pointer declarations
    // int* p1, p2;     // WRONG! Only p1 is a pointer, p2 is int
    // int *p1, *p2;    // Correct: both are pointers
    // int* p1; int* p2; // Clearest: separate lines

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 2: Dereferencing (10 min)
    // ========================================================================
    std::cout << "Exercise 2: Dereferencing\n";
    std::cout << "-------------------------\n";

    int value = 100;
    int* p = &value;

    std::cout << "Original value: " << value << "\n";
    std::cout << "Pointer p points to: " << *p << "\n";

    // TODO 2.1: Modify value through pointer
    *p = 200;  // Changes value through pointer
    std::cout << "After *p = 200:\n";
    std::cout << "  value = " << value << "\n";
    std::cout << "  *p = " << *p << "\n";

    // TODO 2.2: Pointer to different types
    double d = 3.14;
    double* pd = &d;
    std::cout << "\nDouble value: " << *pd << "\n";

    char c = 'A';
    char* pc = &c;
    std::cout << "Char value: " << *pc << "\n";

    // TODO 2.3: Two pointers to same variable
    int num = 50;
    int* ptr1 = &num;
    int* ptr2 = &num;

    std::cout << "\nTwo pointers to same variable:\n";
    std::cout << "  *ptr1 = " << *ptr1 << "\n";
    std::cout << "  *ptr2 = " << *ptr2 << "\n";

    *ptr1 = 60;  // Change through ptr1
    std::cout << "After *ptr1 = 60:\n";
    std::cout << "  *ptr2 = " << *ptr2 << " (also changed!)\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 3: Null Pointers (10 min)
    // ========================================================================
    std::cout << "Exercise 3: Null Pointers\n";
    std::cout << "-------------------------\n";

    // TODO 3.1: Different ways to create null pointers
    int* p1 = nullptr;     // C++11 - BEST
    int* p2 = NULL;        // C-style (macro, avoid in C++)
    int* p3 = 0;           // Old style (avoid)

    std::cout << "nullptr pointer: " << p1 << "\n";

    // TODO 3.2: Check if pointer is null before dereferencing
    int* nullPtr = nullptr;

    if (nullPtr != nullptr) {
        std::cout << "Pointer is valid: " << *nullPtr << "\n";
    } else {
        std::cout << "Pointer is null - cannot dereference!\n";
    }

    // TODO 3.3: Why nullptr is better than NULL
    // nullptr has type nullptr_t
    // NULL is typically #define NULL 0
    // nullptr is type-safe, NULL is not

    // void func(int x) { ... }
    // void func(int* p) { ... }
    // func(NULL);     // Ambiguous! Calls func(int) - WRONG
    // func(nullptr);  // Calls func(int*) - CORRECT

    // TODO 3.4: Always initialize pointers
    // int* bad;         // Uninitialized - DANGEROUS!
    // int* good = nullptr;  // Good practice

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 4: Pointer Sizes (5 min)
    // ========================================================================
    std::cout << "Exercise 4: Pointer Sizes\n";
    std::cout << "-------------------------\n";

    // TODO 4.1: All pointers have same size (architecture-dependent)
    std::cout << "sizeof(int*):    " << sizeof(int*) << " bytes\n";
    std::cout << "sizeof(double*): " << sizeof(double*) << " bytes\n";
    std::cout << "sizeof(char*):   " << sizeof(char*) << " bytes\n";

    // On 64-bit systems: all pointers are 8 bytes
    // On 32-bit systems: all pointers are 4 bytes

    // TODO 4.2: Pointer size vs pointed-to size
    int n = 42;
    int* pn = &n;

    std::cout << "\nsizeof(n):  " << sizeof(n) << " bytes (the int)\n";
    std::cout << "sizeof(pn): " << sizeof(pn) << " bytes (the pointer)\n";
    std::cout << "sizeof(*pn): " << sizeof(*pn) << " bytes (dereferenced = int)\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 5: Pointers and Arrays (10 min)
    // ========================================================================
    std::cout << "Exercise 5: Pointers and Arrays\n";
    std::cout << "--------------------------------\n";

    int arr[5] = {10, 20, 30, 40, 50};

    // TODO 5.1: Array name is a pointer to first element
    int* arrPtr = arr;  // Same as &arr[0]

    std::cout << "arr:      " << arr << "\n";
    std::cout << "&arr[0]:  " << &arr[0] << "\n";
    std::cout << "arrPtr:   " << arrPtr << "\n";

    // TODO 5.2: Access array elements using pointer
    std::cout << "\nAccessing via pointer:\n";
    std::cout << "*arrPtr:       " << *arrPtr << " (arr[0])\n";
    std::cout << "*(arrPtr + 1): " << *(arrPtr + 1) << " (arr[1])\n";
    std::cout << "*(arrPtr + 2): " << *(arrPtr + 2) << " (arr[2])\n";

    // TODO 5.3: Pointer arithmetic
    std::cout << "\nPointer arithmetic:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "arrPtr[" << i << "] = " << arrPtr[i]
                  << " (address: " << (arrPtr + i) << ")\n";
    }

    // TODO 5.4: Difference between pointers
    int* first = &arr[0];
    int* last = &arr[4];
    std::cout << "\nDistance between elements: " << (last - first) << " elements\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 6: Pointer to Pointer (10 min)
    // ========================================================================
    std::cout << "Exercise 6: Pointer to Pointer\n";
    std::cout << "-------------------------------\n";

    int val = 42;
    int* ptr_to_val = &val;
    int** ptr_to_ptr = &ptr_to_val;

    std::cout << "val:           " << val << "\n";
    std::cout << "&val:          " << &val << "\n";
    std::cout << "ptr_to_val:    " << ptr_to_val << " (holds &val)\n";
    std::cout << "*ptr_to_val:   " << *ptr_to_val << " (value of val)\n";
    std::cout << "&ptr_to_val:   " << &ptr_to_val << "\n";
    std::cout << "ptr_to_ptr:    " << ptr_to_ptr << " (holds &ptr_to_val)\n";
    std::cout << "*ptr_to_ptr:   " << *ptr_to_ptr << " (value of ptr_to_val = &val)\n";
    std::cout << "**ptr_to_ptr:  " << **ptr_to_ptr << " (value of val)\n";

    // TODO 6.1: Modify value through double pointer
    **ptr_to_ptr = 100;
    std::cout << "\nAfter **ptr_to_ptr = 100:\n";
    std::cout << "val = " << val << "\n";

    // TODO 6.2: Common use: arrays of strings
    const char* names[] = {"Alice", "Bob", "Charlie"};
    const char** namePtr = names;

    std::cout << "\nArray of strings:\n";
    for (int i = 0; i < 3; i++) {
        std::cout << names[i] << "\n";
    }

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 7: Const Pointers (10 min)
    // ========================================================================
    std::cout << "Exercise 7: Const Pointers\n";
    std::cout << "--------------------------\n";

    int a = 10, b = 20;

    // TODO 7.1: Pointer to const (can't change value, can change pointer)
    const int* ptr1 = &a;
    // *ptr1 = 30;  // ERROR: can't modify value
    ptr1 = &b;      // OK: can change what it points to
    std::cout << "Pointer to const: " << *ptr1 << "\n";

    // TODO 7.2: Const pointer (can change value, can't change pointer)
    int* const ptr2 = &a;
    *ptr2 = 30;     // OK: can modify value
    // ptr2 = &b;   // ERROR: can't change what it points to
    std::cout << "Const pointer: " << *ptr2 << "\n";

    // TODO 7.3: Const pointer to const (can't change either)
    const int* const ptr3 = &a;
    // *ptr3 = 40;  // ERROR: can't modify value
    // ptr3 = &b;   // ERROR: can't change what it points to
    std::cout << "Const pointer to const: " << *ptr3 << "\n";

    // Mnemonic: Read right to left
    // const int* p     → pointer to const int
    // int* const p     → const pointer to int
    // const int* const p → const pointer to const int

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 8: Common Pointer Pitfalls (10 min)
    // ========================================================================
    std::cout << "Exercise 8: Common Pitfalls\n";
    std::cout << "----------------------------\n";

    // TODO 8.1: Uninitialized pointers
    // int* bad;           // Contains garbage
    // *bad = 10;          // CRASH! Undefined behavior
    int* good = nullptr;   // Safe initialization
    if (good) *good = 10;  // Won't execute

    // TODO 8.2: Dangling pointer (pointing to deallocated memory)
    // int* p = new int(42);
    // delete p;
    // *p = 50;  // DANGLING POINTER! Undefined behavior
    // p = nullptr;  // Good practice after delete

    // TODO 8.3: Memory leak (lost pointer)
    // int* leak = new int(42);
    // leak = new int(50);  // Previous memory leaked!

    // TODO 8.4: Double delete
    // int* p = new int(42);
    // delete p;
    // delete p;  // UNDEFINED BEHAVIOR!

    // TODO 8.5: Comparing pointers of different types
    // int* pi = ...;
    // double* pd = ...;
    // if (pi == pd) { }  // Usually a mistake

    std::cout << "Various pointer pitfalls demonstrated (see comments)\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 9: Practical Applications (5 min)
    // ========================================================================
    std::cout << "Exercise 9: Practical Applications\n";
    std::cout << "-----------------------------------\n";

    // TODO 9.1: Swap two variables using pointers
    int x1 = 5, y1 = 10;
    int* px = &x1;
    int* py = &y1;

    std::cout << "Before swap: x=" << x1 << ", y=" << y1 << "\n";

    // Swap using pointers
    int temp = *px;
    *px = *py;
    *py = temp;

    std::cout << "After swap:  x=" << x1 << ", y=" << y1 << "\n";

    // TODO 9.2: Find max in array using pointer
    int numbers[] = {23, 45, 12, 67, 34};
    int* pNum = numbers;
    int maxVal = *pNum;

    for (int i = 1; i < 5; i++) {
        if (*(pNum + i) > maxVal) {
            maxVal = *(pNum + i);
        }
    }

    std::cout << "\nMax value: " << maxVal << "\n";

    // TODO 9.3: Reverse array using pointers
    int toReverse[] = {1, 2, 3, 4, 5};
    int* left = toReverse;
    int* right = toReverse + 4;

    std::cout << "\nOriginal: ";
    for (int i = 0; i < 5; i++) std::cout << toReverse[i] << " ";

    while (left < right) {
        int tmp = *left;
        *left = *right;
        *right = tmp;
        left++;
        right--;
    }

    std::cout << "\nReversed: ";
    for (int i = 0; i < 5; i++) std::cout << toReverse[i] << " ";
    std::cout << "\n";

    std::cout << "\n";

    // ========================================================================
    // CHALLENGE EXERCISES (Optional - 15 min)
    // ========================================================================
    std::cout << "Challenge Exercises\n";
    std::cout << "-------------------\n";

    // CHALLENGE 1: Implement strcpy using pointers
    // void my_strcpy(char* dest, const char* src) {
    //     while (*src != '\0') {
    //         *dest++ = *src++;
    //     }
    //     *dest = '\0';
    // }


    // CHALLENGE 2: Find element in array, return pointer or nullptr
    // int* find(int* arr, int size, int target) {
    //     for (int i = 0; i < size; i++) {
    //         if (arr[i] == target) {
    //             return &arr[i];
    //         }
    //     }
    //     return nullptr;
    // }


    // CHALLENGE 3: Implement pointer-based linked list node
    // struct Node {
    //     int data;
    //     Node* next;
    // };
    //
    // Node* head = new Node{10, nullptr};
    // head->next = new Node{20, nullptr};


    // CHALLENGE 4: Triple pointer (pointer to pointer to pointer)
    // int*** ppp = ...;
    // ***ppp accesses the value


    std::cout << "\n";

    return 0;
}

// ========================================================================
// COMMON INTERVIEW QUESTIONS
// ========================================================================
/*
 * Q1: What is a pointer?
 * A: A pointer is a variable that stores the memory address of another variable.
 *
 *    Key points:
 *    - Pointers have a type (int*, char*, etc.)
 *    - Size is architecture-dependent (4 bytes on 32-bit, 8 bytes on 64-bit)
 *    - Can be dereferenced to access the value at that address
 *    - Enables dynamic memory allocation and efficient data structures
 *
 *    Example:
 *    int x = 42;
 *    int* p = &x;  // p holds the address of x
 *    *p = 50;      // Changes x to 50
 *
 * Q2: What's the difference between & and *?
 * A: & (address-of operator):
 *    - Returns the memory address of a variable
 *    - Example: &x gives the address of x
 *
 *    * (dereference operator):
 *    - Accesses the value at a memory address
 *    - Example: *p gives the value pointed to by p
 *
 *    Also:
 *    * in declaration means "pointer to"
 *    & in declaration means "reference to" (different from address-of!)
 *
 * Q3: What's the difference between NULL and nullptr?
 * A: NULL (C-style):
 *    - Usually defined as 0 or (void*)0
 *    - Not type-safe
 *    - Can cause ambiguity in function overloading
 *
 *    nullptr (C++11):
 *    - Has type nullptr_t
 *    - Type-safe
 *    - Recommended in modern C++
 *
 *    Example:
 *    void f(int x);
 *    void f(int* p);
 *    f(NULL);     // Calls f(int) - wrong!
 *    f(nullptr);  // Calls f(int*) - correct!
 *
 * Q4: What's the difference between these const pointers?
 * A: const int* p     → Pointer to const int
 *    - Can't change *p
 *    - Can change p
 *
 *    int* const p     → Const pointer to int
 *    - Can change *p
 *    - Can't change p
 *
 *    const int* const p → Const pointer to const int
 *    - Can't change *p
 *    - Can't change p
 *
 *    Mnemonic: Read right to left
 *
 * Q5: What's a dangling pointer?
 * A: A pointer that points to memory that has been freed or is no longer valid.
 *
 *    Causes:
 *    1. Deleting memory: delete p; *p = 10; // DANGLING
 *    2. Returning local address: return &local; // DANGLING
 *    3. Object goes out of scope
 *
 *    Prevention:
 *    - Set pointer to nullptr after delete
 *    - Use smart pointers (unique_ptr, shared_ptr)
 *    - Avoid returning addresses of local variables
 *
 * Q6: What's pointer arithmetic?
 * A: Adding/subtracting integers to/from pointers.
 *
 *    int arr[5] = {10, 20, 30, 40, 50};
 *    int* p = arr;
 *
 *    p + 1 points to arr[1] (not +1 byte, but +sizeof(int) bytes!)
 *    *(p + 2) gives arr[2]
 *    p[3] same as *(p + 3) same as arr[3]
 *
 *    Pointer subtraction:
 *    int* p1 = &arr[0];
 *    int* p2 = &arr[3];
 *    p2 - p1 = 3 (number of elements, not bytes)
 *
 * Q7: What's the relationship between pointers and arrays?
 * A: Array name is a constant pointer to first element.
 *
 *    int arr[5];
 *    arr is equivalent to &arr[0]
 *    arr[i] is equivalent to *(arr + i)
 *
 *    Key differences:
 *    - arr is a constant (can't do arr++)
 *    - sizeof(arr) gives total array size
 *    - sizeof(pointer) gives pointer size
 *
 *    When passed to function, array decays to pointer
 *
 * Q8: What's a void pointer?
 * A: Generic pointer that can point to any data type.
 *
 *    void* p;
 *    int x = 42;
 *    double d = 3.14;
 *    p = &x;  // OK
 *    p = &d;  // OK
 *
 *    Limitations:
 *    - Can't dereference directly: *p is ERROR
 *    - Must cast first: *(int*)p
 *    - No pointer arithmetic (size unknown)
 *
 *    Use cases:
 *    - Generic functions (malloc, memcpy)
 *    - Type-agnostic data structures
 */

/*
 * POINTERS IN GPU PROGRAMMING:
 * =============================
 *
 * 1. Host vs Device Pointers:
 *    float* h_data;  // Host pointer (CPU memory)
 *    float* d_data;  // Device pointer (GPU memory)
 *
 *    cudaMalloc(&d_data, size * sizeof(float));
 *
 *    CANNOT mix:
 *    - Can't dereference device pointer on host
 *    - Can't use host pointer in kernel
 *
 * 2. Unified Memory (CUDA 6+):
 *    float* data;
 *    cudaMallocManaged(&data, size * sizeof(float));
 *    // Can use on both CPU and GPU!
 *
 * 3. Pointer Passing to Kernels:
 *    __global__ void kernel(float* data) {
 *        int idx = threadIdx.x;
 *        data[idx] = ...;  // Pointer arithmetic on GPU
 *    }
 *
 * 4. Shared Memory Pointers:
 *    __shared__ float shared[256];
 *    float* s_ptr = shared;  // Pointer to shared memory
 *
 * 5. Constant Memory:
 *    __constant__ float c_data[1024];
 *    // Read-only, cached access
 *
 * 6. Texture Memory (legacy):
 *    Accessed through special pointers/objects
 *    Modern: use cudaTextureObject_t
 *
 * COMPILATION:
 * ============
 * g++ -std=c++17 -Wall -Wextra 01_pointers_basics.cpp -o pointers
 * ./pointers
 *
 * LEARNING CHECKLIST:
 * ===================
 * ☐ Understand what pointers are and why they're needed
 * ☐ Can declare and initialize pointers correctly
 * ☐ Master address-of (&) and dereference (*) operators
 * ☐ Know the difference between NULL and nullptr
 * ☐ Understand const pointers and pointer to const
 * ☐ Can perform pointer arithmetic
 * ☐ Know the relationship between pointers and arrays
 * ☐ Understand pointer-to-pointer concepts
 * ☐ Aware of common pointer pitfalls
 * ☐ Can use pointers for practical problems
 *
 * NEXT STEPS:
 * ===========
 * - Move to 02_references.cpp
 * - Study smart pointers (unique_ptr, shared_ptr)
 * - Learn about function pointers
 * - Understand memory alignment
 * - Practice with CUDA device pointers
 */
