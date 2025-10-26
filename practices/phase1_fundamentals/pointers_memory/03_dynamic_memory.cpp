/*
 * Exercise: Dynamic Memory Allocation
 * Difficulty: Intermediate
 * Time: 45-55 minutes
 * Topics: new/delete, heap vs stack, memory leaks, RAII, smart pointers
 *
 * LEARNING OBJECTIVES:
 * - Understand stack vs heap memory
 * - Master new and delete operators
 * - Learn to prevent memory leaks
 * - Understand ownership and RAII
 * - Practice with smart pointers (unique_ptr, shared_ptr)
 * - Learn common pitfalls and best practices
 *
 * INTERVIEW RELEVANCE:
 * - Memory management is a core C++ interview topic
 * - Memory leaks are common bugs in production code
 * - Understanding RAII is fundamental to modern C++
 * - Smart pointers are best practice in modern code
 * - CUDA requires careful memory management (cudaMalloc/cudaFree)
 */

#include <iostream>
#include <memory>    // For smart pointers
#include <vector>
#include <string>

int main() {
    std::cout << "=== Dynamic Memory Allocation Exercises ===\n\n";

    // ========================================================================
    // EXERCISE 1: Stack vs Heap (5 min)
    // ========================================================================
    std::cout << "Exercise 1: Stack vs Heap\n";
    std::cout << "-------------------------\n";

    // TODO 1.1: Stack allocation (automatic)
    int stackVar = 42;  // Lives on stack, automatically cleaned up
    std::cout << "Stack variable: " << stackVar << "\n";
    std::cout << "Address: " << &stackVar << "\n";

    // TODO 1.2: Heap allocation (dynamic)
    int* heapVar = new int(100);  // Lives on heap, manual cleanup required
    std::cout << "Heap variable: " << *heapVar << "\n";
    std::cout << "Address: " << heapVar << "\n";

    delete heapVar;  // MUST delete manually!

    // TODO 1.3: Key differences
    /*
     * STACK:
     * - Fixed size (usually 1-8 MB)
     * - Fast allocation/deallocation
     * - Automatic cleanup (RAII)
     * - Limited lifetime (scope-based)
     * - Use for: small, fixed-size, short-lived objects
     *
     * HEAP:
     * - Large size (limited by RAM)
     * - Slower allocation/deallocation
     * - Manual cleanup required
     * - Flexible lifetime
     * - Use for: large objects, dynamic size, long lifetime
     */

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 2: new and delete (10 min)
    // ========================================================================
    std::cout << "Exercise 2: new and delete\n";
    std::cout << "---------------------------\n";

    // TODO 2.1: Allocate single object
    int* p1 = new int;       // Uninitialized
    int* p2 = new int(42);   // Initialized to 42
    int* p3 = new int{100};  // C++11 uniform initialization

    std::cout << "*p2 = " << *p2 << "\n";
    std::cout << "*p3 = " << *p3 << "\n";

    delete p1;
    delete p2;
    delete p3;

    // TODO 2.2: Allocate array
    int* arr = new int[5];  // Array of 5 ints

    for (int i = 0; i < 5; i++) {
        arr[i] = i * 10;
    }

    std::cout << "\nArray: ";
    for (int i = 0; i < 5; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << "\n";

    delete[] arr;  // MUST use delete[] for arrays!

    // TODO 2.3: Allocate object
    struct Point {
        int x, y;
        Point(int a, int b) : x(a), y(b) {
            std::cout << "Point constructed: (" << x << ", " << y << ")\n";
        }
        ~Point() {
            std::cout << "Point destroyed: (" << x << ", " << y << ")\n";
        }
    };

    Point* pt = new Point(10, 20);
    std::cout << "Point: (" << pt->x << ", " << pt->y << ")\n";
    delete pt;  // Calls destructor

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 3: Memory Leaks (10 min)
    // ========================================================================
    std::cout << "Exercise 3: Memory Leaks\n";
    std::cout << "------------------------\n";

    // TODO 3.1: Classic memory leak
    {
        int* leak = new int(42);
        // Forgot to delete!
    }  // leak goes out of scope, pointer lost, memory never freed

    // TODO 3.2: Leak due to early return
    // void function() {
    //     int* data = new int[1000];
    //     if (error) return;  // LEAK! Never reached delete
    //     delete[] data;
    // }

    // TODO 3.3: Leak due to exception
    // void riskyFunction() {
    //     int* data = new int[1000];
    //     riskyOperation();  // Throws exception
    //     delete[] data;     // Never executed!
    // }

    // TODO 3.4: Overwriting pointer
    int* ptr = new int(10);
    ptr = new int(20);  // LEAK! Lost pointer to first allocation
    delete ptr;  // Only frees second allocation

    std::cout << "Memory leak examples shown (see comments)\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 4: RAII (Resource Acquisition Is Initialization) (10 min)
    // ========================================================================
    std::cout << "Exercise 4: RAII Pattern\n";
    std::cout << "------------------------\n";

    // TODO 4.1: Manual management (bad)
    {
        int* manual = new int(42);
        // ... use manual ...
        delete manual;  // Easy to forget!
    }

    // TODO 4.2: RAII wrapper (good)
    class IntWrapper {
        int* ptr;
    public:
        IntWrapper(int val) : ptr(new int(val)) {
            std::cout << "Allocated: " << *ptr << "\n";
        }
        ~IntWrapper() {
            std::cout << "Freed: " << *ptr << "\n";
            delete ptr;
        }
        int get() const { return *ptr; }
    };

    {
        IntWrapper wrapped(42);
        std::cout << "Value: " << wrapped.get() << "\n";
    }  // Automatically cleaned up!

    // TODO 4.3: STL containers use RAII
    {
        std::vector<int> vec(1000);  // Allocates memory
        // ... use vec ...
    }  // Automatically freed!

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 5: Smart Pointers - unique_ptr (10 min)
    // ========================================================================
    std::cout << "Exercise 5: unique_ptr\n";
    std::cout << "----------------------\n";

    // TODO 5.1: Basic unique_ptr
    {
        std::unique_ptr<int> up1(new int(42));
        std::unique_ptr<int> up2 = std::make_unique<int>(100);  // Preferred

        std::cout << "*up1 = " << *up1 << "\n";
        std::cout << "*up2 = " << *up2 << "\n";
    }  // Automatically deleted!

    // TODO 5.2: unique_ptr with arrays
    {
        std::unique_ptr<int[]> arr = std::make_unique<int[]>(5);
        for (int i = 0; i < 5; i++) {
            arr[i] = i;
        }
        std::cout << "Array[2] = " << arr[2] << "\n";
    }  // Automatically deleted with delete[]

    // TODO 5.3: Moving unique_ptr (transfer ownership)
    std::unique_ptr<int> up1 = std::make_unique<int>(42);
    std::unique_ptr<int> up2 = std::move(up1);  // Transfer ownership

    // up1 is now nullptr
    std::cout << "up1 is " << (up1 ? "valid" : "null") << "\n";
    std::cout << "up2 is " << (up2 ? "valid" : "null") << "\n";

    // TODO 5.4: Cannot copy unique_ptr
    // std::unique_ptr<int> up3 = up2;  // ERROR! Can't copy
    // std::unique_ptr<int> up3 = std::move(up2);  // OK - move

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 6: Smart Pointers - shared_ptr (10 min)
    // ========================================================================
    std::cout << "Exercise 6: shared_ptr\n";
    std::cout << "----------------------\n";

    // TODO 6.1: Basic shared_ptr
    {
        std::shared_ptr<int> sp1 = std::make_shared<int>(42);
        std::cout << "*sp1 = " << *sp1 << "\n";
        std::cout << "Use count: " << sp1.use_count() << "\n";

        {
            std::shared_ptr<int> sp2 = sp1;  // Share ownership
            std::cout << "After copy - use count: " << sp1.use_count() << "\n";
        }  // sp2 destroyed, but memory still alive

        std::cout << "After inner scope - use count: " << sp1.use_count() << "\n";
    }  // Last shared_ptr destroyed, memory freed

    // TODO 6.2: Shared ownership example
    struct Resource {
        std::string name;
        Resource(std::string n) : name(n) {
            std::cout << "Resource " << name << " created\n";
        }
        ~Resource() {
            std::cout << "Resource " << name << " destroyed\n";
        }
    };

    std::shared_ptr<Resource> res1 = std::make_shared<Resource>("A");
    std::shared_ptr<Resource> res2 = res1;  // Both own it
    std::shared_ptr<Resource> res3 = res1;  // All three own it

    std::cout << "Use count: " << res1.use_count() << "\n";

    res2.reset();  // res2 no longer owns it
    std::cout << "After reset - use count: " << res1.use_count() << "\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 7: Common Pitfalls (5 min)
    // ========================================================================
    std::cout << "Exercise 7: Common Pitfalls\n";
    std::cout << "----------------------------\n";

    // TODO 7.1: Double delete
    // int* p = new int(42);
    // delete p;
    // delete p;  // UNDEFINED BEHAVIOR!

    // TODO 7.2: delete vs delete[]
    // int* arr = new int[10];
    // delete arr;    // WRONG! Should be delete[]
    // delete[] arr;  // Correct

    // TODO 7.3: Memory leak in array of pointers
    int** matrix = new int*[3];
    for (int i = 0; i < 3; i++) {
        matrix[i] = new int[4];
    }

    // Must delete each row first
    for (int i = 0; i < 3; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;

    // TODO 7.4: Accessing deleted memory
    // int* p = new int(42);
    // delete p;
    // std::cout << *p;  // UNDEFINED BEHAVIOR!

    std::cout << "Common pitfalls demonstrated (see comments)\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 8: Practical Applications (5 min)
    // ========================================================================
    std::cout << "Exercise 8: Practical Applications\n";
    std::cout << "-----------------------------------\n";

    // TODO 8.1: Dynamic array with unknown size
    int size;
    std::cout << "Enter array size: ";
    size = 5;  // Hardcoded for demo
    std::cout << size << "\n";

    int* dynArray = new int[size];
    for (int i = 0; i < size; i++) {
        dynArray[i] = i * i;
    }

    std::cout << "Dynamic array: ";
    for (int i = 0; i < size; i++) {
        std::cout << dynArray[i] << " ";
    }
    std::cout << "\n";

    delete[] dynArray;

    // TODO 8.2: Dynamically allocated linked list
    struct Node {
        int data;
        Node* next;
    };

    Node* head = new Node{1, nullptr};
    head->next = new Node{2, nullptr};
    head->next->next = new Node{3, nullptr};

    std::cout << "\nLinked list: ";
    Node* current = head;
    while (current) {
        std::cout << current->data << " ";
        current = current->next;
    }
    std::cout << "\n";

    // Clean up
    while (head) {
        Node* temp = head;
        head = head->next;
        delete temp;
    }

    std::cout << "\n";

    return 0;
}

// ========================================================================
// COMMON INTERVIEW QUESTIONS
// ========================================================================
/*
 * Q1: What's the difference between stack and heap?
 * A: Stack:
 *    - Fixed size (usually 1-8 MB)
 *    - Fast (just move stack pointer)
 *    - Automatic cleanup
 *    - LIFO (last in, first out)
 *    - Used for: local variables, function calls
 *
 *    Heap:
 *    - Large size (limited by RAM)
 *    - Slower (complex allocator)
 *    - Manual cleanup required
 *    - Random access
 *    - Used for: large data, unknown size, shared data
 *
 * Q2: What's the difference between new/delete and malloc/free?
 * A: new/delete (C++):
 *    - Type-safe
 *    - Calls constructors/destructors
 *    - Can be overloaded
 *    - Throws exception on failure
 *
 *    malloc/free (C):
 *    - Not type-safe (returns void*)
 *    - No constructors/destructors
 *    - Cannot be overloaded
 *    - Returns NULL on failure
 *
 *    NEVER mix: don't use free on new, or delete on malloc!
 *
 * Q3: What's a memory leak?
 * A: Memory that was allocated but never freed.
 *
 *    Causes:
 *    - Forgot to delete
 *    - Early return before delete
 *    - Exception before delete
 *    - Lost pointer to allocated memory
 *
 *    Prevention:
 *    - Use smart pointers (unique_ptr, shared_ptr)
 *    - Use RAII pattern
 *    - Use containers (vector, string, etc.)
 *    - Use tools (valgrind, AddressSanitizer)
 *
 * Q4: What's RAII?
 * A: Resource Acquisition Is Initialization
 *
 *    Principle: Tie resource lifetime to object lifetime
 *    - Constructor acquires resource
 *    - Destructor releases resource
 *    - Automatic cleanup via scope
 *
 *    Examples:
 *    - std::vector (manages dynamic array)
 *    - std::unique_ptr (manages pointer)
 *    - std::lock_guard (manages mutex)
 *    - std::fstream (manages file)
 *
 * Q5: What's the difference between unique_ptr and shared_ptr?
 * A: unique_ptr:
 *    - Exclusive ownership
 *    - Cannot be copied
 *    - Can be moved
 *    - Zero overhead
 *    - Use when: single owner
 *
 *    shared_ptr:
 *    - Shared ownership
 *    - Can be copied
 *    - Reference counting
 *    - Small overhead (control block)
 *    - Use when: multiple owners
 *
 *    Default choice: unique_ptr (cheaper)
 *    Use shared_ptr only when truly needed
 *
 * Q6: What's the difference between delete and delete[]?
 * A: delete:
 *    - For single objects allocated with new
 *    - Calls one destructor
 *
 *    delete[]:
 *    - For arrays allocated with new[]
 *    - Calls destructor for each element
 *
 *    MUST match:
 *    - new → delete
 *    - new[] → delete[]
 *
 *    Mismatch = undefined behavior!
 *
 * Q7: How do you prevent double delete?
 * A: Set pointer to nullptr after delete:
 *
 *    delete p;
 *    p = nullptr;
 *
 *    Deleting nullptr is safe (does nothing)
 *    delete nullptr;  // OK
 *
 *    Or use smart pointers (automatic)
 *
 * Q8: What's a dangling pointer?
 * A: Pointer to memory that has been freed.
 *
 *    int* p = new int(42);
 *    delete p;
 *    *p = 50;  // DANGLING! Undefined behavior
 *
 *    Prevention:
 *    - Set to nullptr after delete
 *    - Use smart pointers
 *    - Don't return pointers to local variables
 */

/*
 * DYNAMIC MEMORY IN GPU PROGRAMMING:
 * ===================================
 *
 * 1. CUDA Memory Allocation:
 *    // Host (CPU) memory
 *    float* h_data = new float[size];
 *
 *    // Device (GPU) memory
 *    float* d_data;
 *    cudaMalloc(&d_data, size * sizeof(float));
 *    cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);
 *
 *    // Clean up
 *    cudaFree(d_data);
 *    delete[] h_data;
 *
 * 2. Unified Memory:
 *    float* data;
 *    cudaMallocManaged(&data, size * sizeof(float));
 *    // Can use on both CPU and GPU
 *    cudaFree(data);
 *
 * 3. Pinned Memory (faster transfers):
 *    float* h_pinned;
 *    cudaMallocHost(&h_pinned, size * sizeof(float));
 *    // Faster cudaMemcpy
 *    cudaFreeHost(h_pinned);
 *
 * 4. Memory Pools:
 *    cudaMemPool_t pool;
 *    cudaMemPoolCreate(&pool, &poolProps);
 *    cudaMallocAsync(&d_data, size, pool, stream);
 *    cudaFreeAsync(d_data, stream);
 *
 * 5. Common Pitfalls:
 *    - Forgetting to free device memory
 *    - Mixing host/device pointers
 *    - Not checking cudaMalloc return values
 *    - Memory leaks harder to detect on GPU
 *
 * COMPILATION:
 * ============
 * g++ -std=c++17 -Wall -Wextra 03_dynamic_memory.cpp -o dynmem
 * ./dynmem
 *
 * LEARNING CHECKLIST:
 * ===================
 * ☐ Understand stack vs heap differences
 * ☐ Can use new/delete correctly
 * ☐ Know how to prevent memory leaks
 * ☐ Understand RAII pattern
 * ☐ Can use unique_ptr and shared_ptr
 * ☐ Know difference between delete and delete[]
 * ☐ Aware of double delete and dangling pointers
 * ☐ Understand ownership and lifetime
 *
 * NEXT STEPS:
 * ===========
 * - Move to 04_pointer_arithmetic.cpp
 * - Study weak_ptr
 * - Learn about custom deleters
 * - Understand memory allocators
 * - Practice CUDA memory management
 */
