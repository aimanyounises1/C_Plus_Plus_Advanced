/*
 * Exercise: Structures
 * Difficulty: Beginner
 * Time: 40-50 minutes
 * Topics: struct definition, member access, nested structures, memory layout, AoS vs SoA
 *
 * LEARNING OBJECTIVES:
 * - Understand struct syntax and usage
 * - Learn member access and initialization
 * - Master nested structures
 * - Understand memory layout and padding
 * - Learn AoS vs SoA patterns (critical for GPU!)
 * - Practice structure alignment
 *
 * INTERVIEW RELEVANCE:
 * - Structures are fundamental to data organization
 * - Memory layout questions are common
 * - AoS vs SoA is critical for GPU performance
 * - Understanding padding helps with optimization
 * - Structure passing (by value vs reference) is frequently asked
 */

#include <iostream>
#include <string>
#include <cstring>  // For memset

int main() {
    std::cout << "=== Structures Exercises ===\n\n";

    // ========================================================================
    // EXERCISE 1: Basic Structure Definition (5 min)
    // ========================================================================
    std::cout << "Exercise 1: Basic Structure Definition\n";
    std::cout << "---------------------------------------\n";

    // TODO 1.1: Define a simple structure for a Point
    // struct Point {
    //     int x;
    //     int y;
    // };


    // TODO 1.2: Create and initialize a Point
    // Point p1;
    // p1.x = 10;
    // p1.y = 20;


    // TODO 1.3: Initialize using designated initializers (C++20) or aggregate initialization
    // Point p2 = {30, 40};


    // TODO 1.4: Print the point
    // std::cout << "Point: (" << p1.x << ", " << p1.y << ")\n";


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 2: Structure with Different Data Types (5 min)
    // ========================================================================
    std::cout << "Exercise 2: Mixed Data Types\n";
    std::cout << "-----------------------------\n";

    // TODO 2.1: Define a Student structure
    // struct Student {
    //     int id;
    //     char name[50];
    //     float gpa;
    //     bool isEnrolled;
    // };


    // TODO 2.2: Create and initialize a student
    // Student alice;
    // alice.id = 12345;
    // strcpy(alice.name, "Alice");
    // alice.gpa = 3.8f;
    // alice.isEnrolled = true;


    // TODO 2.3: Print student information


    // TODO 2.4: Create using aggregate initialization
    // Student bob = {12346, "Bob", 3.5f, true};


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 3: Nested Structures (10 min)
    // ========================================================================
    std::cout << "Exercise 3: Nested Structures\n";
    std::cout << "------------------------------\n";

    // TODO 3.1: Define nested structures
    // struct Date {
    //     int day;
    //     int month;
    //     int year;
    // };
    //
    // struct Person {
    //     char name[50];
    //     int age;
    //     Date birthdate;  // Nested structure
    // };


    // TODO 3.2: Create and initialize a person with birthdate
    // Person john;
    // strcpy(john.name, "John");
    // john.age = 25;
    // john.birthdate.day = 15;
    // john.birthdate.month = 6;
    // john.birthdate.year = 1998;


    // TODO 3.3: Print person with birthdate
    // std::cout << john.name << " was born on "
    //           << john.birthdate.month << "/" << john.birthdate.day << "/"
    //           << john.birthdate.year << "\n";


    // TODO 3.4: Initialize using aggregate initialization
    // Person jane = {"Jane", 30, {10, 3, 1993}};


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 4: Arrays of Structures (10 min)
    // ========================================================================
    std::cout << "Exercise 4: Arrays of Structures\n";
    std::cout << "---------------------------------\n";

    // TODO 4.1: Create an array of structures
    // Point points[3] = {
    //     {1, 2},
    //     {3, 4},
    //     {5, 6}
    // };


    // TODO 4.2: Loop through and print all points


    // TODO 4.3: Find the point farthest from origin
    // Distance from origin: sqrt(x^2 + y^2)
    // For simplicity, use x^2 + y^2 (no need for sqrt)


    // TODO 4.4: Calculate average position


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 5: Memory Layout and sizeof (10 min) - IMPORTANT!
    // ========================================================================
    std::cout << "Exercise 5: Memory Layout\n";
    std::cout << "-------------------------\n";

    // TODO 5.1: Check size of different structures
    struct TinyStruct {
        char c;     // 1 byte
    };

    struct SmallStruct {
        char c;     // 1 byte
        int i;      // 4 bytes
    };

    struct PaddedStruct {
        char c;     // 1 byte + 3 bytes padding
        int i;      // 4 bytes
        char d;     // 1 byte + 3 bytes padding
    };

    struct OptimizedStruct {
        int i;      // 4 bytes
        char c;     // 1 byte
        char d;     // 1 byte + 2 bytes padding
    };

    std::cout << "sizeof(TinyStruct): " << sizeof(TinyStruct) << " bytes\n";
    std::cout << "sizeof(SmallStruct): " << sizeof(SmallStruct) << " bytes\n";
    std::cout << "sizeof(PaddedStruct): " << sizeof(PaddedStruct) << " bytes\n";
    std::cout << "sizeof(OptimizedStruct): " << sizeof(OptimizedStruct) << " bytes\n";

    // TODO 5.2: Explain padding
    /*
     * Why padding?
     * - CPU reads memory in aligned chunks (typically 4 or 8 bytes)
     * - Unaligned access is slower (or causes crash on some architectures)
     * - Compiler adds padding to ensure proper alignment
     *
     * Example: PaddedStruct
     * char c   [1 byte] [3 bytes padding]
     * int i    [4 bytes aligned to 4-byte boundary]
     * char d   [1 byte] [3 bytes padding]
     * Total: 12 bytes (not 6!)
     *
     * OptimizedStruct: Reorder fields by size (largest first)
     * int i    [4 bytes]
     * char c   [1 byte]
     * char d   [1 byte] [2 bytes padding]
     * Total: 8 bytes (saved 4 bytes!)
     */

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 6: Passing Structures to Functions (5 min)
    // ========================================================================
    std::cout << "Exercise 6: Function Parameters\n";
    std::cout << "--------------------------------\n";

    // TODO 6.1: Pass by value (makes a copy)
    // void printPoint(Point p) {
    //     std::cout << "(" << p.x << ", " << p.y << ")\n";
    // }


    // TODO 6.2: Pass by reference (no copy, can modify)
    // void movePoint(Point& p, int dx, int dy) {
    //     p.x += dx;
    //     p.y += dy;
    // }


    // TODO 6.3: Pass by const reference (no copy, can't modify)
    // void printPointConst(const Point& p) {
    //     std::cout << "(" << p.x << ", " << p.y << ")\n";
    //     // p.x = 10;  // ERROR: can't modify
    // }


    // TODO 6.4: Return a structure
    // Point createPoint(int x, int y) {
    //     Point p = {x, y};
    //     return p;
    // }


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 7: AoS vs SoA (10 min) - GPU CRITICAL!
    // ========================================================================
    std::cout << "Exercise 7: AoS vs SoA\n";
    std::cout << "----------------------\n";

    // Array of Structures (AoS)
    struct Particle_AoS {
        float x, y, z;     // Position
        float vx, vy, vz;  // Velocity
    };

    Particle_AoS particles_aos[1000];

    // Initialize (example)
    for (int i = 0; i < 1000; i++) {
        particles_aos[i].x = i * 0.1f;
        particles_aos[i].y = i * 0.2f;
        particles_aos[i].z = i * 0.3f;
        particles_aos[i].vx = 1.0f;
        particles_aos[i].vy = 2.0f;
        particles_aos[i].vz = 3.0f;
    }

    // Memory layout: [x,y,z,vx,vy,vz][x,y,z,vx,vy,vz][x,y,z,vx,vy,vz]...
    std::cout << "AoS: Particle 0 at " << &particles_aos[0] << "\n";
    std::cout << "AoS: Particle 1 at " << &particles_aos[1] << "\n";
    std::cout << "AoS: Distance between particles: "
              << (char*)&particles_aos[1] - (char*)&particles_aos[0] << " bytes\n";

    // Structure of Arrays (SoA) - Better for GPU!
    struct Particles_SoA {
        float x[1000];
        float y[1000];
        float z[1000];
        float vx[1000];
        float vy[1000];
        float vz[1000];
    };

    Particles_SoA particles_soa;

    // Initialize
    for (int i = 0; i < 1000; i++) {
        particles_soa.x[i] = i * 0.1f;
        particles_soa.y[i] = i * 0.2f;
        particles_soa.z[i] = i * 0.3f;
        particles_soa.vx[i] = 1.0f;
        particles_soa.vy[i] = 2.0f;
        particles_soa.vz[i] = 3.0f;
    }

    // Memory layout: [x0,x1,x2,...][y0,y1,y2,...][z0,z1,z2,...]...
    std::cout << "\nSoA: x array at " << particles_soa.x << "\n";
    std::cout << "SoA: y array at " << particles_soa.y << "\n";

    // TODO 7.1: Benchmark: Update all x positions
    // AoS: Access is strided (skip vx,vy,vz to get to next x)
    // SoA: Access is contiguous (all x values are next to each other)


    // TODO 7.2: Explain why SoA is better for GPU
    /*
     * SoA Advantages for GPU:
     * 1. Memory Coalescing: Adjacent threads access adjacent memory
     *    - Thread 0 reads x[0], Thread 1 reads x[1], etc.
     *    - GPU can combine these into a single memory transaction
     * 2. Cache Efficiency: Better spatial locality
     * 3. Vectorization: Easy to use SIMD instructions
     * 4. Component Operations: Can process one component at a time
     *
     * AoS Disadvantages for GPU:
     * 1. Strided Access: Thread 0 reads x at offset 0, Thread 1 at offset 24
     * 2. Wasted Bandwidth: Load entire structure even if only need one field
     * 3. Poor Cache Usage: Load unnecessary data
     *
     * When to use AoS:
     * - CPU code with object-oriented design
     * - Access all fields together frequently
     * - Small datasets
     *
     * When to use SoA:
     * - GPU programming (CUDA, compute shaders)
     * - SIMD/vectorization on CPU
     * - Large datasets with component-wise operations
     */

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 8: Practical Applications (5 min)
    // ========================================================================
    std::cout << "Exercise 8: Practical Applications\n";
    std::cout << "-----------------------------------\n";

    // TODO 8.1: Define a Rectangle structure
    // struct Rectangle {
    //     Point topLeft;
    //     int width;
    //     int height;
    // };


    // TODO 8.2: Calculate area
    // int area = rect.width * rect.height;


    // TODO 8.3: Define a Book structure
    // struct Book {
    //     char title[100];
    //     char author[50];
    //     int year;
    //     float price;
    // };


    // TODO 8.4: Create an array of books and find the most expensive


    std::cout << "\n";

    // ========================================================================
    // CHALLENGE EXERCISES (Optional - 15 min)
    // ========================================================================
    std::cout << "Challenge Exercises\n";
    std::cout << "-------------------\n";

    // CHALLENGE 1: Implement a simple linked list node
    // struct Node {
    //     int data;
    //     Node* next;
    // };


    // CHALLENGE 2: Create a structure for a 3D vector with operations
    // struct Vector3 {
    //     float x, y, z;
    // };
    // Implement: add, subtract, dot product, magnitude


    // CHALLENGE 3: Implement structure comparison
    // bool operator==(const Point& a, const Point& b)


    // CHALLENGE 4: Pack a structure tightly (no padding)
    // Use #pragma pack(1) or __attribute__((packed))


    // CHALLENGE 5: Implement a union
    // union can hold different types in the same memory location


    std::cout << "\n";

    return 0;
}

// ========================================================================
// COMMON INTERVIEW QUESTIONS
// ========================================================================
/*
 * Q1: What's the difference between struct and class in C++?
 * A: In C++, the only difference is default access:
 *    - struct: members are public by default
 *    - class: members are private by default
 *
 *    Both can have:
 *    - Member functions
 *    - Constructors/destructors
 *    - Inheritance
 *    - Access specifiers (public, private, protected)
 *
 *    Convention: Use struct for Plain Old Data (POD) types
 *                Use class for objects with methods and encapsulation
 *
 * Q2: What is structure padding and why does it exist?
 * A: Padding is extra bytes added by the compiler to align structure
 *    members to natural boundaries (typically 4 or 8 bytes).
 *
 *    Why?
 *    - CPUs read memory in aligned chunks
 *    - Unaligned access is slower (or illegal on some architectures)
 *    - Better performance at the cost of memory
 *
 *    How to minimize:
 *    - Order members from largest to smallest
 *    - Group similar-sized members together
 *    - Use #pragma pack(1) for tight packing (slower access!)
 *
 * Q3: What is Array of Structures (AoS) vs Structure of Arrays (SoA)?
 * A: AoS: struct Point {float x,y,z;}; Point pts[1000];
 *    - Natural OOP style
 *    - Good when accessing all fields together
 *    - Poor for SIMD/GPU (strided memory access)
 *
 *    SoA: struct Points {float x[1000], y[1000], z[1000];};
 *    - Better for SIMD/GPU (contiguous memory access)
 *    - Memory coalescing on GPU
 *    - Good for component-wise operations
 *
 *    GPU Example:
 *    AoS: Thread i reads pts[i].x (stride: sizeof(Point))
 *    SoA: Thread i reads x[i] (stride: sizeof(float)) - COALESCED!
 *
 * Q4: How do you pass a structure to a function?
 * A: Three ways:
 *
 *    1. By value (makes a copy):
 *       void func(Point p) { ... }
 *       - Safe (can't modify original)
 *       - Slow for large structures (copies all data)
 *
 *    2. By reference (no copy, can modify):
 *       void func(Point& p) { ... }
 *       - Fast (no copy)
 *       - Can modify original
 *
 *    3. By const reference (no copy, read-only):
 *       void func(const Point& p) { ... }
 *       - Fast (no copy)
 *       - Safe (can't modify)
 *       - BEST PRACTICE for large read-only structures
 *
 * Q5: What is a union and when would you use it?
 * A: Union: All members share the same memory location
 *    union Data {
 *        int i;
 *        float f;
 *        char c;
 *    };
 *    sizeof(Data) = max(sizeof(int), sizeof(float), sizeof(char))
 *
 *    Use cases:
 *    - Type punning (interpret same bytes as different types)
 *    - Memory-constrained systems
 *    - Implementing variants/tagged unions
 *
 *    Caution: Only one member is valid at a time!
 *
 * Q6: What is struct alignment?
 * A: The compiler aligns structures to natural boundaries:
 *    - char: 1-byte aligned
 *    - short: 2-byte aligned
 *    - int, float: 4-byte aligned
 *    - long, double, pointer: 8-byte aligned (on 64-bit)
 *
 *    The structure itself is aligned to the largest member alignment.
 *
 *    Example:
 *    struct S { char c; int i; };
 *    - 'c' at offset 0
 *    - 3 bytes padding
 *    - 'i' at offset 4 (4-byte aligned)
 *    - Total: 8 bytes
 *
 * Q7: How can you initialize a structure in C++?
 * A: Several ways:
 *
 *    1. Member-wise:
 *       Point p;
 *       p.x = 10;
 *       p.y = 20;
 *
 *    2. Aggregate initialization:
 *       Point p = {10, 20};
 *
 *    3. Designated initializers (C++20):
 *       Point p = {.x = 10, .y = 20};
 *
 *    4. Constructor (if defined):
 *       Point p(10, 20);
 *
 * Q8: What are nested structures?
 * A: A structure that contains another structure as a member:
 *
 *    struct Date { int day, month, year; };
 *    struct Person {
 *        char name[50];
 *        Date birthdate;  // Nested
 *    };
 *
 *    Access: person.birthdate.day
 *
 *    Memory: Date is embedded inside Person (not a pointer!)
 */

/*
 * STRUCTURES IN GPU PROGRAMMING:
 * ===============================
 *
 * 1. Prefer SoA over AoS:
 *    Bad:
 *    struct Particle { float x, y, z; };
 *    Particle particles[1000000];
 *
 *    Good:
 *    struct Particles {
 *        float *x, *y, *z;
 *    };
 *    cudaMalloc(&particles.x, N * sizeof(float));
 *
 * 2. Memory Alignment:
 *    Use __align__(N) for proper GPU alignment:
 *    struct __align__(16) Vector4 {
 *        float x, y, z, w;
 *    };
 *
 * 3. Constant Structures:
 *    Small structures can go in constant memory:
 *    __constant__ struct Config {
 *        int width;
 *        int height;
 *        float threshold;
 *    } config;
 *
 * 4. Avoid Large Structures:
 *    Passing large structs by value uses register pressure
 *    Use pointers or split into separate parameters
 *
 * 5. Coalescing Example:
 *    // BAD: Strided access
 *    __global__ void bad(Particle* p) {
 *        int i = threadIdx.x;
 *        float x = p[i].x;  // Thread 0 at offset 0, thread 1 at offset 12
 *    }
 *
 *    // GOOD: Coalesced access
 *    __global__ void good(float* x) {
 *        int i = threadIdx.x;
 *        float val = x[i];  // Thread 0 at offset 0, thread 1 at offset 4
 *    }
 *
 * COMPILATION:
 * ============
 * g++ -std=c++17 -Wall -Wextra 01_structures.cpp -o structures
 * ./structures
 *
 * LEARNING CHECKLIST:
 * ===================
 * ☐ Can define and use structures
 * ☐ Understand member access
 * ☐ Know how to create nested structures
 * ☐ Understand memory layout and padding
 * ☐ Can optimize structure size
 * ☐ Know when to use AoS vs SoA
 * ☐ Understand alignment requirements
 * ☐ Can pass structures to functions efficiently
 * ☐ Aware of GPU memory coalescing
 *
 * NEXT STEPS:
 * ===========
 * - Move to 02_enumerations.cpp
 * - Study memory alignment in detail
 * - Practice SoA transformations
 * - Learn about structure packing
 * - Understand GPU coalescing patterns
 */
