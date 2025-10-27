/*
 * ============================================================================
 * Exercise: Header Files and Modularity in C++
 * ============================================================================
 * Difficulty: Intermediate
 * Time: 45-60 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Understand the purpose of header files (.h/.hpp)
 * 2. Master include guards and #pragma once
 * 3. Separate interface from implementation
 * 4. Organize large projects into modules
 * 5. Avoid common header file pitfalls (circular dependencies, multiple definitions)
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Large codebases require good modularity
 * - Understanding compilation model is essential
 * - Header-only libraries common in C++ (templates, GPU code)
 * - Include optimization affects build times
 * - Separating device and host code in CUDA requires header knowledge
 *
 * PREREQUISITES:
 * - Function declaration and definition
 * - Basic understanding of compilation process
 * - File system navigation
 * ============================================================================
 */

/*
 * ============================================================================
 * THEORY: Why Header Files?
 * ============================================================================
 *
 * C++ uses a compilation model that requires declarations before use:
 *
 * 1. COMPILATION UNITS: Each .cpp file is compiled separately into an object file
 *    - Compiler needs to know function signatures (declarations) to compile
 *    - Doesn't need to know implementation details
 *
 * 2. LINKING: Object files are linked together to form executable
 *    - Linker resolves function calls to actual implementations
 *    - One Definition Rule (ODR): Each function can be defined only once
 *
 * 3. HEADER FILES: Contain declarations that can be included in multiple files
 *    - .h or .hpp extension (convention)
 *    - Included via #include directive
 *    - Must be guarded to prevent multiple inclusion
 *
 * ============================================================================
 * BASIC STRUCTURE:
 * ============================================================================
 *
 * math_utils.h (header - interface):
 * -----------------------------------
 * #ifndef MATH_UTILS_H
 * #define MATH_UTILS_H
 *
 * int add(int a, int b);           // Declaration
 * double square(double x);         // Declaration
 *
 * #endif // MATH_UTILS_H
 *
 *
 * math_utils.cpp (implementation):
 * -----------------------------------
 * #include "math_utils.h"
 *
 * int add(int a, int b) {          // Definition
 *     return a + b;
 * }
 *
 * double square(double x) {        // Definition
 *     return x * x;
 * }
 *
 *
 * main.cpp (usage):
 * -----------------------------------
 * #include "math_utils.h"          // Get declarations
 *
 * int main() {
 *     int sum = add(5, 3);         // Can call because declared
 *     return 0;
 * }
 *
 * ============================================================================
 */

#include <iostream>
#include <string>
#include <vector>

using namespace std;

/*
 * ============================================================================
 * EXERCISE 1: Basic Header File Creation (15 minutes)
 * ============================================================================
 * Learn to separate declarations and definitions
 */

// TODO 1.1: Create a header file "calculator.h"
// Create a new file called calculator.h with the following content:
//
// #ifndef CALCULATOR_H
// #define CALCULATOR_H
//
// double add(double a, double b);
// double subtract(double a, double b);
// double multiply(double a, double b);
// double divide(double a, double b);
//
// #endif // CALCULATOR_H


// TODO 1.2: Create implementation file "calculator.cpp"
// Create calculator.cpp and implement the functions:
// - add: return a + b
// - subtract: return a - b
// - multiply: return a * b
// - divide: return a / b (check for division by zero!)


// TODO 1.3: Use the header in main
// Include calculator.h and test all functions
// Compile with: g++ -std=c++17 03_header_files.cpp calculator.cpp -o header_demo


/*
 * ============================================================================
 * EXERCISE 2: Include Guards vs #pragma once (10 minutes)
 * ============================================================================
 * Understand mechanisms to prevent multiple inclusion
 */

// TODO 2.1: Create "point.h" with traditional include guards
// Traditional include guards:
// #ifndef POINT_H
// #define POINT_H
//
// struct Point {
//     double x, y;
// };
//
// double distance(const Point& p1, const Point& p2);
//
// #endif // POINT_H


// TODO 2.2: Create "circle.h" with #pragma once
// Modern alternative:
// #pragma once
//
// #include "point.h"
//
// struct Circle {
//     Point center;
//     double radius;
// };
//
// double area(const Circle& c);


// TODO 2.3: Test both approaches
// Include both headers in your main file
// Verify no compilation errors occur even if included multiple times


/*
 * ============================================================================
 * EXERCISE 3: Class Header Organization (15 minutes)
 * ============================================================================
 * Learn best practices for organizing class definitions
 */

// TODO 3.1: Create "student.h" with class declaration
// #ifndef STUDENT_H
// #define STUDENT_H
//
// #include <string>
// #include <vector>
//
// class Student {
// private:
//     std::string name;
//     int id;
//     std::vector<double> grades;
//
// public:
//     Student(const std::string& name, int id);
//
//     void addGrade(double grade);
//     double getAverageGrade() const;
//     std::string getName() const;
//     int getId() const;
// };
//
// #endif // STUDENT_H


// TODO 3.2: Create "student.cpp" with implementations
// Implement all member functions in student.cpp
// Remember to include "student.h" at the top


// TODO 3.3: Use the Student class
// Create several Student objects, add grades, print averages


/*
 * ============================================================================
 * EXERCISE 4: Forward Declarations (10 minutes)
 * ============================================================================
 * Reduce compilation dependencies with forward declarations
 */

// TODO 4.1: Create headers with forward declarations
// Create "engine.h":
// #pragma once
// class Car;  // Forward declaration
//
// class Engine {
// public:
//     void start();
//     void attachToCar(Car* car);
// private:
//     Car* attachedCar;
// };
//
// Create "car.h":
// #pragma once
// #include "engine.h"
//
// class Car {
// public:
//     Car(const std::string& model);
//     Engine* getEngine();
// private:
//     std::string model;
//     Engine engine;
// };


// TODO 4.2: Implement in .cpp files
// Implement Engine and Car classes
// Note: .cpp files need full definitions, so include both headers


// TODO 4.3: Compare with full inclusion
// Try replacing forward declaration with #include "car.h" in engine.h
// Observe circular dependency issue!


/*
 * ============================================================================
 * EXERCISE 5: Inline Functions and Templates (15 minutes)
 * ============================================================================
 * Understand when definitions must be in headers
 */

// TODO 5.1: Create "array_utils.h" with inline functions
// #pragma once
// #include <vector>
//
// // Inline function - definition must be in header
// inline int sum(const std::vector<int>& arr) {
//     int total = 0;
//     for (int val : arr) total += val;
//     return total;
// }
//
// // Template function - definition must be in header
// template<typename T>
// T maximum(const std::vector<T>& arr) {
//     T max = arr[0];
//     for (const T& val : arr) {
//         if (val > max) max = val;
//     }
//     return max;
// }


// TODO 5.2: Use inline and template functions
// Include array_utils.h and test both functions
// Try with different types for maximum (int, double, string)


// TODO 5.3: Understand why definitions are in header
// Templates and inline functions need to be visible at compile time
// This is why they're typically in headers


/*
 * ============================================================================
 * CHALLENGE EXERCISES (Optional - 20 minutes)
 * ============================================================================
 */

// CHALLENGE 1: Create a Math Library Module
// Design a complete math library with:
// - math_lib.h: Master header that includes all sub-headers
// - vector2d.h/.cpp: 2D vector operations (add, subtract, dot product, etc.)
// - vector3d.h/.cpp: 3D vector operations
// - matrix.h/.cpp: 2x2 and 3x3 matrix operations
// - constants.h: Mathematical constants (PI, E, etc.)
// Demonstrate proper organization and include relationships


// CHALLENGE 2: Detect and Fix Circular Dependencies
// Create intentionally circular dependencies:
// - a.h includes b.h
// - b.h includes a.h
// Observe compilation error
// Fix using forward declarations


// CHALLENGE 3: Header-Only Library
// Create a complete header-only utility library
// Requirements:
// - All code in .h file (no .cpp)
// - Uses inline, constexpr, or templates
// - Provides string utilities (trim, split, join, etc.)
// - Include guards or #pragma once
// - Comprehensive documentation


/*
 * ============================================================================
 * PRACTICAL APPLICATION: Project Structure (15 minutes)
 * ============================================================================
 */

// APPLICATION 1: Organize a Game Engine Module
// Create a simple game engine structure:
//
// include/
//   game_engine.h      (master header)
//   entity.h           (game entity base class)
//   player.h           (player class)
//   enemy.h            (enemy class)
//   renderer.h         (rendering interface)
//
// src/
//   entity.cpp
//   player.cpp
//   enemy.cpp
//   renderer.cpp
//   main.cpp
//
// Implement basic functionality and demonstrate modular compilation


/*
 * ============================================================================
 * COMMON INTERVIEW QUESTIONS & ANSWERS
 * ============================================================================
 *
 * Q1: What is the difference between #include <> and #include ""?
 * A: - #include <file>: Searches system/standard library directories first
 *    - #include "file": Searches current directory first, then system directories
 *    Convention:
 *    - Use <> for standard library and external libraries: #include <iostream>
 *    - Use "" for project headers: #include "myheader.h"
 *
 * Q2: What are include guards and why are they necessary?
 * A: Include guards prevent multiple inclusion of the same header:
 *    #ifndef MYHEADER_H
 *    #define MYHEADER_H
 *    // ... header content ...
 *    #endif
 *
 *    Why necessary:
 *    - Headers can be included by multiple files
 *    - Without guards, definitions would be duplicated
 *    - Causes "redefinition" errors
 *    - Guards ensure content is processed only once per compilation unit
 *
 *    Modern alternative: #pragma once (non-standard but widely supported)
 *
 * Q3: What is the One Definition Rule (ODR)?
 * A: The One Definition Rule states:
 *    - A variable or function can be declared multiple times
 *    - But can only be defined once across all compilation units
 *    - Exceptions: inline functions, templates, constexpr (can be in headers)
 *
 *    Violations cause linker errors:
 *    - "multiple definition of..." or "redefinition of..."
 *
 *    Solution:
 *    - Declarations in headers (.h)
 *    - Definitions in source files (.cpp)
 *    - Use inline or static for header-defined functions
 *
 * Q4: When should functions be defined in headers?
 * A: Functions should be in headers when:
 *    1. Template functions: Compiler needs full definition
 *    2. Inline functions: Small, frequently called functions
 *    3. constexpr functions: Compile-time evaluation
 *    4. Header-only libraries: Convenience (no linking needed)
 *
 *    Mark with appropriate keywords:
 *    - inline: Suggests inlining, allows multiple definitions
 *    - constexpr: Compile-time evaluation
 *    - template: Compiler generates code per instantiation
 *
 *    Trade-offs:
 *    - Pro: No separate .cpp, easier to use
 *    - Con: Longer compilation times, larger object files
 *
 * Q5: What are forward declarations and when should you use them?
 * A: Forward declaration declares a type without full definition:
 *    class MyClass;  // Forward declaration
 *
 *    Use when:
 *    - Only need pointer or reference to the type
 *    - Want to break circular dependencies
 *    - Want to reduce header dependencies (faster compilation)
 *
 *    Cannot use forward declarations for:
 *    - Creating objects (compiler needs size)
 *    - Accessing members
 *    - Deriving from the class
 *
 *    Example:
 *    // header.h
 *    class B;  // Forward declaration
 *    class A {
 *        B* ptr;  // OK: just a pointer
 *    };
 *
 * Q6: Explain the compilation and linking process.
 * A: Process steps:
 *    1. Preprocessing:
 *       - Process #include, #define, #ifdef, etc.
 *       - Result: Pure C++ code
 *    2. Compilation:
 *       - Each .cpp file compiled to object file (.o or .obj)
 *       - Compiler checks syntax, types
 *       - Generates machine code
 *    3. Linking:
 *       - Combines object files
 *       - Resolves function calls to definitions
 *       - Creates executable
 *
 *    Headers role:
 *    - Copied into each .cpp via #include (preprocessing)
 *    - Provide declarations so compiler knows function signatures
 *    - Not compiled directly (only through .cpp files)
 *
 * Q7: What causes circular dependencies and how do you fix them?
 * A: Circular dependency occurs when:
 *    - a.h includes b.h
 *    - b.h includes a.h
 *    Result: Infinite inclusion (without guards) or compilation failure
 *
 *    Solutions:
 *    1. Forward declarations: Declare class instead of including header
 *       // a.h
 *       class B;  // Instead of #include "b.h"
 *       class A { B* ptr; };
 *
 *    2. Move implementation to .cpp: Include headers in .cpp, not .h
 *
 *    3. Redesign: Often indicates poor architecture, consider refactoring
 *
 *    4. Break dependency: Create interface/base class
 *
 * Q8: What is the difference between .h and .hpp?
 * A: Both are header files, difference is convention:
 *    - .h: Traditional C/C++ header extension
 *    - .hpp: Explicitly indicates C++ (not C-compatible)
 *
 *    Some projects use:
 *    - .h for C-compatible headers
 *    - .hpp for C++-only headers (templates, classes)
 *
 *    Technical: No compiler difference, purely organizational
 *
 *    Also seen:
 *    - .hxx, .h++, .hh: C++ headers (rare)
 *    - .inl: Inline implementation files
 *
 * Q9: How do you organize a large C++ project?
 * A: Best practices:
 *
 *    Directory structure:
 *    project/
 *      include/          # Public headers
 *        module1/
 *        module2/
 *      src/              # Implementation files
 *        module1/
 *        module2/
 *      tests/            # Test files
 *      build/            # Build artifacts (git ignored)
 *
 *    Header organization:
 *    - One class per header (usually)
 *    - Module-level master headers
 *    - Separate public and internal headers
 *
 *    Include strategy:
 *    - Include what you use
 *    - Prefer forward declarations in headers
 *    - Use angle brackets for external: <iostream>
 *    - Use quotes for internal: "myheader.h"
 *
 *    Naming:
 *    - Clear, descriptive names
 *    - Consistent casing (snake_case or CamelCase)
 *    - Match .h and .cpp names
 *
 * Q10: What is a precompiled header and when should you use it?
 * A: Precompiled header (PCH): Compiled version of commonly included headers
 *
 *    How it works:
 *    1. Identify rarely-changing headers (STL, libraries)
 *    2. Compile them once into binary format
 *    3. Reuse compiled result instead of recompiling
 *
 *    Benefits:
 *    - Faster compilation (10-50% speedup on large projects)
 *    - Especially beneficial for large header-only libraries
 *
 *    When to use:
 *    - Large projects with long build times
 *    - Many files include same headers
 *    - Headers rarely change
 *
 *    Caveats:
 *    - Setup complexity
 *    - May hide include dependencies
 *    - Must be first include in each .cpp
 *
 *    Example (MSVC): stdafx.h (precompiled header)
 *
 * ============================================================================
 * GPU/CUDA RELEVANCE FOR NVIDIA INTERVIEW:
 * ============================================================================
 *
 * 1. CUDA Header Organization:
 *    - Separate .cu (device code) and .cpp (host code)
 *    - .cuh for CUDA headers
 *    - __host__, __device__, __global__ decorators in headers
 *    - Template-heavy for generic kernel code
 *
 * 2. Header-Only Libraries:
 *    - Common in GPU programming (CUB, Thrust)
 *    - All template code, must be in headers
 *    - Allows compiler to optimize across inline boundaries
 *
 * 3. Compilation Model:
 *    - NVCC separates device and host code
 *    - Device code compiled to PTX/SASS
 *    - Understanding headers crucial for proper separation
 *
 * 4. Include Optimization:
 *    - Large projects (like game engines using GPU)
 *    - Minimize header dependencies for faster builds
 *    - Forward declarations reduce recompilation
 *
 * 5. API Design:
 *    - Nvidia libraries use clear header organization
 *    - Study cudart.h, cuda.h for professional examples
 *    - Separation of public API from implementation
 *
 * ============================================================================
 * COMPILATION & EXECUTION:
 * ============================================================================
 *
 * For this exercise, you'll create multiple files:
 *
 * Single file compilation:
 *   g++ -std=c++17 03_header_files.cpp -o header_demo
 *
 * Multiple file compilation:
 *   g++ -std=c++17 main.cpp calculator.cpp student.cpp -o myprogram
 *
 * Or compile separately then link:
 *   g++ -std=c++17 -c main.cpp        # Creates main.o
 *   g++ -std=c++17 -c calculator.cpp  # Creates calculator.o
 *   g++ -std=c++17 main.o calculator.o -o myprogram
 *
 * Using Make (advanced):
 *   Create Makefile for automatic compilation
 *
 * ============================================================================
 * EXPECTED OUTPUT (after completing exercises):
 * ============================================================================
 *
 * Should demonstrate:
 * - Successful multi-file compilation
 * - Calculator operations working across files
 * - Student class operations
 * - Template and inline function usage
 * - No multiple definition errors
 * - Modular, organized code structure
 *
 * ============================================================================
 * LEARNING CHECKLIST:
 * ============================================================================
 *
 * After completing these exercises, you should be able to:
 * ☐ Create proper header files with include guards
 * ☐ Separate declarations from definitions
 * ☐ Compile multi-file projects
 * ☐ Use forward declarations to reduce dependencies
 * ☐ Organize code into logical modules
 * ☐ Understand when to put definitions in headers
 * ☐ Avoid circular dependencies
 * ☐ Apply the One Definition Rule correctly
 * ☐ Structure large projects professionally
 * ☐ Optimize build times with good header organization
 *
 * ============================================================================
 * NEXT STEPS:
 * ============================================================================
 *
 * 1. ✅ Phase 1 Complete! You've finished all fundamentals
 * 2. Move to Phase 2: Object-Oriented Programming
 * 3. Study open-source C++ projects to see real header organization
 * 4. Practice creating modular code in personal projects
 * 5. Learn build systems (CMake, Make) for larger projects
 * 6. Explore header-only libraries (Eigen, JSON, etc.)
 *
 * ============================================================================
 */

// This file demonstrates basic header concepts
// For full exercises, create separate .h and .cpp files as instructed above

int main() {
    cout << "=== Header Files and Modularity Practice ===" << endl;
    cout << "This exercise requires creating multiple files!" << endl;
    cout << "Follow the TODOs above to create header and implementation files." << endl;

    // After creating files, test them here

    return 0;
}
