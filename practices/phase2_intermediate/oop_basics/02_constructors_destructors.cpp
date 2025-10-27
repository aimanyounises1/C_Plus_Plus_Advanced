/*
 * ============================================================================
 * Exercise: Constructors and Destructors in C++
 * ============================================================================
 * Difficulty: Intermediate
 * Time: 45-60 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Understand object lifecycle (construction and destruction)
 * 2. Master different types of constructors
 * 3. Learn member initialization lists
 * 4. Understand when and why destructors are needed
 * 5. Practice the Rule of Three/Five/Zero
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Resource management critical (GPU memory, handles, streams)
 * - RAII pattern essential for exception safety
 * - Understanding copy/move semantics for performance
 * - Proper cleanup prevents memory leaks (CPU and GPU)
 * - Constructor initialization order matters for complex objects
 *
 * PREREQUISITES:
 * - Classes and objects basics
 * - Member variables and functions
 * - Pointers and dynamic memory
 * ============================================================================
 */

#include <iostream>
#include <string>
#include <cstring>

using namespace std;

/*
 * ============================================================================
 * THEORY: Constructors and Destructors
 * ============================================================================
 *
 * CONSTRUCTOR: Special member function that initializes an object
 * - Same name as class
 * - No return type (not even void)
 * - Called automatically when object is created
 * - Can be overloaded
 *
 * Types of constructors:
 * 1. Default constructor: No parameters
 * 2. Parameterized constructor: Takes parameters
 * 3. Copy constructor: Creates object from another object
 * 4. Move constructor: Transfers resources from temporary
 *
 * DESTRUCTOR: Special member function that cleans up an object
 * - Name: ~ClassName()
 * - No parameters, no return type
 * - Called automatically when object is destroyed
 * - Cannot be overloaded (only one destructor)
 * - Use for releasing resources (memory, file handles, etc.)
 *
 * ============================================================================
 */

/*
 * ============================================================================
 * EXERCISE 1: Default and Parameterized Constructors (15 minutes)
 * ============================================================================
 */

// TODO 1.1: Create a "Person" class with multiple constructors
class Person {
private:
    string name;
    int age;

public:
    // TODO: Implement default constructor
    // Person() {
    //     name = "Unknown";
    //     age = 0;
    //     cout << "Default constructor called" << endl;
    // }

    // TODO: Implement parameterized constructor
    // Person(string n, int a) {
    //     name = n;
    //     age = a;
    //     cout << "Parameterized constructor called" << endl;
    // }

    void display() {
        cout << "Name: " << name << ", Age: " << age << endl;
    }
};

// TODO 1.2: Test constructor overloading
// Create persons using different constructors
// Observe which constructor gets called


/*
 * ============================================================================
 * EXERCISE 2: Member Initialization Lists (15 minutes)
 * ============================================================================
 * Best practice for initializing members
 */

// TODO 2.1: Create a "Rectangle" class using initialization lists
class Rectangle {
private:
    const double width;   // const members must use initialization list
    const double height;

public:
    // TODO: Use initialization list syntax
    // Rectangle(double w, double h) : width(w), height(h) {
    //     cout << "Rectangle constructed" << endl;
    // }

    double area() const { return width * height; }
};

// TODO 2.2: Compare initialization list vs assignment
// Try initializing const members with assignment - observe error!


// TODO 2.3: Understand initialization order
// Create a class with multiple members
// Order in initialization list doesn't matter - members initialize in declaration order!
class Example {
    int a;
    int b;
    int c;
public:
    // Members initialize in order: a, then b, then c (regardless of list order)
    Example(int x) : c(x), b(c+1), a(b+1) {
        // Be careful! If b depends on c, and c is initialized after b, undefined behavior!
    }
};


/*
 * ============================================================================
 * EXERCISE 3: Destructors and Resource Management (15 minutes)
 * ============================================================================
 */

// TODO 3.1: Create a "DynamicArray" class with destructor
class DynamicArray {
private:
    int* data;
    int size;

public:
    // Constructor allocates memory
    DynamicArray(int s) : size(s) {
        data = new int[size];
        cout << "Array of size " << size << " constructed" << endl;
    }

    // TODO: Implement destructor to free memory
    // ~DynamicArray() {
    //     delete[] data;
    //     cout << "Array destroyed" << endl;
    // }

    void set(int index, int value) {
        if (index >= 0 && index < size) data[index] = value;
    }

    int get(int index) const {
        return (index >= 0 && index < size) ? data[index] : 0;
    }
};

// TODO 3.2: Test object lifecycle
// Create DynamicArray objects
// Observe constructor and destructor calls
// Try creating in different scopes


/*
 * ============================================================================
 * EXERCISE 4: Copy Constructor (15 minutes)
 * ============================================================================
 * Deep vs shallow copy
 */

// TODO 4.1: Create a "String" class with copy constructor
class String {
private:
    char* data;
    int length;

public:
    // Constructor
    String(const char* str) {
        length = strlen(str);
        data = new char[length + 1];
        strcpy(data, str);
        cout << "String constructed: " << data << endl;
    }

    // TODO: Implement copy constructor (deep copy)
    // String(const String& other) {
    //     length = other.length;
    //     data = new char[length + 1];
    //     strcpy(data, other.data);
    //     cout << "String copied: " << data << endl;
    // }

    // Destructor
    ~String() {
        cout << "String destroyed: " << data << endl;
        delete[] data;
    }

    void print() const {
        cout << data << endl;
    }
};

// TODO 4.2: Test copy constructor
// String s1("Hello");
// String s2 = s1;  // Copy constructor called
// String s3(s1);   // Also copy constructor


// TODO 4.3: Understand shallow copy problem
// Without copy constructor, both objects would point to same memory!
// When one is destroyed, the other has dangling pointer


/*
 * ============================================================================
 * EXERCISE 5: Delegating Constructors (C++11) (10 minutes)
 * ============================================================================
 */

// TODO 5.1: Create a class with delegating constructors
class Point3D {
private:
    double x, y, z;

public:
    // Main constructor
    Point3D(double x, double y, double z) : x(x), y(y), z(z) {
        cout << "3D Point constructed" << endl;
    }

    // TODO: Delegate to main constructor
    // Point3D() : Point3D(0, 0, 0) {}
    // Point3D(double x, double y) : Point3D(x, y, 0) {}
};


/*
 * ============================================================================
 * CHALLENGE EXERCISES (Optional - 20 minutes)
 * ============================================================================
 */

// CHALLENGE 1: Implement the Rule of Three
// Create a class that manages a resource (dynamic array)
// Implement: Destructor, Copy Constructor, Copy Assignment Operator
// Ensure deep copying and proper cleanup


// CHALLENGE 2: Move Constructor and Move Assignment (Rule of Five)
// Extend Challenge 1 to include move semantics
// Move constructor: Transfer ownership without copying
// Move assignment: Transfer ownership in assignment
// Use std::move to test


// CHALLENGE 3: Resource Acquisition Is Initialization (RAII)
// Create a "FileHandle" class
// Constructor opens file
// Destructor closes file
// Ensure file is always closed even if exception occurs


/*
 * ============================================================================
 * COMMON INTERVIEW QUESTIONS & ANSWERS
 * ============================================================================
 *
 * Q1: What is a constructor and what are its types?
 * A: Constructor: Special member function that initializes an object
 *    Types:
 *    1. Default: MyClass() {}
 *    2. Parameterized: MyClass(int x) : data(x) {}
 *    3. Copy: MyClass(const MyClass& other) { ... }
 *    4. Move: MyClass(MyClass&& other) noexcept { ... }
 *
 *    Characteristics:
 *    - Same name as class
 *    - No return type
 *    - Can be overloaded
 *    - Called automatically on object creation
 *
 * Q2: What is a member initialization list and why use it?
 * A: Syntax: Constructor(params) : member1(value1), member2(value2) { }
 *
 *    Advantages:
 *    - More efficient (direct initialization vs assignment)
 *    - Required for: const members, reference members, members without default constructor
 *    - Preferred for all members (best practice)
 *
 *    Example:
 *    class Example {
 *        const int x;
 *        int& ref;
 *    public:
 *        Example(int val, int& r) : x(val), ref(r) {}  // Must use list
 *    };
 *
 * Q3: What is the difference between shallow copy and deep copy?
 * A: Shallow copy: Copies member values directly
 *    - Default copy constructor does shallow copy
 *    - Problem: If member is pointer, both objects point to same memory
 *    - Deleting one object invalidates the other
 *
 *    Deep copy: Allocates new memory and copies contents
 *    - Custom copy constructor needed
 *    - Each object has independent memory
 *    - Safe but potentially expensive
 *
 *    Example:
 *    class Array {
 *        int* data;
 *        // Deep copy constructor:
 *        Array(const Array& other) {
 *            data = new int[other.size];
 *            memcpy(data, other.data, other.size * sizeof(int));
 *        }
 *    };
 *
 * Q4: What is the Rule of Three?
 * A: If a class needs one of these, it likely needs all three:
 *    1. Destructor: ~MyClass()
 *    2. Copy constructor: MyClass(const MyClass&)
 *    3. Copy assignment operator: MyClass& operator=(const MyClass&)
 *
 *    Why: If you manage resources (pointers, handles), you need custom:
 *    - Destructor to release resources
 *    - Copy constructor/assignment for deep copying
 *
 *    Modern C++: Rule of Five (adds move constructor and move assignment)
 *    Or Rule of Zero: Use smart pointers, avoid manual management
 *
 * Q5: When is a destructor called?
 * A: Destructor called when object goes out of scope:
 *    - Local object: At end of block
 *    - Dynamic object: When delete is called
 *    - Global/static object: At program termination
 *    - Temporary object: At end of full expression
 *    - Exception: During stack unwinding
 *
 *    Order: Reverse of construction
 *    - Derived class destructor, then base class destructor
 *    - Members destroyed in reverse declaration order
 *
 * Q6: What happens if you don't define a constructor?
 * A: Compiler provides defaults:
 *    - Default constructor: MyClass() = default;
 *      - Only if no other constructors defined
 *      - Default-initializes members
 *    - Copy constructor: Member-wise copy (shallow)
 *    - Move constructor: Member-wise move (C++11+)
 *
 *    You lose default constructor if you define any constructor!
 *    To keep it: explicitly = default or define it
 *
 * Q7: Can constructors be private? Why would you do that?
 * A: Yes! Reasons:
 *    1. Singleton pattern: Only one instance allowed
 *       class Singleton {
 *           static Singleton* instance;
 *           Singleton() {}  // Private
 *       public:
 *           static Singleton* getInstance() { ... }
 *       };
 *
 *    2. Factory pattern: Control object creation
 *    3. Prevent object creation on stack (force heap allocation)
 *
 *    Friends can still access private constructors
 *
 * Q8: What is constructor delegation?
 * A: C++11 feature: Constructor calls another constructor
 *    class Example {
 *        int x, y;
 *    public:
 *        Example(int x, int y) : x(x), y(y) {}
 *        Example() : Example(0, 0) {}  // Delegates to above
 *    };
 *
 *    Benefits:
 *    - Avoid code duplication
 *    - Centralize initialization logic
 *    - Cleaner code
 *
 *    Note: Delegation must be only thing in initialization list
 *
 * Q9: What is RAII?
 * A: Resource Acquisition Is Initialization
 *    - Tie resource lifetime to object lifetime
 *    - Acquire resource in constructor
 *    - Release resource in destructor
 *    - Automatic cleanup, exception-safe
 *
 *    Examples:
 *    - std::unique_ptr, std::shared_ptr
 *    - std::lock_guard
 *    - File handles, database connections
 *
 *    Benefits:
 *    - No manual cleanup needed
 *    - Prevents resource leaks
 *    - Exception safe
 *
 * Q10: What is the order of constructor/destructor calls in inheritance?
 * A: Construction order (top to bottom):
 *    1. Base class constructor
 *    2. Member variables (in declaration order)
 *    3. Derived class constructor body
 *
 *    Destruction order (bottom to top, reverse of construction):
 *    1. Derived class destructor body
 *    2. Member variables (reverse declaration order)
 *    3. Base class destructor
 *
 *    Example:
 *    class Base { Base() { cout << "Base\n"; } };
 *    class Derived : public Base {
 *        Derived() { cout << "Derived\n"; }  // Prints: Base, then Derived
 *    };
 *
 * ============================================================================
 * GPU/CUDA RELEVANCE FOR NVIDIA INTERVIEW:
 * ============================================================================
 *
 * 1. GPU Memory Management:
 *    - Wrap cudaMalloc/cudaFree in RAII classes
 *    - Constructor allocates GPU memory
 *    - Destructor frees it (exception-safe)
 *    - Example: thrust::device_vector uses RAII
 *
 * 2. Stream Management:
 *    - cudaStream_t wrapped in class
 *    - Constructor creates stream
 *    - Destructor destroys stream
 *    - Ensures proper cleanup
 *
 * 3. Copy Semantics:
 *    - Understand host-to-device copies
 *    - cudaMemcpy in copy constructor
 *    - Move semantics for large data transfers
 *    - Avoid unnecessary copies (performance!)
 *
 * 4. Resource Handles:
 *    - cuBLAS handles, cuDNN handles
 *    - RAII pattern for initialization/cleanup
 *    - Thread safety considerations
 *
 * 5. Unified Memory:
 *    - cudaMallocManaged in constructor
 *    - Automatic management with destructors
 *    - Copy semantics need careful design
 *
 * ============================================================================
 * COMPILATION & EXECUTION:
 * ============================================================================
 *
 * Compile with:
 *   g++ -std=c++17 -Wall 02_constructors_destructors.cpp -o constructors
 *
 * Run:
 *   ./constructors
 *
 * ============================================================================
 * EXPECTED OUTPUT (after completing exercises):
 * ============================================================================
 *
 * Should show:
 * - Constructor calls (default and parameterized)
 * - Destructor calls (in reverse order)
 * - Copy constructor behavior
 * - Deep vs shallow copy differences
 * - RAII pattern in action
 *
 * ============================================================================
 * LEARNING CHECKLIST:
 * ============================================================================
 *
 * After completing these exercises, you should be able to:
 * ☐ Write default and parameterized constructors
 * ☐ Use member initialization lists correctly
 * ☐ Implement destructors for resource cleanup
 * ☐ Create deep copy constructors
 * ☐ Understand constructor delegation
 * ☐ Apply the Rule of Three/Five
 * ☐ Implement RAII pattern
 * ☐ Explain constructor/destructor call order
 * ☐ Recognize when custom constructors are needed
 * ☐ Debug memory leaks related to constructors/destructors
 *
 * ============================================================================
 * NEXT STEPS:
 * ============================================================================
 *
 * 1. Study access specifiers (03_access_specifiers.cpp)
 * 2. Learn about operator overloading
 * 3. Practice move semantics (C++11)
 * 4. Explore smart pointers (unique_ptr, shared_ptr)
 * 5. Study inheritance and polymorphism
 *
 * ============================================================================
 */

int main() {
    cout << "=== Constructors and Destructors Practice ===" << endl;

    // Test your implementations here

    return 0;
}
