/*
 * ============================================================================
 * Exercise: Classes and Objects in C++
 * ============================================================================
 * Difficulty: Beginner/Intermediate
 * Time: 40-50 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Understand object-oriented programming fundamentals
 * 2. Create classes with member variables and functions
 * 3. Instantiate and use objects
 * 4. Understand the difference between class and struct
 * 5. Practice method definition (inside and outside class)
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - OOP is fundamental to C++ and large codebases
 * - Classes model real-world entities (GPUs, textures, buffers)
 * - Understanding object lifecycle critical for resource management
 * - CUDA uses OOP principles for kernel configuration
 * - Driver APIs heavily object-oriented
 *
 * PREREQUISITES:
 * - Functions and function parameters
 * - Basic understanding of data types
 * - Familiarity with structs (optional but helpful)
 * ============================================================================
 */

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

/*
 * ============================================================================
 * THEORY: What is a Class?
 * ============================================================================
 *
 * A class is a user-defined data type that encapsulates:
 * 1. DATA (member variables / attributes / fields)
 * 2. BEHAVIOR (member functions / methods)
 *
 * Key concepts:
 * - ENCAPSULATION: Bundling data and methods that operate on that data
 * - OBJECT: An instance of a class (like a variable is an instance of a type)
 * - MEMBER: A variable or function belonging to a class
 * - this: Pointer to the current object
 *
 * Basic syntax:
 * class ClassName {
 *     // Member variables
 *     int data;
 *
 *     // Member functions
 *     void doSomething() {
 *         // Can access 'data' directly
 *     }
 * };
 *
 * ============================================================================
 */

/*
 * ============================================================================
 * EXERCISE 1: Your First Class (10 minutes)
 * ============================================================================
 * Create a simple class and instantiate objects
 */

// TODO 1.1: Create a "Rectangle" class
// Requirements:
// - Member variables: width (double), height (double)
// - Member function: calculateArea() that returns width * height
// - Member function: calculatePerimeter() that returns 2 * (width + height)
// Note: For now, make everything public (we'll learn access control later)

class Rectangle {
public:
    // TODO: Add member variables

    // TODO: Add member functions
};

// TODO 1.2: Create Rectangle objects
// In main(), create:
// - A Rectangle with width=5.0, height=3.0
// - Calculate and print its area and perimeter


// TODO 1.3: Create multiple objects
// Create an array or vector of 3 different Rectangle objects
// Calculate the total area of all rectangles


/*
 * ============================================================================
 * EXERCISE 2: Methods Defined Outside Class (10 minutes)
 * ============================================================================
 * Learn to separate declaration from definition
 */

// TODO 2.1: Create a "Circle" class with external method definitions
// Class declaration:
class Circle {
public:
    double radius;

    // Method declarations (prototypes)
    double getArea();           // Implement outside class
    double getCircumference();  // Implement outside class
    void   setRadius(double r); // Implement outside class
};

// TODO: Implement the methods outside the class using ClassName::methodName syntax
// Example:
// double Circle::getArea() {
//     return 3.14159 * radius * radius;
// }


// TODO 2.2: Test the Circle class
// Create Circle objects and test all methods


/*
 * ============================================================================
 * EXERCISE 3: Understanding 'this' Pointer (10 minutes)
 * ============================================================================
 * Learn about the implicit 'this' pointer
 */

// TODO 3.1: Create a "Point" class that uses 'this'
class Point {
public:
    double x, y;

    // TODO: Implement setCoordinates that uses 'this' to disambiguate
    // void setCoordinates(double x, double y) {
    //     this->x = x;  // this->x is the member, x is the parameter
    //     this->y = y;
    // }

    // TODO: Implement a method that returns 'this' for method chaining
    // Point* moveTo(double x, double y) {
    //     this->x = x;
    //     this->y = y;
    //     return this;
    // }
};

// TODO 3.2: Test method chaining
// If you return 'this', you can chain calls:
// Point p;
// p.moveTo(5, 10)->moveTo(15, 20);


// TODO 3.3: Calculate distance between two points
// Add a method: double distanceTo(const Point& other) const
// Use formula: sqrt((x2-x1)² + (y2-y1)²)


/*
 * ============================================================================
 * EXERCISE 4: Class vs Struct (5 minutes)
 * ============================================================================
 * Understand the difference
 */

// TODO 4.1: Create equivalent struct and class
struct PersonStruct {
    string name;
    int age;
    void introduce() { cout << "Hi, I'm " << name << endl; }
};

class PersonClass {
    string name;  // Default: private in class
    int age;
};

// TODO 4.2: Observe the difference
// Try to access members of both
// Note: struct members are public by default, class members are private
// Otherwise, they're identical in C++!


/*
 * ============================================================================
 * EXERCISE 5: Practical Application - Bank Account (15 minutes)
 * ============================================================================
 * Build a more complete class
 */

// TODO 5.1: Create a "BankAccount" class
// Requirements:
// - Member variables: accountNumber (string), holderName (string), balance (double)
// - Method: deposit(double amount) - adds to balance
// - Method: withdraw(double amount) - subtracts from balance (check sufficient funds!)
// - Method: getBalance() - returns current balance
// - Method: displayInfo() - prints account information


// TODO 5.2: Create multiple bank accounts
// Create accounts for 3 different people
// Perform various transactions
// Display final balances


/*
 * ============================================================================
 * EXERCISE 6: Array of Objects (10 minutes)
 * ============================================================================
 * Work with multiple objects
 */

// TODO 6.1: Create a "Student" class
class Student {
public:
    string name;
    int id;
    double gpa;

    void display() {
        cout << "ID: " << id << ", Name: " << name << ", GPA: " << gpa << endl;
    }
};

// TODO 6.2: Create an array or vector of Students
// Store at least 5 students
// Find the student with the highest GPA
// Find the average GPA


/*
 * ============================================================================
 * CHALLENGE EXERCISES (Optional - 15 minutes)
 * ============================================================================
 */

// CHALLENGE 1: Create a "Matrix" class
// Represent a 2x2 matrix
// Methods:
// - add(const Matrix& other) - matrix addition
// - multiply(const Matrix& other) - matrix multiplication
// - determinant() - calculate determinant
// - print() - display the matrix


// CHALLENGE 2: Create a "Time" class
// Represent time as hours, minutes, seconds
// Methods:
// - addTime(const Time& other) - add two times (handle overflow!)
// - toSeconds() - convert to total seconds
// - fromSeconds(int seconds) - create Time from seconds
// - print() - display in HH:MM:SS format


// CHALLENGE 3: Create a "Vector3D" class
// Represent a 3D vector (x, y, z)
// Methods:
// - add, subtract, scale
// - dotProduct, crossProduct
// - magnitude, normalize
// GPU-relevant: Used extensively in graphics and physics!


/*
 * ============================================================================
 * COMMON INTERVIEW QUESTIONS & ANSWERS
 * ============================================================================
 *
 * Q1: What is the difference between a class and a struct in C++?
 * A: The only difference is default access:
 *    - struct: members are public by default
 *    - class: members are private by default
 *    Convention:
 *    - Use struct for simple data containers (POD - Plain Old Data)
 *    - Use class for objects with behavior and encapsulation
 *    Note: In C, structs cannot have member functions; in C++, they can!
 *
 * Q2: What is a member function and how is it different from a regular function?
 * A: Member function (method):
 *    - Belongs to a class
 *    - Can access private members
 *    - Has implicit 'this' pointer to the object
 *    - Called on an object: obj.method()
 *    Regular function:
 *    - Standalone, not tied to a class
 *    - Cannot access private class members
 *    - No 'this' pointer
 *    - Called directly: function()
 *
 * Q3: What is the 'this' pointer?
 * A: - Implicit pointer available in non-static member functions
 *    - Points to the object on which the method was called
 *    - Type: ClassName* const (constant pointer to the object)
 *    Uses:
 *    - Disambiguate member variables from parameters
 *    - Return the object itself (for method chaining)
 *    - Pass the object to other functions
 *    Example:
 *    void setX(int x) { this->x = x; }  // Disambiguate
 *    Point& move() { return *this; }     // Return object for chaining
 *
 * Q4: Can you have a function with the same name as a member variable?
 * A: Yes, but it's confusing. Use 'this->' to disambiguate:
 *    class Example {
 *        int value;
 *        void setValue(int value) {
 *            this->value = value;  // this->value is member, value is parameter
 *        }
 *    };
 *    Better practice: Use different names (e.g., value_ or m_value for members)
 *
 * Q5: What is method overloading within a class?
 * A: Having multiple methods with the same name but different parameters:
 *    class Calculator {
 *        int add(int a, int b) { return a + b; }
 *        double add(double a, double b) { return a + b; }
 *        int add(int a, int b, int c) { return a + b + c; }
 *    };
 *    Compiler selects the right method based on arguments (compile-time polymorphism)
 *
 * Q6: How do you define a method outside the class definition?
 * A: Use the scope resolution operator (::):
 *    // In header or class definition:
 *    class MyClass {
 *        void myMethod();
 *    };
 *
 *    // In implementation file:
 *    void MyClass::myMethod() {
 *        // Implementation
 *    }
 *    Benefits:
 *    - Separates interface from implementation
 *    - Keeps class definition clean
 *    - Reduces header dependencies
 *
 * Q7: Can you return an object from a function?
 * A: Yes! Objects can be:
 *    - Returned by value: MyClass getObject() { return MyClass(); }
 *    - Returned by reference: MyClass& getRef() { return globalObj; }
 *    - Returned by pointer: MyClass* getPtr() { return &obj; }
 *    Note: Don't return reference or pointer to local variables (undefined behavior!)
 *    Modern C++: Move semantics make returning by value efficient
 *
 * Q8: What is the size of an object?
 * A: Size = sum of member variables + padding + vtable pointer (if virtual functions)
 *    Example:
 *    class Example {
 *        int x;     // 4 bytes
 *        double y;  // 8 bytes
 *    };
 *    Size may be 12 bytes or 16 bytes (due to alignment padding)
 *    Use sizeof(Example) to check
 *    Member functions don't add to size (stored separately)
 *
 * Q9: What is method chaining and how do you implement it?
 * A: Method chaining: Calling multiple methods in sequence
 *    Example: obj.method1().method2().method3();
 *    Implementation: Return *this (by reference) from each method
 *    class Builder {
 *        Builder& setName(string n) { name = n; return *this; }
 *        Builder& setAge(int a) { age = a; return *this; }
 *    };
 *    Usage: Builder().setName("John").setAge(30);
 *    Popular in builder patterns and fluent interfaces
 *
 * Q10: What is the difference between declaring a method inside vs outside the class?
 * A: Declared inside:
 *    - Implicitly inline (compiler may inline it)
 *    - Definition is visible to all who include the header
 *    - Good for small, simple methods
 *    Declared outside:
 *    - Not inline (unless explicitly marked)
 *    - Can be in separate .cpp file
 *    - Reduces compilation dependencies
 *    - Good for larger methods
 *    Best practice: Small getters/setters inside, complex logic outside
 *
 * ============================================================================
 * GPU/CUDA RELEVANCE FOR NVIDIA INTERVIEW:
 * ============================================================================
 *
 * 1. CUDA Kernel Configuration:
 *    - dim3 is a class: dim3 grid(10, 10), threads(32, 32);
 *    - cudaDeviceProp: Properties object for GPU devices
 *
 * 2. Resource Management Classes:
 *    - Thrust vectors: thrust::device_vector<float>
 *    - RAII wrappers for GPU memory
 *    - Smart pointers for unified memory
 *
 * 3. Matrix/Vector Classes:
 *    - Common in graphics: Vector3, Matrix4x4
 *    - Efficiently passed to GPU kernels
 *    - Operator overloading for intuitive math
 *
 * 4. Object Sizes Matter:
 *    - Kernel parameters have size limits (4KB typically)
 *    - Passing large objects by value = expensive copy
 *    - Prefer passing pointers or small structs
 *
 * 5. Method Inlining:
 *    - __device__ __forceinline__ for GPU methods
 *    - Small methods (like vector operations) should inline
 *    - Reduces register pressure
 *
 * ============================================================================
 * COMPILATION & EXECUTION:
 * ============================================================================
 *
 * Compile with:
 *   g++ -std=c++17 -Wall 01_classes_objects.cpp -o classes
 *
 * Run:
 *   ./classes
 *
 * ============================================================================
 * EXPECTED OUTPUT (after completing exercises):
 * ============================================================================
 *
 * Should demonstrate:
 * - Rectangle area and perimeter calculations
 * - Circle methods working correctly
 * - Point distance calculations
 * - Bank account transactions
 * - Student GPA calculations
 * - Challenge problems (if attempted)
 *
 * ============================================================================
 * LEARNING CHECKLIST:
 * ============================================================================
 *
 * After completing these exercises, you should be able to:
 * ☐ Create classes with member variables and functions
 * ☐ Instantiate objects and call methods
 * ☐ Define methods outside the class using ::
 * ☐ Understand and use the 'this' pointer
 * ☐ Explain the difference between class and struct
 * ☐ Work with arrays/vectors of objects
 * ☐ Implement method chaining
 * ☐ Return objects from functions
 * ☐ Understand object memory layout basics
 * ☐ Apply OOP to real-world problems
 *
 * ============================================================================
 * NEXT STEPS:
 * ============================================================================
 *
 * 1. Complete 02_constructors_destructors.cpp (object lifecycle)
 * 2. Learn about access specifiers (03_access_specifiers.cpp)
 * 3. Study member initialization and const methods
 * 4. Practice designing classes for real problems
 * 5. Read about the Rule of Three/Five/Zero
 *
 * ============================================================================
 */

int main() {
    cout << "=== Classes and Objects Practice ===" << endl;

    // Test your implementations here

    return 0;
}
