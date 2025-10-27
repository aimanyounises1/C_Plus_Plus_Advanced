/*
 * ==================================================================================================
 * Exercise: Polymorphism in C++
 * ==================================================================================================
 * Difficulty: Intermediate/Advanced | Time: 40-50 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master compile-time vs runtime polymorphism
 * 2. Understand virtual functions and vtables
 * 3. Practice function overriding
 * 4. Learn dynamic binding
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Polymorphic GPU resource management
 * - Virtual function overhead in tight loops
 * - Plugin architectures for GPU algorithms
 * - Strategy pattern for kernel selection
 * ==================================================================================================
 */

#include <iostream>
#include <vector>
using namespace std;

/*
 * ==================================================================================================
 * THEORY: Polymorphism
 * ==================================================================================================
 *
 * POLYMORPHISM: "Many forms" - same interface, different implementations
 *
 * Types:
 * 1. COMPILE-TIME (Static):
 *    - Function overloading
 *    - Operator overloading
 *    - Templates
 *    - Resolved at compile time
 *
 * 2. RUNTIME (Dynamic):
 *    - Virtual functions
 *    - Resolved at runtime via vtable
 *    - Requires base class pointer/reference
 *
 * VIRTUAL FUNCTION:
 * - Declared with 'virtual' keyword
 * - Can be overridden in derived classes
 * - Use 'override' keyword (C++11) for safety
 * - Enables dynamic dispatch
 *
 * VTABLE (Virtual Table):
 * - Compiler-generated table of function pointers
 * - One vtable per class with virtual functions
 * - Each object has vptr (pointer to vtable)
 * - Adds 8 bytes per object (on 64-bit)
 * ==================================================================================================
 */

/*
 * EXERCISE 1: Compile-Time Polymorphism (10 min)
 */

// TODO 1.1: Function overloading (compile-time)
class Calculator {
public:
    int add(int a, int b) { return a + b; }
    double add(double a, double b) { return a + b; }
    int add(int a, int b, int c) { return a + b + c; }
};

// Compiler selects correct function at compile time

/*
 * EXERCISE 2: Runtime Polymorphism (20 min)
 */

// TODO 2.1: Virtual functions for runtime polymorphism
class Shape {
public:
    virtual double area() const {
        return 0.0;
    }

    virtual void draw() const {
        cout << "Drawing generic shape" << endl;
    }

    virtual ~Shape() {}  // Virtual destructor important!
};

class Circle : public Shape {
private:
    double radius;

public:
    Circle(double r) : radius(r) {}

    double area() const override {
        return 3.14159 * radius * radius;
    }

    void draw() const override {
        cout << "Drawing circle with radius " << radius << endl;
    }
};

class Rectangle : public Shape {
private:
    double width, height;

public:
    Rectangle(double w, double h) : width(w), height(h) {}

    double area() const override {
        return width * height;
    }

    void draw() const override {
        cout << "Drawing rectangle " << width << "x" << height << endl;
    }
};

// TODO 2.2: Test polymorphism
void processShape(const Shape& shape) {
    shape.draw();
    cout << "Area: " << shape.area() << endl;
}

/*
 * EXERCISE 3: Pure Virtual vs Virtual (10 min)
 */

// TODO 3.1: Virtual function with default implementation
class Animal {
public:
    virtual void makeSound() {
        cout << "Generic animal sound" << endl;
    }
};

// TODO 3.2: Override in derived class
class Dog : public Animal {
public:
    void makeSound() override {
        cout << "Woof!" << endl;
    }
};

class Cat : public Animal {
public:
    void makeSound() override {
        cout << "Meow!" << endl;
    }
};

/*
 * EXERCISE 4: Dynamic Binding Demo (10 min)
 */

// TODO 4.1: Demonstrate dynamic binding
void demonstratePolymorphism() {
    vector<Shape*> shapes;
    shapes.push_back(new Circle(5.0));
    shapes.push_back(new Rectangle(4.0, 6.0));
    shapes.push_back(new Circle(3.0));

    cout << "=== Polymorphic behavior ===" << endl;
    for (Shape* shape : shapes) {
        processShape(*shape);  // Calls correct overridden method
    }

    // Cleanup
    for (Shape* shape : shapes) {
        delete shape;
    }
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is polymorphism?
 * A: "Many forms" - same interface, different behaviors. Compile-time (overloading) or runtime (virtual)
 *
 * Q2: How does virtual function work?
 * A: Vtable mechanism - each class has table of function pointers, objects have vptr to vtable
 *
 * Q3: What is the cost of virtual functions?
 * A: - Extra indirection (vtable lookup)
 *    - 8 bytes per object (vptr)
 *    - Prevents inlining
 *    - Typically negligible unless tight loop
 *
 * Q4: Why virtual destructor?
 * A: Ensure derived destructor called when deleting through base pointer. Prevents resource leaks
 *
 * Q5: What is override keyword for?
 * A: C++11 - compiler checks you're actually overriding. Catches typos/signature mismatches
 *
 * Q6: Can static functions be virtual?
 * A: No! Static functions don't have 'this' pointer, can't use vtable
 *
 * Q7: What is dynamic binding?
 * A: Determining which function to call at runtime based on object's actual type
 *
 * Q8: Difference between overloading and overriding?
 * A: Overloading: Same name, different parameters (compile-time)
 *    Overriding: Same signature, different implementation (runtime, requires virtual)
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * ==================================================================================================
 *
 * 1. Performance Consideration:
 *    - Virtual functions in device code have overhead
 *    - Avoid in performance-critical kernels
 *    - Prefer templates for compile-time polymorphism
 *
 * 2. Host-Side Polymorphism:
 *    - Kernel selection strategy pattern
 *    - Different algorithms for different GPU architectures
 *    - Plugin system for custom kernels
 *
 * 3. Memory Management:
 *    - Polymorphic memory allocators (device/host/unified)
 *    - Virtual destructors for proper cleanup
 *
 * 4. Device Code Restrictions:
 *    - Limited virtual function support in CUDA
 *    - Prefer static polymorphism (templates) for device code
 *
 * ==================================================================================================
 * COMPILATION: g++ -std=c++17 02_polymorphism.cpp -o polymorphism
 *
 * LEARNING CHECKLIST:
 * ☐ Understand compile-time vs runtime polymorphism
 * ☐ Use virtual functions correctly
 * ☐ Know vtable mechanism
 * ☐ Always use virtual destructors in base classes
 * ☐ Use override keyword for safety
 * ☐ Recognize performance implications
 * ==================================================================================================
 */

int main() {
    cout << "=== Polymorphism Practice ===" << endl;

    demonstratePolymorphism();

    return 0;
}
