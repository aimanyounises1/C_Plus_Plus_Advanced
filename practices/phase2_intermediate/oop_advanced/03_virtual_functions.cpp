/*
 * ==================================================================================================
 * Exercise: Virtual Functions in C++
 * ==================================================================================================
 * Difficulty: Advanced | Time: 45-55 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Deep dive into virtual function mechanics
 * 2. Understand vtable and vptr
 * 3. Master virtual destructors
 * 4. Learn final and override keywords
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Performance implications in hot paths
 * - Memory layout with virtual functions
 * - Devirtualization optimizations
 * - Plugin architectures
 * ==================================================================================================
 */

#include <iostream>
#include <typeinfo>
using namespace std;

/*
 * THEORY: Virtual Functions Deep Dive
 *
 * VTABLE (Virtual Table):
 * - Compiler creates one vtable per class with virtual functions
 * - Contains pointers to virtual function implementations
 * - Shared by all objects of the class
 *
 * VPTR (Virtual Pointer):
 * - Each object has a vptr (hidden member)
 * - Points to the vtable of its class
 * - Initialized by constructor
 * - Adds 8 bytes (64-bit) per object
 *
 * DYNAMIC DISPATCH:
 * 1. Access object's vptr
 * 2. Follow vptr to vtable
 * 3. Index into vtable
 * 4. Call function through pointer
 * ==================================================================================================
 */

/*
 * EXERCISE 1: Virtual Destructors (15 min)
 */

class Base {
public:
    Base() { cout << "Base constructor" << endl; }

    // CRITICAL: Virtual destructor
    virtual ~Base() { cout << "Base destructor" << endl; }

    virtual void display() { cout << "Base display" << endl; }
};

class Derived : public Base {
private:
    int* data;

public:
    Derived() : data(new int[100]) {
        cout << "Derived constructor (allocated memory)" << endl;
    }

    ~Derived() override {
        delete[] data;
        cout << "Derived destructor (freed memory)" << endl;
    }

    void display() override { cout << "Derived display" << endl; }
};

// TODO: Test with and without virtual destructor
void testVirtualDestructor() {
    Base* ptr = new Derived();
    delete ptr;  // Without virtual destructor: memory leak!
}

/*
 * EXERCISE 2: Override and Final (10 min)
 */

class Animal {
public:
    virtual void makeSound() { cout << "Animal sound" << endl; }

    // final: Cannot be overridden further
    virtual void eat() final { cout << "Animal eats" << endl; }
};

class Dog : public Animal {
public:
    void makeSound() override { cout << "Woof!" << endl; }

    // ERROR if uncommented: eat() is final in base
    // void eat() override { cout << "Dog eats" << endl; }
};

// final class: Cannot be inherited
class Cat final : public Animal {
public:
    void makeSound() override { cout << "Meow!" << endl; }
};

// ERROR if uncommented: Cat is final
// class Kitten : public Cat { };

/*
 * EXERCISE 3: Pure Virtual vs Virtual (10 min)
 */

class Interface {
public:
    virtual void required() = 0;  // Pure virtual (must override)
    virtual void optional() { cout << "Default implementation" << endl; }
    virtual ~Interface() {}
};

class Implementation : public Interface {
public:
    void required() override { cout << "Required implemented" << endl; }
    // optional() can use default or override
};

/*
 * EXERCISE 4: Virtual Function Performance (10 min)
 */

class NonVirtual {
public:
    void compute() { /* work */ }
};

class Virtual {
public:
    virtual void compute() { /* work */ }
};

// Measure sizeof
void sizeComparison() {
    cout << "NonVirtual size: " << sizeof(NonVirtual) << " bytes" << endl;
    cout << "Virtual size: " << sizeof(Virtual) << " bytes (includes vptr)" << endl;
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: Why do we need virtual destructors?
 * A: When deleting through base pointer, ensures derived destructor called. Prevents leaks
 *
 * Q2: What is vtable?
 * A: Compiler-generated table of function pointers for virtual functions, one per class
 *
 * Q3: What is vptr?
 * A: Hidden pointer in each object pointing to its class's vtable
 *
 * Q4: Cost of virtual functions?
 * A: - Memory: +8 bytes per object (vptr)
 *    - Performance: One extra indirection, prevents inlining
 *    - Usually negligible unless tight loop
 *
 * Q5: What is 'final' keyword?
 * A: C++11 - prevents override (function) or inheritance (class)
 *
 * Q6: What is 'override' keyword?
 * A: C++11 - explicitly marks override, compiler verifies signature matches
 *
 * Q7: Can constructor be virtual?
 * A: No! Object doesn't exist yet, no vptr to use
 *
 * Q8: Can we have virtual function in template?
 * A: Yes! Template and virtual are orthogonal features
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Avoid virtuals in device kernels (performance)
 * - Host-side polymorphism for algorithm selection
 * - Virtual destructors critical for GPU resource classes
 * - Prefer templates for zero-cost abstraction on GPU
 *
 * COMPILATION: g++ -std=c++17 03_virtual_functions.cpp -o virtual
 * ==================================================================================================
 */

int main() {
    cout << "=== Virtual Functions Practice ===" << endl;

    testVirtualDestructor();
    sizeComparison();

    return 0;
}
