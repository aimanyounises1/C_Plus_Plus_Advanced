/*
 * ==================================================================================================
 * Exercise: Inheritance in C++
 * ==================================================================================================
 * Difficulty: Intermediate | Time: 45-55 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master single, multiple, and multilevel inheritance
 * 2. Understand access control in inheritance
 * 3. Learn constructor/destructor call order
 * 4. Practice IS-A relationships
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Class hierarchies in graphics APIs
 * - Device/Stream inheritance patterns
 * - Polymorphic GPU resource management
 * - Virtual function dispatch performance
 * ==================================================================================================
 */

#include <iostream>
#include <string>
using namespace std;

/*
 * ==================================================================================================
 * THEORY: Inheritance
 * ==================================================================================================
 *
 * INHERITANCE: Derive new class from existing class
 * - Base class (parent, super class)
 * - Derived class (child, sub class)
 * - Reuse code, establish IS-A relationship
 *
 * Syntax: class Derived : access_specifier Base { };
 *
 * Access specifiers in inheritance:
 * - public: IS-A relationship (most common)
 * - protected: Implementation inheritance
 * - private: Hide base class interface
 *
 * Types:
 * 1. Single: class B : public A
 * 2. Multiple: class C : public A, public B
 * 3. Multilevel: class C : public B, class B : public A
 * 4. Hierarchical: Multiple classes inherit from one
 * 5. Hybrid: Combination
 * ==================================================================================================
 */

/*
 * EXERCISE 1: Single Inheritance (15 min)
 */

// TODO 1.1: Create base and derived classes
class Animal {
protected:
    string name;
    int age;

public:
    Animal(const string& n, int a) : name(n), age(a) {
        cout << "Animal constructor: " << name << endl;
    }

    virtual ~Animal() {
        cout << "Animal destructor: " << name << endl;
    }

    void eat() { cout << name << " is eating" << endl; }
    void sleep() { cout << name << " is sleeping" << endl; }
};

class Dog : public Animal {
private:
    string breed;

public:
    Dog(const string& n, int a, const string& b)
        : Animal(n, a), breed(b) {
        cout << "Dog constructor" << endl;
    }

    ~Dog() {
        cout << "Dog destructor" << endl;
    }

    void bark() { cout << name << " barks!" << endl; }
};

// TODO 1.2: Test constructor/destructor order
// Create Dog object, observe call sequence

/*
 * EXERCISE 2: Access Control in Inheritance (15 min)
 */

// TODO 2.1: Understand public vs protected vs private inheritance
class Base {
public:
    int pub;
protected:
    int prot;
private:
    int priv;
};

// Public inheritance (IS-A)
class DerivedPublic : public Base {
    // pub remains public
    // prot remains protected
    // priv not accessible
};

// Protected inheritance (implementation detail)
class DerivedProtected : protected Base {
    // pub becomes protected
    // prot remains protected
    // priv not accessible
};

// Private inheritance (implementation only)
class DerivedPrivate : private Base {
    // pub becomes private
    // prot becomes private
    // priv not accessible
};

/*
 * EXERCISE 3: Multiple Inheritance (15 min)
 */

// TODO 3.1: Create class inheriting from multiple bases
class Flyable {
public:
    virtual void fly() { cout << "Flying..." << endl; }
};

class Swimmable {
public:
    virtual void swim() { cout << "Swimming..." << endl; }
};

class Duck : public Animal, public Flyable, public Swimmable {
public:
    Duck(const string& n, int a) : Animal(n, a) {}

    void fly() override { cout << name << " flies!" << endl; }
    void swim() override { cout << name << " swims!" << endl; }
};

// TODO 3.2: Test multiple inheritance
// Create Duck, call methods from all base classes

/*
 * EXERCISE 4: Diamond Problem (10 min)
 */

// TODO 4.1: Understand diamond problem
class PoweredDevice {
public:
    void powerOn() { cout << "Powering on..." << endl; }
};

// Without virtual inheritance: two copies of PoweredDevice
class Scanner : public PoweredDevice { };
class Printer : public PoweredDevice { };
class Copier : public Scanner, public Printer {
    // Ambiguous: Which powerOn()? Scanner's or Printer's?
};

// TODO 4.2: Fix with virtual inheritance
class VScanner : virtual public PoweredDevice { };
class VPrinter : virtual public PoweredDevice { };
class VCopier : public VScanner, public VPrinter {
    // Now only one copy of PoweredDevice
};

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is inheritance?
 * A: Mechanism to create new class from existing class, inheriting members and behavior
 *
 * Q2: When to use public inheritance?
 * A: IS-A relationship: Derived IS-A Base (Dog IS-A Animal)
 *
 * Q3: Constructor/destructor call order in inheritance?
 * A: Construction: Base → Derived. Destruction: Derived → Base (reverse order)
 *
 * Q4: What is the diamond problem?
 * A: Multiple inheritance causing duplicate base class. Fix with virtual inheritance
 *
 * Q5: Can you override non-virtual functions?
 * A: Yes, but it's hiding (not polymorphic). Use virtual for true override
 *
 * Q6: What is protected inheritance used for?
 * A: Implementation detail inheritance (not IS-A). Rarely used
 *
 * Q7: Can constructors be inherited?
 * A: No (pre-C++11). C++11: using Base::Base to inherit constructors
 *
 * Q8: What members are not inherited?
 * A: Constructors, destructors, assignment operators, friend functions
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Device types: GPUDevice inherits from Device
 * - Stream hierarchies: CudaStream, CudaStreamNonBlocking
 * - Memory types: DeviceMemory, HostMemory, UnifiedMemory
 * - Kernel launchers: Polymorphic kernel execution
 *
 * COMPILATION: g++ -std=c++17 01_inheritance.cpp -o inheritance
 *
 * LEARNING CHECKLIST:
 * ☐ Create single inheritance hierarchies
 * ☐ Understand public/protected/private inheritance
 * ☐ Know constructor/destructor call order
 * ☐ Handle multiple inheritance
 * ☐ Solve diamond problem with virtual inheritance
 * ==================================================================================================
 */

int main() {
    cout << "=== Inheritance Practice ===" << endl;

    Dog d("Buddy", 3, "Golden Retriever");
    d.eat();
    d.bark();

    return 0;
}
