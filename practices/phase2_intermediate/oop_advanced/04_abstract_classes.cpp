/*
 * ==================================================================================================
 * Exercise: Abstract Classes and Interfaces in C++
 * ==================================================================================================
 * Difficulty: Intermediate/Advanced | Time: 40-50 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master pure virtual functions
 * 2. Design abstract base classes
 * 3. Implement interfaces (C++ style)
 * 4. Practice interface segregation
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Plugin architecture for GPU algorithms
 * - Strategy pattern for kernel selection
 * - Abstract device interfaces
 * - Testable GPU code design
 * ==================================================================================================
 */

#include <iostream>
#include <vector>
#include <memory>
using namespace std;

/*
 * THEORY: Abstract Classes
 *
 * ABSTRACT CLASS: Has at least one pure virtual function
 * - Cannot instantiate
 * - Used as interface or base class
 * - Derived classes must implement pure virtuals
 *
 * PURE VIRTUAL FUNCTION: virtual void func() = 0;
 * - No implementation (= 0)
 * - Must be overridden in derived class
 * - Makes class abstract
 *
 * INTERFACE (C++ style): All pure virtual functions
 * - No data members (convention)
 * - No implementation
 * - Like Java/C# interface
 * ==================================================================================================
 */

/*
 * EXERCISE 1: Simple Abstract Class (15 min)
 */

// Abstract base class
class Shape {
public:
    virtual double area() const = 0;  // Pure virtual
    virtual double perimeter() const = 0;  // Pure virtual
    virtual void draw() const = 0;  // Pure virtual

    virtual ~Shape() {}  // Virtual destructor

    // Can have non-pure virtual functions
    virtual void printInfo() const {
        cout << "Area: " << area() << ", Perimeter: " << perimeter() << endl;
    }
};

class Circle : public Shape {
private:
    double radius;

public:
    Circle(double r) : radius(r) {}

    double area() const override {
        return 3.14159 * radius * radius;
    }

    double perimeter() const override {
        return 2 * 3.14159 * radius;
    }

    void draw() const override {
        cout << "Drawing circle" << endl;
    }
};

// TODO: Implement Rectangle class

/*
 * EXERCISE 2: Interface Pattern (15 min)
 */

// Pure interface (like Java interface)
class ISerializable {
public:
    virtual string serialize() const = 0;
    virtual void deserialize(const string& data) = 0;
    virtual ~ISerializable() {}
};

class ICloneable {
public:
    virtual ICloneable* clone() const = 0;
    virtual ~ICloneable() {}
};

// Class implementing multiple interfaces
class Document : public ISerializable, public ICloneable {
private:
    string content;

public:
    Document(const string& c = "") : content(c) {}

    string serialize() const override {
        return "DOC:" + content;
    }

    void deserialize(const string& data) override {
        if (data.substr(0, 4) == "DOC:") {
            content = data.substr(4);
        }
    }

    ICloneable* clone() const override {
        return new Document(*this);
    }

    void print() const { cout << "Content: " << content << endl; }
};

/*
 * EXERCISE 3: Strategy Pattern (10 min)
 */

// Strategy interface
class ISortStrategy {
public:
    virtual void sort(vector<int>& data) = 0;
    virtual string getName() const = 0;
    virtual ~ISortStrategy() {}
};

class BubbleSort : public ISortStrategy {
public:
    void sort(vector<int>& data) override {
        // Bubble sort implementation
        cout << "Sorting with bubble sort" << endl;
    }

    string getName() const override { return "Bubble Sort"; }
};

class QuickSort : public ISortStrategy {
public:
    void sort(vector<int>& data) override {
        // Quick sort implementation
        cout << "Sorting with quick sort" << endl;
    }

    string getName() const override { return "Quick Sort"; }
};

class Sorter {
private:
    ISortStrategy* strategy;

public:
    Sorter(ISortStrategy* s) : strategy(s) {}

    void setStrategy(ISortStrategy* s) { strategy = s; }

    void execute(vector<int>& data) {
        cout << "Using strategy: " << strategy->getName() << endl;
        strategy->sort(data);
    }
};

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is an abstract class?
 * A: Class with at least one pure virtual function. Cannot be instantiated
 *
 * Q2: What is a pure virtual function?
 * A: Virtual function with = 0. Must be overridden in derived class
 *
 * Q3: Can abstract class have constructor?
 * A: Yes! Called by derived class constructors
 *
 * Q4: Can abstract class have data members?
 * A: Yes! Unlike Java interfaces (pre-Java 8)
 *
 * Q5: Difference between interface and abstract class?
 * A: In C++, no formal difference. Convention: interface = all pure virtual, no data
 *
 * Q6: Can you have pointer to abstract class?
 * A: Yes! Can't create object, but can have pointers/references
 *
 * Q7: Why use abstract classes?
 * A: - Define common interface
 *    - Enforce contract in derived classes
 *    - Enable polymorphism
 *    - Separate interface from implementation
 *
 * Q8: What happens if derived class doesn't implement pure virtual?
 * A: Derived class also becomes abstract, cannot be instantiated
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Abstract kernel interfaces for different algorithms
 * - Strategy pattern for selecting GPU vs CPU path
 * - Plugin systems for custom CUDA kernels
 * - Memory allocator interfaces (device/host/unified)
 *
 * COMPILATION: g++ -std=c++17 04_abstract_classes.cpp -o abstract
 * ==================================================================================================
 */

int main() {
    cout << "=== Abstract Classes Practice ===" << endl;

    Circle c(5.0);
    c.draw();
    c.printInfo();

    Document doc("Hello World");
    string serialized = doc.serialize();
    cout << "Serialized: " << serialized << endl;

    Sorter sorter(new BubbleSort());
    vector<int> data = {5, 2, 8, 1, 9};
    sorter.execute(data);

    return 0;
}
