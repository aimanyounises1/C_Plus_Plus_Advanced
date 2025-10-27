/*
 * ==================================================================================================
 * Exercise: Factory Pattern
 * ==================================================================================================
 * Difficulty: Intermediate/Advanced | Time: 45-55 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master Factory Method pattern
 * 2. Understand Abstract Factory pattern
 * 3. Learn object creation abstraction
 * 4. Practice polymorphic object creation
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Kernel factory for different GPU architectures
 * - Memory allocator factory (device/host/unified)
 * - Stream factory for different execution policies
 * - Driver API abstraction
 * ==================================================================================================
 */

#include <iostream>
#include <memory>
#include <string>
using namespace std;

/*
 * EXERCISE 1: Simple Factory (10 min)
 */

// Product hierarchy
class Shape {
public:
    virtual void draw() const = 0;
    virtual ~Shape() = default;
};

class Circle : public Shape {
public:
    void draw() const override {
        cout << "Drawing Circle" << endl;
    }
};

class Rectangle : public Shape {
public:
    void draw() const override {
        cout << "Drawing Rectangle" << endl;
    }
};

class Triangle : public Shape {
public:
    void draw() const override {
        cout << "Drawing Triangle" << endl;
    }
};

// Simple Factory
class ShapeFactory {
public:
    static unique_ptr<Shape> createShape(const string& type) {
        if (type == "circle") return make_unique<Circle>();
        if (type == "rectangle") return make_unique<Rectangle>();
        if (type == "triangle") return make_unique<Triangle>();
        return nullptr;
    }
};

/*
 * EXERCISE 2: Factory Method Pattern (15 min)
 */

// Product
class Document {
public:
    virtual void open() = 0;
    virtual ~Document() = default;
};

class PDFDocument : public Document {
public:
    void open() override {
        cout << "Opening PDF document" << endl;
    }
};

class WordDocument : public Document {
public:
    void open() override {
        cout << "Opening Word document" << endl;
    }
};

// Creator (Factory Method)
class Application {
public:
    virtual unique_ptr<Document> createDocument() = 0;
    virtual ~Application() = default;

    void newDocument() {
        auto doc = createDocument();
        doc->open();
    }
};

class PDFApplication : public Application {
public:
    unique_ptr<Document> createDocument() override {
        return make_unique<PDFDocument>();
    }
};

class WordApplication : public Application {
public:
    unique_ptr<Document> createDocument() override {
        return make_unique<WordDocument>();
    }
};

/*
 * EXERCISE 3: Abstract Factory Pattern (20 min)
 */

// Abstract Products
class Button {
public:
    virtual void render() const = 0;
    virtual ~Button() = default;
};

class Checkbox {
public:
    virtual void render() const = 0;
    virtual ~Checkbox() = default;
};

// Concrete Products - Windows
class WindowsButton : public Button {
public:
    void render() const override {
        cout << "Rendering Windows button" << endl;
    }
};

class WindowsCheckbox : public Checkbox {
public:
    void render() const override {
        cout << "Rendering Windows checkbox" << endl;
    }
};

// Concrete Products - Mac
class MacButton : public Button {
public:
    void render() const override {
        cout << "Rendering Mac button" << endl;
    }
};

class MacCheckbox : public Checkbox {
public:
    void render() const override {
        cout << "Rendering Mac checkbox" << endl;
    }
};

// Abstract Factory
class GUIFactory {
public:
    virtual unique_ptr<Button> createButton() const = 0;
    virtual unique_ptr<Checkbox> createCheckbox() const = 0;
    virtual ~GUIFactory() = default;
};

// Concrete Factories
class WindowsFactory : public GUIFactory {
public:
    unique_ptr<Button> createButton() const override {
        return make_unique<WindowsButton>();
    }
    unique_ptr<Checkbox> createCheckbox() const override {
        return make_unique<WindowsCheckbox>();
    }
};

class MacFactory : public GUIFactory {
public:
    unique_ptr<Button> createButton() const override {
        return make_unique<MacButton>();
    }
    unique_ptr<Checkbox> createCheckbox() const override {
        return make_unique<MacCheckbox>();
    }
};

// Client code
void renderUI(const GUIFactory& factory) {
    auto button = factory.createButton();
    auto checkbox = factory.createCheckbox();
    button->render();
    checkbox->render();
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is the Factory pattern?
 * A: Creates objects without specifying exact class, promotes loose coupling
 *
 * Q2: Simple Factory vs Factory Method?
 * A: Simple Factory: Static method creates objects (not true pattern)
 *    Factory Method: Subclasses decide which class to instantiate
 *
 * Q3: What is Abstract Factory?
 * A: Creates families of related objects without specifying concrete classes
 *
 * Q4: When to use Factory Method?
 * A: - Class can't anticipate object types to create
 *    - Subclasses should specify object types
 *    - Delegate creation to subclasses
 *
 * Q5: When to use Abstract Factory?
 * A: - System should be independent of how products are created
 *    - Need families of related objects
 *    - Want to enforce constraints on related products
 *
 * Q6: Advantages of Factory patterns?
 * A: - Loose coupling (client doesn't know concrete classes)
 *    - Single Responsibility (creation logic in one place)
 *    - Open/Closed Principle (add new types without modifying existing)
 *    - Encapsulation of object creation
 *
 * Q7: Disadvantages?
 * A: - Increases code complexity
 *    - More classes to maintain
 *    - Can be overkill for simple cases
 *
 * Q8: Factory vs Builder pattern?
 * A: Factory: Creates different types of objects
 *    Builder: Creates complex object step-by-step
 *
 * Q9: How to add new product type?
 * A: Factory Method: Add new Product subclass and Creator subclass
 *    Abstract Factory: Add new ConcreteFactory and Product variants
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Memory allocator factory: Choose device/host/unified memory
 * - Kernel factory: Select optimized kernel for GPU architecture (Volta, Ampere, etc.)
 * - Stream factory: Create different stream types (default, non-blocking, priority)
 * - Algorithm factory: Select CUDA/cuBLAS/cuDNN implementation
 *
 * Example use case:
 * class MemoryFactory {
 * public:
 *     virtual void* allocate(size_t size) = 0;
 *     virtual void deallocate(void* ptr) = 0;
 * };
 *
 * class DeviceMemoryFactory : public MemoryFactory {
 *     void* allocate(size_t size) override {
 *         void* ptr;
 *         cudaMalloc(&ptr, size);
 *         return ptr;
 *     }
 * };
 *
 * class UnifiedMemoryFactory : public MemoryFactory {
 *     void* allocate(size_t size) override {
 *         void* ptr;
 *         cudaMallocManaged(&ptr, size);
 *         return ptr;
 *     }
 * };
 *
 * COMPILATION: g++ -std=c++14 02_factory.cpp -o factory
 * ==================================================================================================
 */

int main() {
    cout << "=== Factory Pattern Practice ===" << endl;

    // Simple Factory
    cout << "\n1. Simple Factory:" << endl;
    auto circle = ShapeFactory::createShape("circle");
    auto rect = ShapeFactory::createShape("rectangle");
    circle->draw();
    rect->draw();

    // Factory Method
    cout << "\n2. Factory Method:" << endl;
    unique_ptr<Application> pdfApp = make_unique<PDFApplication>();
    unique_ptr<Application> wordApp = make_unique<WordApplication>();
    pdfApp->newDocument();
    wordApp->newDocument();

    // Abstract Factory
    cout << "\n3. Abstract Factory:" << endl;
    WindowsFactory winFactory;
    MacFactory macFactory;

    cout << "Windows UI:" << endl;
    renderUI(winFactory);

    cout << "Mac UI:" << endl;
    renderUI(macFactory);

    return 0;
}
