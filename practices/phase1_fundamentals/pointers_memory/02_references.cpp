/*
 * Exercise: References
 * Difficulty: Beginner
 * Time: 35-45 minutes
 * Topics: Reference variables, pass by reference, const references, reference vs pointer
 *
 * LEARNING OBJECTIVES:
 * - Understand what references are
 * - Master pass-by-reference vs pass-by-value
 * - Learn const references for efficient parameter passing
 * - Understand reference initialization rules
 * - Know when to use references vs pointers
 * - Understand rvalue references (C++11) basics
 *
 * INTERVIEW RELEVANCE:
 * - References are fundamental to modern C++
 * - Pass-by-reference is constantly asked in interviews
 * - Understanding when to use const& is critical
 * - Move semantics (rvalue references) is important
 * - Function return values and references are common topics
 */

#include <iostream>
#include <string>
#include <vector>

int main() {
    std::cout << "=== References Exercises ===\n\n";

    // ========================================================================
    // EXERCISE 1: Basic References (5 min)
    // ========================================================================
    std::cout << "Exercise 1: Basic References\n";
    std::cout << "-----------------------------\n";

    // TODO 1.1: Declare and initialize a reference
    int x = 42;
    int& ref = x;  // ref is an alias for x

    std::cout << "x = " << x << "\n";
    std::cout << "ref = " << ref << "\n";
    std::cout << "&x = " << &x << "\n";
    std::cout << "&ref = " << &ref << " (same address!)\n";

    // TODO 1.2: Modify through reference
    ref = 100;
    std::cout << "\nAfter ref = 100:\n";
    std::cout << "x = " << x << " (also changed!)\n";
    std::cout << "ref = " << ref << "\n";

    // TODO 1.3: References MUST be initialized
    // int& bad;        // ERROR: must be initialized
    // int& good = x;   // OK

    // TODO 1.4: References cannot be rebound
    int y = 50;
    // ref = y;  // This assigns y's VALUE to ref (and thus to x)
    //           // It does NOT make ref refer to y!
    std::cout << "\nAfter ref = y (where y=50):\n";
    std::cout << "x = " << x << "\n";   // x is now 50
    std::cout << "&ref = " << &ref << " (still points to x)\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 2: Pass by Value vs Pass by Reference (10 min)
    // ========================================================================
    std::cout << "Exercise 2: Pass by Value vs Reference\n";
    std::cout << "---------------------------------------\n";

    // TODO 2.1: Pass by value (makes a copy)
    auto passByValue = [](int x) {
        x = 999;  // Only changes the copy
    };

    int val1 = 10;
    std::cout << "Before passByValue: " << val1 << "\n";
    passByValue(val1);
    std::cout << "After passByValue: " << val1 << " (unchanged)\n";

    // TODO 2.2: Pass by reference (no copy)
    auto passByReference = [](int& x) {
        x = 999;  // Changes the original
    };

    int val2 = 10;
    std::cout << "\nBefore passByReference: " << val2 << "\n";
    passByReference(val2);
    std::cout << "After passByReference: " << val2 << " (changed!)\n";

    // TODO 2.3: Demonstrate with large object
    std::vector<int> largeVec(1000000, 42);

    auto byValue = [](std::vector<int> v) {
        // Copies 1 million ints - SLOW!
        return v.size();
    };

    auto byReference = [](const std::vector<int>& v) {
        // No copy - FAST!
        return v.size();
    };

    std::cout << "\nPassing large vector by value: size = " << byValue(largeVec) << "\n";
    std::cout << "Passing large vector by ref: size = " << byReference(largeVec) << "\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 3: Const References (10 min)
    // ========================================================================
    std::cout << "Exercise 3: Const References\n";
    std::cout << "-----------------------------\n";

    // TODO 3.1: Const reference (can't modify)
    int original = 100;
    const int& constRef = original;

    std::cout << "constRef = " << constRef << "\n";
    // constRef = 200;  // ERROR: can't modify through const reference

    original = 200;  // OK to modify original
    std::cout << "After original = 200, constRef = " << constRef << "\n";

    // TODO 3.2: Const reference to literal
    const int& literalRef = 42;  // OK! Creates temporary
    std::cout << "literalRef = " << literalRef << "\n";

    // int& badRef = 42;  // ERROR: non-const ref can't bind to literal

    // TODO 3.3: Best practice: pass by const reference
    auto printString = [](const std::string& str) {
        // Efficient (no copy) and safe (can't modify)
        std::cout << "String: " << str << "\n";
    };

    std::string message = "Hello, References!";
    printString(message);

    // TODO 3.4: When to use const reference
    /*
     * Use const& for:
     * - Large objects (strings, vectors, etc.)
     * - When you don't need to modify the parameter
     * - Function parameters (default choice for objects)
     *
     * Use value for:
     * - Small types (int, char, pointers, etc.)
     * - When you need a copy anyway
     */

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 4: References vs Pointers (10 min)
    // ========================================================================
    std::cout << "Exercise 4: References vs Pointers\n";
    std::cout << "-----------------------------------\n";

    int a = 10;
    int b = 20;

    // Reference
    int& r = a;
    r = b;  // Assigns b's value to a (a becomes 20)
    std::cout << "Reference: a = " << a << " (value copied from b)\n";

    a = 10; // Reset

    // Pointer
    int* p = &a;
    p = &b;  // Changes what p points to
    std::cout << "Pointer: a = " << a << " (unchanged, pointer now points to b)\n";
    std::cout << "*p = " << *p << " (value of b)\n";

    // TODO 4.1: Key differences
    /*
     * References:
     * - Must be initialized
     * - Cannot be null
     * - Cannot be rebound
     * - Cleaner syntax (automatic dereferencing)
     * - Preferred in modern C++ when possible
     *
     * Pointers:
     * - Can be uninitialized (dangerous!)
     * - Can be null
     * - Can be reassigned
     * - Explicit dereferencing
     * - Needed for: dynamic memory, arrays, optional parameters
     */

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 5: Returning References (5 min)
    // ========================================================================
    std::cout << "Exercise 5: Returning References\n";
    std::cout << "---------------------------------\n";

    // TODO 5.1: Return reference to static/global
    auto getGlobal = []() -> int& {
        static int global = 100;
        return global;  // OK: static has lifetime beyond function
    };

    getGlobal() = 200;  // Can assign to returned reference!
    std::cout << "Global after modification: " << getGlobal() << "\n";

    // TODO 5.2: NEVER return reference to local
    // auto dangerous = []() -> int& {
    //     int local = 42;
    //     return local;  // DANGLING REFERENCE! local is destroyed
    // };

    // TODO 5.3: Return const reference (common for getters)
    class Container {
        std::vector<int> data = {1, 2, 3};
    public:
        const std::vector<int>& getData() const {
            return data;  // OK: data outlives function
        }
    };

    Container c;
    std::cout << "Container size: " << c.getData().size() << "\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 6: Rvalue References (C++11) (5 min)
    // ========================================================================
    std::cout << "Exercise 6: Rvalue References\n";
    std::cout << "------------------------------\n";

    // TODO 6.1: Lvalue vs Rvalue
    int lvalue = 10;  // lvalue: has a name, can take address
    // int rvalue = 20 + 30;  // 20+30 is rvalue: temporary, no address

    // TODO 6.2: Lvalue reference (normal reference)
    int& lref = lvalue;  // OK
    // int& lref2 = 42;  // ERROR: can't bind to rvalue

    // TODO 6.3: Rvalue reference
    int&& rref = 42;  // OK: binds to rvalue
    // int&& rref2 = lvalue;  // ERROR: can't bind to lvalue

    std::cout << "Rvalue reference: " << rref << "\n";

    // TODO 6.4: Move semantics (brief intro)
    std::vector<int> vec1 = {1, 2, 3};
    std::vector<int> vec2 = std::move(vec1);  // Moves data, not copy

    std::cout << "After move:\n";
    std::cout << "  vec1 size: " << vec1.size() << " (moved from)\n";
    std::cout << "  vec2 size: " << vec2.size() << " (moved to)\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 7: Practical Applications (5 min)
    // ========================================================================
    std::cout << "Exercise 7: Practical Applications\n";
    std::cout << "-----------------------------------\n";

    // TODO 7.1: Swap using references
    auto swap = [](int& a, int& b) {
        int temp = a;
        a = b;
        b = temp;
    };

    int x1 = 5, y1 = 10;
    std::cout << "Before swap: x=" << x1 << ", y=" << y1 << "\n";
    swap(x1, y1);
    std::cout << "After swap: x=" << x1 << ", y=" << y1 << "\n";

    // TODO 7.2: Return multiple values using references
    auto divmod = [](int dividend, int divisor, int& quotient, int& remainder) {
        quotient = dividend / divisor;
        remainder = dividend % divisor;
    };

    int q, r;
    divmod(17, 5, q, r);
    std::cout << "\n17 / 5 = " << q << " remainder " << r << "\n";

    // TODO 7.3: Range-based for loop with references
    std::vector<int> nums = {1, 2, 3, 4, 5};

    std::cout << "\nOriginal: ";
    for (int n : nums) std::cout << n << " ";

    // Modify through reference
    for (int& n : nums) {
        n *= 2;
    }

    std::cout << "\nDoubled: ";
    for (int n : nums) std::cout << n << " ";
    std::cout << "\n";

    std::cout << "\n";

    return 0;
}

// ========================================================================
// COMMON INTERVIEW QUESTIONS
// ========================================================================
/*
 * Q1: What is a reference?
 * A: A reference is an alias for another variable. Once initialized,
 *    it always refers to the same variable and cannot be rebound.
 *
 *    Key properties:
 *    - Must be initialized
 *    - Cannot be null
 *    - Cannot be rebound
 *    - Same address as the variable it refers to
 *    - Automatic dereferencing
 *
 *    Example:
 *    int x = 42;
 *    int& ref = x;  // ref is an alias for x
 *    ref = 100;     // Changes x to 100
 *
 * Q2: What's the difference between reference and pointer?
 * A: References:
 *    - Must be initialized: int& r = x;
 *    - Cannot be null
 *    - Cannot be rebound
 *    - Syntax: automatic dereferencing
 *    - Safer, preferred when possible
 *
 *    Pointers:
 *    - Can be uninitialized
 *    - Can be null
 *    - Can be reassigned
 *    - Syntax: explicit dereferencing with *
 *    - Needed for: dynamic memory, arrays, optional values
 *
 * Q3: When should you use const reference parameters?
 * A: Use const& for function parameters when:
 *    - The object is large (string, vector, custom objects)
 *    - You don't need to modify the parameter
 *    - You want to avoid copying for performance
 *
 *    Example:
 *    void processData(const std::vector<int>& data) {
 *        // No copy, can't modify - perfect!
 *    }
 *
 *    Don't use for:
 *    - Small types (int, char, pointers) - pass by value is fine
 *    - When you need to modify the parameter (use non-const&)
 *
 * Q4: Can you have a reference to a pointer?
 * A: Yes! int*& refToPtr;
 *
 *    Example:
 *    int x = 42;
 *    int* p = &x;
 *    int*& rp = p;  // Reference to pointer
 *
 *    rp = nullptr;  // Changes p to nullptr
 *
 *    Use case: Modifying a pointer in a function
 *    void allocate(int*& ptr) {
 *        ptr = new int(42);  // Modifies the original pointer
 *    }
 *
 * Q5: What's an rvalue reference?
 * A: Rvalue reference (T&&) binds to temporary objects (rvalues).
 *
 *    Lvalue: has a name, can take address
 *    int x = 10;  // x is lvalue
 *
 *    Rvalue: temporary, no name
 *    10 + 20;     // result is rvalue
 *
 *    Rvalue reference:
 *    int&& rr = 42;           // OK
 *    int&& rr2 = getValue();  // OK if getValue() returns by value
 *
 *    Used for:
 *    - Move semantics (avoid copying)
 *    - Perfect forwarding
 *
 * Q6: What happens if you return a reference to a local variable?
 * A: UNDEFINED BEHAVIOR! The local variable is destroyed when the
 *    function returns, leaving a dangling reference.
 *
 *    BAD:
 *    int& dangerous() {
 *        int local = 42;
 *        return local;  // local is destroyed!
 *    }
 *
 *    GOOD:
 *    int& safe() {
 *        static int s = 42;
 *        return s;  // static survives function
 *    }
 *
 *    Or return by value:
 *    int safe2() {
 *        int local = 42;
 *        return local;  // Copy is made
 *    }
 *
 * Q7: Can references be null?
 * A: NO! References must always refer to a valid object.
 *
 *    int& ref;  // ERROR: must be initialized
 *
 *    However, you can create a reference to a dereferenced null pointer:
 *    int* p = nullptr;
 *    int& r = *p;  // UNDEFINED BEHAVIOR! Compiles but crashes
 *
 *    Always ensure pointer is valid before dereferencing!
 *
 * Q8: What's the difference between these?
 * A: const int& r     → Reference to const int
 *    - Can't modify through reference
 *    - Can bind to temporaries
 *
 *    int& const r     → INVALID (references are always const)
 *    - Doesn't compile
 *    - References can't be rebound anyway
 *
 *    Use const int& for read-only parameters
 */

/*
 * REFERENCES IN GPU PROGRAMMING:
 * ===============================
 *
 * 1. References in Kernel Parameters:
 *    Modern CUDA (11+) supports references in device code:
 *
 *    __device__ void process(float& value) {
 *        value *= 2.0f;
 *    }
 *
 *    Older CUDA: Use pointers instead
 *
 * 2. Host Code:
 *    References work normally on CPU side:
 *
 *    void setupKernel(const Config& config) {
 *        kernel<<<config.blocks, config.threads>>>();
 *    }
 *
 * 3. Pass by Reference for Large Structs:
 *    struct KernelParams {
 *        float* data;
 *        int size;
 *        float threshold;
 *    };
 *
 *    void launchKernel(const KernelParams& params) {
 *        // Efficient: no copy of struct
 *    }
 *
 * 4. Const References for Configuration:
 *    class GPUContext {
 *        const DeviceProperties& getProps() const {
 *            return props;
 *        }
 *    };
 *
 * 5. Move Semantics with GPU Buffers:
 *    class GpuBuffer {
 *        GpuBuffer(GpuBuffer&& other) {
 *            // Move device pointer, don't copy data
 *            d_data = other.d_data;
 *            other.d_data = nullptr;
 *        }
 *    };
 *
 * COMPILATION:
 * ============
 * g++ -std=c++17 -Wall -Wextra 02_references.cpp -o references
 * ./references
 *
 * LEARNING CHECKLIST:
 * ===================
 * ☐ Understand what references are (aliases)
 * ☐ Know that references must be initialized
 * ☐ Can use pass-by-reference vs pass-by-value
 * ☐ Understand const references for efficiency
 * ☐ Know differences between references and pointers
 * ☐ Can safely return references
 * ☐ Understand rvalue references basics
 * ☐ Know when to use references vs pointers
 *
 * NEXT STEPS:
 * ===========
 * - Move to 03_dynamic_memory.cpp
 * - Study move semantics in depth
 * - Learn perfect forwarding
 * - Understand reference collapsing rules
 * - Practice with smart pointers
 */
