/*
 * ==================================================================================================
 * Exercise: Operator Overloading in C++
 * ==================================================================================================
 * Difficulty: Intermediate/Advanced | Time: 50-60 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master operator overloading syntax
 * 2. Overload arithmetic, comparison, stream operators
 * 3. Understand member vs non-member overloads
 * 4. Learn best practices and limitations
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Vector/Matrix classes for GPU math
 * - Custom complex number types
 * - Stream operators for debugging GPU data
 * - Intuitive APIs for GPU programming
 * ==================================================================================================
 */

#include <iostream>
#include <cmath>
using namespace std;

/*
 * THEORY: Operator Overloading
 *
 * SYNTAX: ReturnType operator@(parameters)
 * where @ is the operator (+, -, *, etc.)
 *
 * MEMBER vs NON-MEMBER:
 * - Member: class.operator@() - implicit 'this' is left operand
 * - Non-member: operator@(left, right) - both operands explicit
 *
 * BEST PRACTICES:
 * - Binary operators (+, -, *, /): non-member for symmetry
 * - Unary operators (++, --): member
 * - Assignment (=, +=, etc.): member (must be member)
 * - Stream operators (<<, >>): non-member (often friend)
 * - Comparison: non-member for consistency
 * ==================================================================================================
 */

/*
 * EXERCISE 1: Arithmetic Operators (15 min)
 */

class Vector2D {
private:
    double x, y;

public:
    Vector2D(double x = 0, double y = 0) : x(x), y(y) {}

    // Addition (non-member for symmetry)
    friend Vector2D operator+(const Vector2D& a, const Vector2D& b) {
        return Vector2D(a.x + b.x, a.y + b.y);
    }

    // Subtraction
    friend Vector2D operator-(const Vector2D& a, const Vector2D& b) {
        return Vector2D(a.x - b.x, a.y - b.y);
    }

    // Scalar multiplication
    friend Vector2D operator*(const Vector2D& v, double scalar) {
        return Vector2D(v.x * scalar, v.y * scalar);
    }

    friend Vector2D operator*(double scalar, const Vector2D& v) {
        return v * scalar;  // Reuse above
    }

    // Dot product
    friend double operator*(const Vector2D& a, const Vector2D& b) {
        return a.x * b.x + a.y * b.y;
    }

    // TODO: Implement division by scalar

    // Stream output
    friend ostream& operator<<(ostream& os, const Vector2D& v) {
        os << "(" << v.x << ", " << v.y << ")";
        return os;
    }
};

/*
 * EXERCISE 2: Comparison Operators (10 min)
 */

class Complex {
private:
    double real, imag;

public:
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}

    // Equality
    friend bool operator==(const Complex& a, const Complex& b) {
        return a.real == b.real && a.imag == b.imag;
    }

    // Inequality
    friend bool operator!=(const Complex& a, const Complex& b) {
        return !(a == b);
    }

    // TODO: Implement <, >, <=, >= based on magnitude

    double magnitude() const {
        return sqrt(real * real + imag * imag);
    }

    friend ostream& operator<<(ostream& os, const Complex& c) {
        os << c.real;
        if (c.imag >= 0) os << "+";
        os << c.imag << "i";
        return os;
    }
};

/*
 * EXERCISE 3: Increment/Decrement (10 min)
 */

class Counter {
private:
    int count;

public:
    Counter(int c = 0) : count(c) {}

    // Pre-increment: ++obj
    Counter& operator++() {
        ++count;
        return *this;
    }

    // Post-increment: obj++
    Counter operator++(int) {  // int is dummy parameter
        Counter temp = *this;
        ++count;
        return temp;
    }

    // TODO: Implement pre/post decrement

    int getCount() const { return count; }
};

/*
 * EXERCISE 4: Assignment Operators (10 min)
 */

class String {
private:
    char* data;
    size_t len;

public:
    String(const char* s = "") {
        len = strlen(s);
        data = new char[len + 1];
        strcpy(data, s);
    }

    ~String() { delete[] data; }

    // Copy assignment (must be member)
    String& operator=(const String& other) {
        if (this != &other) {  // Self-assignment check
            delete[] data;
            len = other.len;
            data = new char[len + 1];
            strcpy(data, other.data);
        }
        return *this;
    }

    // TODO: Implement += operator

    friend ostream& operator<<(ostream& os, const String& s) {
        os << s.data;
        return os;
    }
};

/*
 * EXERCISE 5: Subscript and Function Call (10 min)
 */

class Matrix {
private:
    double data[3][3];

public:
    Matrix() {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                data[i][j] = 0;
    }

    // Subscript operator (must be member)
    double* operator[](int row) {
        return data[row];
    }

    const double* operator[](int row) const {
        return data[row];
    }

    // Function call operator (must be member)
    double& operator()(int row, int col) {
        return data[row][col];
    }
};

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: Which operators cannot be overloaded?
 * A: :: (scope), . (member access), .* (member pointer), ?: (ternary), sizeof, typeid
 *
 * Q2: Which operators must be member functions?
 * A: = (assignment), [] (subscript), () (function call), -> (member access)
 *
 * Q3: Why make binary operators non-member?
 * A: For symmetry: 2 * vec and vec * 2 both work. Member only supports vec * 2
 *
 * Q4: What is the dummy int in operator++(int)?
 * A: Distinguishes post-increment from pre-increment. Compiler passes 0
 *
 * Q5: How to make operator<< work with cout?
 * A: Friend non-member function: friend ostream& operator<<(ostream&, const T&)
 *
 * Q6: Can you change operator precedence?
 * A: No! Overloaded operators keep original precedence and associativity
 *
 * Q7: Should all operators be overloaded?
 * A: No! Only when natural and intuitive. Don't surprise users
 *
 * Q8: Difference between operator= and copy constructor?
 * A: Copy constructor: Initialize new object from existing
 *    operator=: Assign to already-existing object
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Math libraries: Vector/Matrix operations with intuitive syntax
 * - Complex numbers for FFT on GPU
 * - Stream operators for debugging kernel outputs
 * - Custom types with natural math operations
 *
 * Example: vec3 a, b; vec3 c = a + b * 2.0f;  // Natural GPU math
 *
 * COMPILATION: g++ -std=c++17 05_operator_overloading.cpp -o operators
 * ==================================================================================================
 */

int main() {
    cout << "=== Operator Overloading Practice ===" << endl;

    Vector2D v1(3, 4), v2(1, 2);
    Vector2D v3 = v1 + v2;
    cout << "v1 + v2 = " << v3 << endl;
    cout << "v1 * v2 = " << (v1 * v2) << endl;  // Dot product

    Counter cnt(5);
    cout << "Count: " << (++cnt).getCount() << endl;
    cout << "Count: " << (cnt++).getCount() << endl;
    cout << "Count: " << cnt.getCount() << endl;

    return 0;
}
