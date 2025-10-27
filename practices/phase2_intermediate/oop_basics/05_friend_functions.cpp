/*
 * ==================================================================================================
 * Exercise: Friend Functions and Classes in C++
 * ==================================================================================================
 * Difficulty: Intermediate | Time: 35-45 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Understand friend functions and when to use them
 * 2. Master friend classes
 * 3. Learn operator overloading with friends
 * 4. Recognize trade-offs of breaking encapsulation
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Operator overloading for vector/matrix classes (GPU math)
 * - Stream operators (<<, >>) for custom types
 * - Testing frameworks needing access to private members
 * - Performance-critical functions needing direct access
 * ==================================================================================================
 */

#include <iostream>
#include <cmath>

using namespace std;

/*
 * ==================================================================================================
 * THEORY: Friend Functions
 * ==================================================================================================
 *
 * FRIEND FUNCTION:
 * - NOT a member function
 * - Can access private/protected members
 * - Declared inside class with 'friend' keyword
 * - Defined outside class (without 'friend' or '::')
 *
 * When to use:
 * - Operator overloading (<<, >>, +, -, etc.)
 * - Functions needing access to multiple classes
 * - Performance-critical code needing direct access
 *
 * Trade-off:
 * - Breaks encapsulation (use sparingly!)
 * - Creates tight coupling
 * - But: Sometimes necessary and cleaner
 *
 * ==================================================================================================
 */

/*
 * EXERCISE 1: Basic Friend Function (10 min)
 */

// TODO 1.1: Create class with friend function
class Box {
private:
    double width, height, depth;

public:
    Box(double w, double h, double d) : width(w), height(h), depth(d) {}

    // Declare friend function
    friend double getVolume(const Box& box);
};

// TODO: Define friend function (no Box:: prefix!)
// double getVolume(const Box& box) {
//     return box.width * box.height * box.depth;  // Can access private members!
// }

/*
 * EXERCISE 2: Operator Overloading with Friends (15 min)
 */

// TODO 2.1: Overload operators for Vector2D class
class Vector2D {
private:
    double x, y;

public:
    Vector2D(double x = 0, double y = 0) : x(x), y(y) {}

    // Friend functions for operator overloading
    friend Vector2D operator+(const Vector2D& a, const Vector2D& b);
    friend Vector2D operator-(const Vector2D& a, const Vector2D& b);
    friend double operator*(const Vector2D& a, const Vector2D& b);  // Dot product
    friend ostream& operator<<(ostream& os, const Vector2D& v);
};

// TODO: Implement operator overloads
// Vector2D operator+(const Vector2D& a, const Vector2D& b) {
//     return Vector2D(a.x + b.x, a.y + b.y);
// }
//
// ostream& operator<<(ostream& os, const Vector2D& v) {
//     os << "(" << v.x << ", " << v.y << ")";
//     return os;
// }

/*
 * EXERCISE 3: Friend Classes (10 min)
 */

// TODO 3.1: Make one class a friend of another
class Engine;  // Forward declaration

class Car {
private:
    string model;
    int horsePower;

public:
    Car(const string& m, int hp) : model(m), horsePower(hp) {}

    // Declare Engine as friend class
    friend class Engine;
};

class Engine {
public:
    void diagnose(const Car& car) {
        cout << "Diagnosing " << car.model << endl;  // Can access private members
        cout << "HP: " << car.horsePower << endl;
    }
};

/*
 * EXERCISE 4: Stream Operators (10 min)
 */

// TODO 4.1: Overload << and >> operators
class Point {
private:
    int x, y;

public:
    Point(int x = 0, int y = 0) : x(x), y(y) {}

    friend ostream& operator<<(ostream& os, const Point& p);
    friend istream& operator>>(istream& is, Point& p);
};

// TODO: Implement stream operators
// ostream& operator<<(ostream& os, const Point& p) {
//     os << "Point(" << p.x << ", " << p.y << ")";
//     return os;
// }
//
// istream& operator>>(istream& is, Point& p) {
//     is >> p.x >> p.y;
//     return is;
// }

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is a friend function?
 * A: Non-member function that can access private/protected members of a class
 *
 * Q2: Why use friend functions?
 * A: 1) Operator overloading (binary operators like +, -)
 *    2) Functions operating on multiple classes
 *    3) Stream operators (<<, >>)
 *    4) Performance (avoid getter/setter overhead)
 *
 * Q3: Does friend break encapsulation?
 * A: Yes! Use sparingly. It creates tight coupling. But sometimes necessary/cleaner
 *
 * Q4: Can friend functions access private members?
 * A: Yes - that's the entire point! Both private and protected
 *
 * Q5: Is friendship inherited?
 * A: No! Friendship is not inherited, transitive, or reciprocal
 *
 * Q6: When to make a class a friend vs a function?
 * A: Friend class: When entire class needs access (all member functions)
 *    Friend function: When only specific operation needs access
 *
 * Q7: Why use friend for operator<<?
 * A: cout << obj requires operator<<(ostream&, const MyClass&)
 *    This MUST be non-member (cout is on left side)
 *    Friend allows access to private members for printing
 *
 * Q8: Alternative to friend functions?
 * A: Public getter/setter functions. But may be less efficient and verbose
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * ==================================================================================================
 *
 * 1. Math Libraries: Vector/matrix classes use friend operators extensively
 *    - operator+ for vector addition
 *    - operator* for dot/cross product
 *    - Efficient direct access to components
 *
 * 2. Stream Operators: Debug printing of GPU data structures
 *    - Kernel configuration objects
 *    - Device properties
 *    - Matrix/tensor classes
 *
 * 3. Testing: Unit tests as friend classes
 *    - Access private members for verification
 *    - White-box testing of GPU algorithms
 *
 * 4. Performance: Critical path operations
 *    - Direct memory access without virtual call overhead
 *    - Inline-friendly implementations
 *
 * ==================================================================================================
 * COMPILATION: g++ -std=c++17 05_friend_functions.cpp -o friends
 *
 * LEARNING CHECKLIST:
 * ☐ Declare and define friend functions
 * ☐ Overload operators using friends
 * ☐ Create friend classes
 * ☐ Implement stream operators
 * ☐ Understand when to use/avoid friends
 * ☐ Know that friendship isn't inherited
 * ==================================================================================================
 */

int main() {
    cout << "=== Friend Functions Practice ===" << endl;

    // Test your implementations here

    return 0;
}
