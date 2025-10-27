/*
 * ==================================================================================================
 * Exercise: Class Templates in C++
 * ==================================================================================================
 * Difficulty: Intermediate/Advanced | Time: 45-55 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master class template syntax
 * 2. Implement generic data structures
 * 3. Understand template member functions
 * 4. Learn template friend functions
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Generic containers for GPU data
 * - Type-safe kernel parameter wrappers
 * - Smart pointer implementations
 * - Zero-overhead abstractions
 * ==================================================================================================
 */

#include <iostream>
#include <stdexcept>
using namespace std;

/*
 * EXERCISE 1: Basic Class Template (15 min)
 */

template<typename T>
class Array {
private:
    T* data;
    size_t size;

public:
    Array(size_t s) : size(s), data(new T[s]) {}
    ~Array() { delete[] data; }

    T& operator[](size_t i) {
        if (i >= size) throw out_of_range("Index out of range");
        return data[i];
    }

    size_t getSize() const { return size; }
};

/*
 * EXERCISE 2: Multiple Template Parameters (10 min)
 */

template<typename K, typename V>
class Pair {
public:
    K key;
    V value;

    Pair(K k, V v) : key(k), value(v) {}

    void print() const {
        cout << key << " => " << value << endl;
    }
};

/*
 * EXERCISE 3: Template Specialization (15 min)
 */

// Generic Stack
template<typename T>
class Stack {
private:
    T* data;
    int top;
    int capacity;

public:
    Stack(int cap = 10) : capacity(cap), top(-1), data(new T[cap]) {}
    ~Stack() { delete[] data; }

    void push(const T& val) {
        if (top >= capacity - 1) throw overflow_error("Stack full");
        data[++top] = val;
    }

    T pop() {
        if (top < 0) throw underflow_error("Stack empty");
        return data[top--];
    }

    bool isEmpty() const { return top < 0; }
};

// TODO: Specialize for bool (bit-packed)

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is a class template?
 * A: Blueprint for creating classes with different types, resolved at compile time
 *
 * Q2: How to declare template outside class?
 * A: template<typename T> ReturnType ClassName<T>::method() { }
 *
 * Q3: Can class templates have non-template members?
 * A: Yes! Can mix template and non-template members
 *
 * Q4: What is template instantiation?
 * A: Compiler generates actual class code for specific type(s)
 *
 * Q5: Difference between template<class T> and template<typename T>?
 * A: None! Both are equivalent. typename preferred for consistency
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - thrust::device_vector<T> is a class template
 * - Generic GPU memory wrappers
 * - Type-safe kernel launchers
 * - Smart pointers for unified memory
 *
 * COMPILATION: g++ -std=c++17 02_class_templates.cpp -o ctemplates
 * ==================================================================================================
 */

int main() {
    cout << "=== Class Templates Practice ===" << endl;

    Array<int> intArr(5);
    intArr[0] = 10;
    cout << "intArr[0] = " << intArr[0] << endl;

    Pair<string, int> p("Age", 25);
    p.print();

    Stack<double> s;
    s.push(3.14);
    s.push(2.71);
    cout << "Popped: " << s.pop() << endl;

    return 0;
}
