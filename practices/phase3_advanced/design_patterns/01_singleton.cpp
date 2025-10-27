/*
 * ==================================================================================================
 * Exercise: Singleton Pattern
 * ==================================================================================================
 * Difficulty: Intermediate/Advanced | Time: 40-50 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master Singleton design pattern
 * 2. Understand thread-safe singleton implementations
 * 3. Learn Meyer's Singleton (C++11)
 * 4. Practice lazy vs eager initialization
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Device manager singleton
 * - CUDA context management
 * - Resource pool management
 * - Configuration managers
 * ==================================================================================================
 */

#include <iostream>
#include <mutex>
#include <memory>
using namespace std;

/*
 * EXERCISE 1: Basic Singleton (10 min)
 */

class BasicSingleton {
private:
    static BasicSingleton* instance;
    int data;

    // Private constructor
    BasicSingleton() : data(0) {
        cout << "BasicSingleton constructed" << endl;
    }

    // Delete copy constructor and assignment
    BasicSingleton(const BasicSingleton&) = delete;
    BasicSingleton& operator=(const BasicSingleton&) = delete;

public:
    static BasicSingleton* getInstance() {
        if (instance == nullptr) {
            instance = new BasicSingleton();
        }
        return instance;
    }

    void setData(int val) { data = val; }
    int getData() const { return data; }
};

// Initialize static member
BasicSingleton* BasicSingleton::instance = nullptr;

/*
 * EXERCISE 2: Thread-Safe Singleton (15 min)
 */

class ThreadSafeSingleton {
private:
    static ThreadSafeSingleton* instance;
    static mutex mtx;
    int data;

    ThreadSafeSingleton() : data(0) {
        cout << "ThreadSafeSingleton constructed" << endl;
    }

    ThreadSafeSingleton(const ThreadSafeSingleton&) = delete;
    ThreadSafeSingleton& operator=(const ThreadSafeSingleton&) = delete;

public:
    static ThreadSafeSingleton* getInstance() {
        // Double-checked locking
        if (instance == nullptr) {
            lock_guard<mutex> lock(mtx);
            if (instance == nullptr) {
                instance = new ThreadSafeSingleton();
            }
        }
        return instance;
    }

    void setData(int val) { data = val; }
    int getData() const { return data; }
};

ThreadSafeSingleton* ThreadSafeSingleton::instance = nullptr;
mutex ThreadSafeSingleton::mtx;

/*
 * EXERCISE 3: Meyer's Singleton (C++11) (10 min)
 * Best practice - thread-safe by C++11 standard
 */

class MeyersSingleton {
private:
    int data;

    MeyersSingleton() : data(0) {
        cout << "MeyersSingleton constructed" << endl;
    }

    MeyersSingleton(const MeyersSingleton&) = delete;
    MeyersSingleton& operator=(const MeyersSingleton&) = delete;

public:
    static MeyersSingleton& getInstance() {
        static MeyersSingleton instance;  // Thread-safe in C++11
        return instance;
    }

    void setData(int val) { data = val; }
    int getData() const { return data; }
};

/*
 * EXERCISE 4: Singleton with Smart Pointers (10 min)
 */

class SmartSingleton {
private:
    static unique_ptr<SmartSingleton> instance;
    static once_flag initFlag;
    int data;

    SmartSingleton() : data(0) {
        cout << "SmartSingleton constructed" << endl;
    }

public:
    SmartSingleton(const SmartSingleton&) = delete;
    SmartSingleton& operator=(const SmartSingleton&) = delete;

    static SmartSingleton& getInstance() {
        call_once(initFlag, []() {
            instance.reset(new SmartSingleton());
        });
        return *instance;
    }

    void setData(int val) { data = val; }
    int getData() const { return data; }
};

unique_ptr<SmartSingleton> SmartSingleton::instance = nullptr;
once_flag SmartSingleton::initFlag;

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is the Singleton pattern?
 * A: Ensures a class has only one instance and provides global access point
 *
 * Q2: How to make singleton thread-safe?
 * A: - Double-checked locking with mutex
 *    - Meyer's singleton (static local variable in C++11+)
 *    - std::call_once with std::once_flag
 *
 * Q3: Why delete copy constructor in singleton?
 * A: Prevents creating copies, ensuring only one instance exists
 *
 * Q4: Lazy vs eager initialization?
 * A: Lazy: Instance created on first use (Meyer's)
 *    Eager: Instance created at program start (static member)
 *
 * Q5: Problems with singleton?
 * A: - Global state (testing difficulties)
 *    - Hidden dependencies
 *    - Tight coupling
 *    - Destruction order issues
 *
 * Q6: How to destroy singleton?
 * A: - Smart pointers (automatic)
 *    - Meyer's singleton (automatic at program end)
 *    - Manual cleanup method (error-prone)
 *
 * Q7: Double-checked locking pattern?
 * A: Check if instance exists before locking for performance
 *    if (instance == nullptr) {
 *        lock();
 *        if (instance == nullptr) { create(); }
 *    }
 *
 * Q8: Meyer's singleton advantages?
 * A: - Thread-safe by C++11 standard
 *    - No manual locking
 *    - Automatic cleanup
 *    - Lazy initialization
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - CUDA device manager: Single instance managing all GPU devices
 * - Memory pool singleton: Centralized GPU memory allocation
 * - Logger singleton: Thread-safe logging across CUDA streams
 * - Configuration manager: GPU settings and parameters
 *
 * Example use case:
 * class CudaDeviceManager {
 *     static CudaDeviceManager& getInstance() {
 *         static CudaDeviceManager instance;
 *         return instance;
 *     }
 *     void selectDevice(int id);
 *     cudaDeviceProp getProperties();
 * };
 *
 * COMPILATION: g++ -std=c++11 -pthread 01_singleton.cpp -o singleton
 * ==================================================================================================
 */

int main() {
    cout << "=== Singleton Pattern Practice ===" << endl;

    // Basic Singleton
    BasicSingleton* s1 = BasicSingleton::getInstance();
    BasicSingleton* s2 = BasicSingleton::getInstance();
    cout << "Same instance? " << (s1 == s2) << endl;

    s1->setData(42);
    cout << "s2 data: " << s2->getData() << endl;  // Should be 42

    // Meyer's Singleton (recommended)
    MeyersSingleton& m1 = MeyersSingleton::getInstance();
    MeyersSingleton& m2 = MeyersSingleton::getInstance();
    cout << "Same instance? " << (&m1 == &m2) << endl;

    m1.setData(100);
    cout << "m2 data: " << m2.getData() << endl;  // Should be 100

    return 0;
}
