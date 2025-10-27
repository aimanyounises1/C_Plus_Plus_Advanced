/*
 * ==================================================================================================
 * Exercise: std::thread and Thread Management
 * ==================================================================================================
 * Difficulty: Intermediate/Advanced | Time: 45-55 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master std::thread creation and management
 * 2. Understand join() vs detach()
 * 3. Learn thread argument passing
 * 4. Practice thread-safe programming basics
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - CPU-side multi-threading for GPU management
 * - Concurrent CUDA stream management
 * - Host-side data preprocessing
 * - Asynchronous kernel launching
 * ==================================================================================================
 */

#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <string>
using namespace std;

/*
 * EXERCISE 1: Basic Thread Creation (10 min)
 */

void printMessage(const string& msg, int count) {
    for (int i = 0; i < count; i++) {
        cout << msg << " " << i << endl;
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}

void threadBasics() {
    cout << "Creating threads..." << endl;

    // Create and start thread
    thread t1(printMessage, "Thread-1", 3);
    thread t2(printMessage, "Thread-2", 3);

    // Wait for threads to finish
    t1.join();
    t2.join();

    cout << "All threads completed" << endl;
}

/*
 * EXERCISE 2: Lambda Threads (10 min)
 */

void lambdaThreads() {
    int x = 10;

    // Thread with lambda
    thread t1([x]() {
        cout << "Lambda thread, x = " << x << endl;
    });

    // Capture by reference (be careful!)
    thread t2([&x]() {
        x += 5;
        cout << "Modified x = " << x << endl;
    });

    t1.join();
    t2.join();

    cout << "Final x = " << x << endl;
}

/*
 * EXERCISE 3: Thread with Member Functions (10 min)
 */

class Worker {
private:
    int id;
public:
    Worker(int i) : id(i) {}

    void doWork(int iterations) {
        for (int i = 0; i < iterations; i++) {
            cout << "Worker " << id << " iteration " << i << endl;
            this_thread::sleep_for(chrono::milliseconds(100));
        }
    }
};

void memberFunctionThreads() {
    Worker w1(1);
    Worker w2(2);

    // Pass member function and object
    thread t1(&Worker::doWork, &w1, 3);
    thread t2(&Worker::doWork, &w2, 3);

    t1.join();
    t2.join();
}

/*
 * EXERCISE 4: Thread Pool Pattern (15 min)
 */

void processTask(int taskId) {
    cout << "Processing task " << taskId
         << " on thread " << this_thread::get_id() << endl;
    this_thread::sleep_for(chrono::milliseconds(200));
}

void threadPoolDemo() {
    const int NUM_TASKS = 10;
    const int NUM_THREADS = 4;

    vector<thread> pool;

    // Create thread pool
    for (int i = 0; i < NUM_THREADS; i++) {
        pool.emplace_back(processTask, i);
    }

    // Wait for all threads
    for (auto& t : pool) {
        t.join();
    }
}

/*
 * EXERCISE 5: Thread Detachment (10 min)
 */

void backgroundTask(int id) {
    this_thread::sleep_for(chrono::seconds(1));
    cout << "Background task " << id << " completed" << endl;
}

void detachExample() {
    cout << "Starting detached thread" << endl;

    thread t(backgroundTask, 1);

    // Detach: thread runs independently
    t.detach();

    cout << "Main thread continues..." << endl;
    this_thread::sleep_for(chrono::milliseconds(1500));
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is std::thread?
 * A: C++11 class representing single thread of execution
 *
 * Q2: join() vs detach()?
 * A: join(): Wait for thread to finish (blocking)
 *    detach(): Thread runs independently (daemon thread)
 *    Must call one or the other before thread object destroyed
 *
 * Q3: What happens if thread object destroyed without join/detach?
 * A: std::terminate() is called, program aborts
 *
 * Q4: How to pass arguments to thread?
 * A: Pass after function: thread t(func, arg1, arg2, ...)
 *    Arguments are copied by default
 *    Use std::ref() to pass by reference
 *
 * Q5: Can you pass arguments by reference?
 * A: Yes, use std::ref():
 *    thread t(func, std::ref(x))
 *
 * Q6: How to get thread ID?
 * A: t.get_id() or this_thread::get_id()
 *
 * Q7: How many threads can you create?
 * A: Hardware-dependent, use:
 *    thread::hardware_concurrency()
 *
 * Q8: Can you move threads?
 * A: Yes, thread is movable but not copyable:
 *    thread t2 = move(t1);
 *
 * Q9: How to handle exceptions in threads?
 * A: - Catch in thread function
 *    - Use std::promise/future for exception propagation
 *    - Store exception_ptr and rethrow
 *
 * Q10: Thread safety concerns?
 * A: - Shared data needs synchronization (mutex)
 *    - Race conditions on shared variables
 *    - Data races cause undefined behavior
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Multi-threaded CUDA stream management:
 *   - Each CPU thread manages separate CUDA stream
 *   - Concurrent kernel launches from multiple threads
 *   - Overlap computation and data transfer
 *
 * - Host-side preprocessing:
 *   - One thread reads data, another preprocesses, another launches GPU
 *   - Pipeline pattern for continuous GPU feeding
 *
 * - Multi-GPU programming:
 *   - Each thread manages one GPU device
 *   - Parallel data distribution across GPUs
 *
 * Example use case:
 * void launchKernel(int deviceId, float* data, int size) {
 *     cudaSetDevice(deviceId);
 *     // Launch kernel on this device
 * }
 *
 * void multiGPU() {
 *     vector<thread> threads;
 *     for (int i = 0; i < numGPUs; i++) {
 *         threads.emplace_back(launchKernel, i, data[i], sizes[i]);
 *     }
 *     for (auto& t : threads) t.join();
 * }
 *
 * IMPORTANT CUDA NOTE:
 * - CUDA context is thread-local
 * - Each thread needs its own CUDA context or proper synchronization
 * - cudaSetDevice() should be called per thread
 *
 * COMPILATION: g++ -std=c++11 -pthread 01_threads.cpp -o threads
 * ==================================================================================================
 */

int main() {
    cout << "=== Thread Practice ===" << endl;
    cout << "Hardware concurrency: " << thread::hardware_concurrency() << endl;

    cout << "\n1. Basic Threads:" << endl;
    threadBasics();

    cout << "\n2. Lambda Threads:" << endl;
    lambdaThreads();

    cout << "\n3. Member Function Threads:" << endl;
    memberFunctionThreads();

    cout << "\n4. Thread Pool:" << endl;
    threadPoolDemo();

    cout << "\n5. Detached Thread:" << endl;
    detachExample();

    return 0;
}
