/*
 * ==================================================================================================
 * Exercise: Atomic Operations
 * ==================================================================================================
 * Difficulty: Advanced | Time: 40-50 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master std::atomic for lock-free operations
 * 2. Understand memory ordering
 * 3. Learn atomic operations vs mutex
 * 4. Practice lock-free programming basics
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Lock-free counters for GPU metrics
 * - Atomic flags for synchronization
 * - High-performance concurrent data structures
 * - Memory ordering in multi-threaded GPU code
 * ==================================================================================================
 */

#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
using namespace std;

/*
 * EXERCISE 1: Basic Atomic Operations (10 min)
 */

atomic<int> atomicCounter(0);
int normalCounter = 0;

void incrementAtomic(int iterations) {
    for (int i = 0; i < iterations; i++) {
        atomicCounter++;  // Thread-safe without mutex
    }
}

void incrementNormal(int iterations) {
    for (int i = 0; i < iterations; i++) {
        normalCounter++;  // NOT thread-safe!
    }
}

void atomicBasics() {
    const int ITERATIONS = 10000;

    // Atomic counter (thread-safe)
    thread t1(incrementAtomic, ITERATIONS);
    thread t2(incrementAtomic, ITERATIONS);
    t1.join();
    t2.join();
    cout << "Atomic counter: " << atomicCounter << " (expected: " << 2*ITERATIONS << ")" << endl;

    // Normal counter (data race!)
    thread t3(incrementNormal, ITERATIONS);
    thread t4(incrementNormal, ITERATIONS);
    t3.join();
    t4.join();
    cout << "Normal counter: " << normalCounter << " (expected: " << 2*ITERATIONS << ")"
         << " - likely wrong due to race!" << endl;
}

/*
 * EXERCISE 2: Atomic Operations (10 min)
 */

void atomicOperations() {
    atomic<int> x(10);

    cout << "Initial: " << x << endl;

    // Load and store
    int val = x.load();
    x.store(20);
    cout << "After store(20): " << x << endl;

    // Fetch and add
    int old = x.fetch_add(5);
    cout << "fetch_add(5): old=" << old << ", new=" << x << endl;

    // Fetch and sub
    old = x.fetch_sub(3);
    cout << "fetch_sub(3): old=" << old << ", new=" << x << endl;

    // Exchange
    old = x.exchange(100);
    cout << "exchange(100): old=" << old << ", new=" << x << endl;

    // Compare and swap
    int expected = 100;
    bool success = x.compare_exchange_strong(expected, 200);
    cout << "compare_exchange_strong(100, 200): " << (success ? "success" : "failed")
         << ", value=" << x << endl;
}

/*
 * EXERCISE 3: Atomic Flag (Spinlock) (10 min)
 */

atomic_flag spinlock = ATOMIC_FLAG_INIT;

void useSpinlock(int id, int iterations) {
    for (int i = 0; i < iterations; i++) {
        // Acquire spinlock
        while (spinlock.test_and_set(atomic_order_acquire)) {
            // Busy-wait (spin)
        }

        // Critical section
        cout << "Thread " << id << " in critical section" << endl;
        this_thread::sleep_for(chrono::milliseconds(10));

        // Release spinlock
        spinlock.clear(atomic_order_release);
    }
}

void spinlockExample() {
    thread t1(useSpinlock, 1, 3);
    thread t2(useSpinlock, 2, 3);

    t1.join();
    t2.join();
}

/*
 * EXERCISE 4: Memory Ordering (10 min)
 */

atomic<bool> ready(false);
atomic<int> data(0);

void writer() {
    data.store(42, memory_order_relaxed);
    ready.store(true, memory_order_release);  // Synchronizes with acquire
}

void reader() {
    while (!ready.load(memory_order_acquire)) {
        // Wait for ready
    }
    cout << "Data: " << data.load(memory_order_relaxed) << endl;  // Will see 42
}

void memoryOrderingExample() {
    thread t1(writer);
    thread t2(reader);

    t1.join();
    t2.join();
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is std::atomic?
 * A: Template class providing lock-free atomic operations on shared variables
 *
 * Q2: When to use atomic vs mutex?
 * A: Atomic: Simple operations (increment, flag), lock-free, better performance
 *    Mutex: Complex critical sections, multiple operations together
 *
 * Q3: What operations are atomic?
 * A: - load, store
 *    - fetch_add, fetch_sub, fetch_and, fetch_or, fetch_xor
 *    - exchange
 *    - compare_exchange_weak, compare_exchange_strong
 *
 * Q4: What is compare_exchange?
 * A: Atomically: if (value == expected) { value = desired; return true; }
 *    else { expected = value; return false; }
 *    Foundation of lock-free algorithms
 *
 * Q5: Memory ordering levels?
 * A: - memory_order_relaxed: No synchronization, only atomicity
 *    - memory_order_acquire: Load-acquire (pairs with release)
 *    - memory_order_release: Store-release (pairs with acquire)
 *    - memory_order_acq_rel: Both acquire and release
 *    - memory_order_seq_cst: Sequential consistency (default, strongest)
 *
 * Q6: What is sequential consistency?
 * A: Default memory order, ensures total order of all atomic operations
 *    Most intuitive but potentially slowest
 *
 * Q7: Acquire-release semantics?
 * A: release (store): All writes before are visible after acquire
 *    acquire (load): All writes after release are visible
 *    Used for synchronization between threads
 *
 * Q8: What is atomic_flag?
 * A: Simplest atomic type, guaranteed lock-free
 *    Only operations: test_and_set, clear
 *    Used for spinlocks
 *
 * Q9: compare_exchange_weak vs strong?
 * A: weak: May fail spuriously (return false even if equal), used in loops
 *    strong: Never fails spuriously, more expensive
 *
 * Q10: Are all atomics lock-free?
 * A: Not always, depends on platform and type size
 *    Check with: x.is_lock_free()
 *
 * Q11: Performance considerations?
 * A: - Atomics faster than mutex for simple operations
 *    - Still expensive (memory barriers, cache coherence)
 *    - Excessive atomic operations can hurt performance
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Atomic counters for GPU task tracking:
 *   atomic<int> kernelsCompleted(0);
 *   void onKernelComplete() {
 *       kernelsCompleted.fetch_add(1);
 *   }
 *
 * - Lock-free flags for synchronization:
 *   atomic<bool> gpuReady(false);
 *   // GPU thread sets flag
 *   // CPU threads check flag without blocking
 *
 * - Performance monitoring:
 *   atomic<uint64_t> totalKernelTime(0);
 *   // Multiple threads update metrics without locks
 *
 * Note: CUDA also has device-side atomics (atomicAdd, atomicCAS, etc.)
 * for synchronization within kernels
 *
 * Memory ordering is crucial for:
 * - Multi-threaded CUDA stream management
 * - Synchronizing CPU and GPU operations
 * - Lock-free work stealing from GPU task queues
 *
 * COMPILATION: g++ -std=c++11 -pthread 04_atomics.cpp -o atomics
 * ==================================================================================================
 */

int main() {
    cout << "=== Atomic Operations Practice ===" << endl;

    cout << "\n1. Atomic vs Normal Counter:" << endl;
    atomicBasics();

    cout << "\n2. Atomic Operations:" << endl;
    atomicOperations();

    cout << "\n3. Spinlock with atomic_flag:" << endl;
    spinlockExample();

    cout << "\n4. Memory Ordering:" << endl;
    memoryOrderingExample();

    // Check if atomics are lock-free
    cout << "\n5. Lock-free status:" << endl;
    cout << "atomic<int> is lock-free: " << atomic<int>().is_lock_free() << endl;
    cout << "atomic<long long> is lock-free: " << atomic<long long>().is_lock_free() << endl;

    return 0;
}
