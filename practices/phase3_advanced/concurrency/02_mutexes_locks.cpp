/*
 * ==================================================================================================
 * Exercise: Mutexes and Locks
 * ==================================================================================================
 * Difficulty: Intermediate/Advanced | Time: 45-55 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master mutex for thread synchronization
 * 2. Understand RAII locks (lock_guard, unique_lock)
 * 3. Learn shared_mutex for reader-writer pattern
 * 4. Practice deadlock prevention
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Protecting shared GPU resources
 * - Thread-safe CUDA stream pools
 * - Multi-threaded device memory management
 * - Concurrent access to GPU results
 * ==================================================================================================
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <chrono>
using namespace std;

/*
 * EXERCISE 1: Basic Mutex (10 min)
 */

mutex mtx;
int counter = 0;

void incrementCounter(int iterations) {
    for (int i = 0; i < iterations; i++) {
        mtx.lock();
        counter++;
        mtx.unlock();
    }
}

void basicMutex() {
    const int ITERATIONS = 10000;
    thread t1(incrementCounter, ITERATIONS);
    thread t2(incrementCounter, ITERATIONS);

    t1.join();
    t2.join();

    cout << "Final counter: " << counter << " (expected: " << 2*ITERATIONS << ")" << endl;
}

/*
 * EXERCISE 2: lock_guard (RAII) (10 min)
 */

class BankAccount {
private:
    int balance;
    mutable mutex mtx;

public:
    BankAccount(int initial) : balance(initial) {}

    void deposit(int amount) {
        lock_guard<mutex> lock(mtx);  // Auto lock/unlock
        balance += amount;
        cout << "Deposited " << amount << ", balance: " << balance << endl;
    }

    void withdraw(int amount) {
        lock_guard<mutex> lock(mtx);
        if (balance >= amount) {
            balance -= amount;
            cout << "Withdrew " << amount << ", balance: " << balance << endl;
        } else {
            cout << "Insufficient funds" << endl;
        }
    }

    int getBalance() const {
        lock_guard<mutex> lock(mtx);
        return balance;
    }
};

void lockGuardExample() {
    BankAccount account(1000);

    thread t1([&]() {
        for (int i = 0; i < 3; i++) {
            account.deposit(100);
            this_thread::sleep_for(chrono::milliseconds(100));
        }
    });

    thread t2([&]() {
        for (int i = 0; i < 3; i++) {
            account.withdraw(50);
            this_thread::sleep_for(chrono::milliseconds(100));
        }
    });

    t1.join();
    t2.join();

    cout << "Final balance: " << account.getBalance() << endl;
}

/*
 * EXERCISE 3: unique_lock (Flexible Locking) (15 min)
 */

void uniqueLockExample() {
    mutex mtx;
    int data = 0;

    thread t1([&]() {
        unique_lock<mutex> lock(mtx);
        data = 42;
        cout << "Thread 1 set data to " << data << endl;

        // Can unlock early
        lock.unlock();

        // Do other work without holding lock
        this_thread::sleep_for(chrono::milliseconds(100));

        // Can relock
        lock.lock();
        data += 10;
        cout << "Thread 1 modified data to " << data << endl;
    });

    t1.join();
    cout << "Final data: " << data << endl;
}

/*
 * EXERCISE 4: shared_mutex (Reader-Writer Lock) (15 min)
 */

class DataStore {
private:
    vector<int> data;
    mutable shared_mutex mtx;

public:
    // Multiple readers can access simultaneously
    void read(int threadId) {
        shared_lock<shared_mutex> lock(mtx);
        cout << "Reader " << threadId << " reading data (size=" << data.size() << ")" << endl;
        this_thread::sleep_for(chrono::milliseconds(100));
    }

    // Only one writer at a time, no readers allowed
    void write(int value) {
        unique_lock<shared_mutex> lock(mtx);
        data.push_back(value);
        cout << "Writer added " << value << " (size=" << data.size() << ")" << endl;
        this_thread::sleep_for(chrono::milliseconds(100));
    }
};

void readerWriterExample() {
    DataStore store;

    vector<thread> threads;

    // Create readers
    for (int i = 0; i < 5; i++) {
        threads.emplace_back(&DataStore::read, &store, i);
    }

    // Create writers
    for (int i = 0; i < 2; i++) {
        threads.emplace_back(&DataStore::write, &store, i * 10);
    }

    for (auto& t : threads) {
        t.join();
    }
}

/*
 * EXERCISE 5: Deadlock Prevention (10 min)
 */

mutex mtx1, mtx2;

// BAD: Can cause deadlock
void badLocking(int id) {
    if (id == 1) {
        lock_guard<mutex> lock1(mtx1);
        this_thread::sleep_for(chrono::milliseconds(10));
        lock_guard<mutex> lock2(mtx2);
        cout << "Thread 1 acquired both locks" << endl;
    } else {
        lock_guard<mutex> lock2(mtx2);
        this_thread::sleep_for(chrono::milliseconds(10));
        lock_guard<mutex> lock1(mtx1);
        cout << "Thread 2 acquired both locks" << endl;
    }
}

// GOOD: Use std::lock to acquire multiple locks atomically
void goodLocking(int id) {
    lock(mtx1, mtx2);  // Atomic acquisition
    lock_guard<mutex> lock1(mtx1, adopt_lock);
    lock_guard<mutex> lock2(mtx2, adopt_lock);
    cout << "Thread " << id << " safely acquired both locks" << endl;
}

void deadlockExample() {
    cout << "Good locking (no deadlock):" << endl;
    thread t1(goodLocking, 1);
    thread t2(goodLocking, 2);
    t1.join();
    t2.join();
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is a mutex?
 * A: Mutual exclusion primitive that protects shared data from concurrent access
 *
 * Q2: mutex vs lock?
 * A: mutex: The synchronization primitive itself
 *    lock: RAII wrapper that manages mutex (lock_guard, unique_lock)
 *
 * Q3: lock_guard vs unique_lock?
 * A: lock_guard: Simple RAII, locks on construction, unlocks on destruction
 *    unique_lock: Flexible, can unlock/relock, can be moved, works with condition variables
 *
 * Q4: What is a deadlock?
 * A: Two or more threads waiting for each other to release locks, causing infinite wait
 *
 * Q5: How to prevent deadlock?
 * A: - Always acquire locks in same order
 *    - Use std::lock() to acquire multiple locks atomically
 *    - Use try_lock() with timeout
 *    - Use lock hierarchies
 *
 * Q6: What is shared_mutex?
 * A: Allows multiple readers OR one writer (reader-writer lock)
 *    Use shared_lock for readers, unique_lock for writers
 *
 * Q7: When to use shared_mutex?
 * A: When reads are much more frequent than writes
 *    Multiple readers can access simultaneously
 *
 * Q8: What is adopt_lock?
 * A: Tag indicating lock_guard should adopt already-locked mutex
 *    Used with std::lock()
 *
 * Q9: Performance of mutex?
 * A: - Relatively expensive (syscall in many implementations)
 *    - Can cause thread blocking/context switches
 *    - Consider lock-free alternatives for high contention
 *
 * Q10: Recursive mutex?
 * A: std::recursive_mutex allows same thread to lock multiple times
 *    Must unlock same number of times
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Protecting CUDA stream pools:
 *   class StreamPool {
 *       vector<cudaStream_t> streams;
 *       mutex mtx;
 *   public:
 *       cudaStream_t acquire() {
 *           lock_guard<mutex> lock(mtx);
 *           // Get available stream
 *       }
 *   };
 *
 * - Thread-safe device memory management:
 *   class DeviceMemoryManager {
 *       map<void*, size_t> allocations;
 *       mutex mtx;
 *   public:
 *       void* allocate(size_t size) {
 *           lock_guard<mutex> lock(mtx);
 *           void* ptr;
 *           cudaMalloc(&ptr, size);
 *           allocations[ptr] = size;
 *           return ptr;
 *       }
 *   };
 *
 * - Reader-writer for GPU results:
 *   - Multiple threads read results from device
 *   - Single thread writes new results
 *
 * IMPORTANT: CUDA operations themselves are thread-safe at driver level,
 * but application-level resource management needs synchronization
 *
 * COMPILATION: g++ -std=c++17 -pthread 02_mutexes_locks.cpp -o mutexes
 * ==================================================================================================
 */

int main() {
    cout << "=== Mutexes and Locks Practice ===" << endl;

    cout << "\n1. Basic Mutex:" << endl;
    basicMutex();

    cout << "\n2. lock_guard (RAII):" << endl;
    lockGuardExample();

    cout << "\n3. unique_lock (Flexible):" << endl;
    uniqueLockExample();

    cout << "\n4. shared_mutex (Reader-Writer):" << endl;
    readerWriterExample();

    cout << "\n5. Deadlock Prevention:" << endl;
    deadlockExample();

    return 0;
}
