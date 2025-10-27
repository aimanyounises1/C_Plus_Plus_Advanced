/*
 * ==================================================================================================
 * Exercise: Condition Variables
 * ==================================================================================================
 * Difficulty: Advanced | Time: 45-55 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master condition_variable for thread coordination
 * 2. Understand wait(), notify_one(), notify_all()
 * 3. Learn producer-consumer pattern
 * 4. Practice spurious wakeup handling
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Synchronizing CPU threads with GPU operations
 * - Producer-consumer for GPU work queues
 * - Waiting for kernel completion
 * - Pipeline synchronization
 * ==================================================================================================
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <chrono>
using namespace std;

/*
 * EXERCISE 1: Basic Condition Variable (15 min)
 */

mutex mtx;
condition_variable cv;
bool ready = false;

void worker(int id) {
    unique_lock<mutex> lock(mtx);
    cout << "Worker " << id << " waiting..." << endl;

    // Wait until ready is true
    cv.wait(lock, []{ return ready; });

    cout << "Worker " << id << " processing!" << endl;
}

void conditionVariableBasics() {
    thread workers[3];
    for (int i = 0; i < 3; i++) {
        workers[i] = thread(worker, i);
    }

    this_thread::sleep_for(chrono::seconds(1));

    {
        lock_guard<mutex> lock(mtx);
        ready = true;
        cout << "Main: Notifying workers..." << endl;
    }
    cv.notify_all();  // Wake up all waiting threads

    for (auto& t : workers) {
        t.join();
    }
}

/*
 * EXERCISE 2: Producer-Consumer Pattern (20 min)
 */

queue<int> dataQueue;
mutex queueMtx;
condition_variable queueCV;
bool done = false;

void producer(int id, int count) {
    for (int i = 0; i < count; i++) {
        int value = id * 100 + i;

        {
            lock_guard<mutex> lock(queueMtx);
            dataQueue.push(value);
            cout << "Producer " << id << " produced: " << value << endl;
        }

        queueCV.notify_one();  // Notify one consumer
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}

void consumer(int id) {
    while (true) {
        unique_lock<mutex> lock(queueMtx);

        // Wait until queue has data or production is done
        queueCV.wait(lock, []{ return !dataQueue.empty() || done; });

        // Process all available data
        while (!dataQueue.empty()) {
            int value = dataQueue.front();
            dataQueue.pop();
            lock.unlock();  // Release lock while processing

            cout << "Consumer " << id << " consumed: " << value << endl;
            this_thread::sleep_for(chrono::milliseconds(150));

            lock.lock();  // Reacquire lock
        }

        if (done && dataQueue.empty()) {
            break;
        }
    }
}

void producerConsumerExample() {
    const int NUM_PRODUCERS = 2;
    const int NUM_CONSUMERS = 3;
    const int ITEMS_PER_PRODUCER = 5;

    vector<thread> threads;

    // Start consumers
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        threads.emplace_back(consumer, i);
    }

    // Start producers
    for (int i = 0; i < NUM_PRODUCERS; i++) {
        threads.emplace_back(producer, i, ITEMS_PER_PRODUCER);
    }

    // Wait for producers to finish
    for (int i = NUM_CONSUMERS; i < threads.size(); i++) {
        threads[i].join();
    }

    // Signal consumers to stop
    {
        lock_guard<mutex> lock(queueMtx);
        done = true;
    }
    queueCV.notify_all();

    // Wait for consumers
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        threads[i].join();
    }
}

/*
 * EXERCISE 3: Timed Waits (10 min)
 */

void timedWaitExample() {
    mutex mtx;
    condition_variable cv;
    bool ready = false;

    thread worker([&]() {
        unique_lock<mutex> lock(mtx);
        cout << "Worker: Waiting with timeout..." << endl;

        auto status = cv.wait_for(lock, chrono::seconds(2), [&]{ return ready; });

        if (status) {
            cout << "Worker: Condition met!" << endl;
        } else {
            cout << "Worker: Timeout!" << endl;
        }
    });

    // Don't signal, let it timeout
    worker.join();
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is a condition variable?
 * A: Synchronization primitive that allows threads to wait until notified
 *    Used with mutex to avoid busy-waiting
 *
 * Q2: Why need mutex with condition variable?
 * A: - Protect shared state being checked
 *    - Prevent race between check and wait
 *    - Required by condition_variable API
 *
 * Q3: What is spurious wakeup?
 * A: Thread wakes from wait() even though no notify was called
 *    Always use predicate: cv.wait(lock, []{ return condition; })
 *
 * Q4: notify_one() vs notify_all()?
 * A: notify_one(): Wake one waiting thread (more efficient)
 *    notify_all(): Wake all waiting threads
 *    Use notify_all when multiple threads need to check condition
 *
 * Q5: Why use unique_lock instead of lock_guard?
 * A: condition_variable needs to unlock/relock mutex during wait
 *    unique_lock supports manual unlock/lock
 *    lock_guard cannot be unlocked manually
 *
 * Q6: What happens in wait()?
 * A: 1. Atomically unlock mutex
 *    2. Block thread
 *    3. When notified, reacquire mutex
 *    4. Check predicate
 *    5. Return if true, else repeat
 *
 * Q7: Producer-Consumer problem?
 * A: Pattern where producers add items to queue, consumers remove items
 *    Use condition variable to notify when queue has data
 *    Avoid busy-waiting and ensure thread-safe access
 *
 * Q8: wait() vs wait_for() vs wait_until()?
 * A: wait(): Wait indefinitely
 *    wait_for(): Wait for duration (e.g., 2 seconds)
 *    wait_until(): Wait until time point
 *
 * Q9: Common mistakes?
 * A: - Not using predicate (spurious wakeups)
 *    - Holding lock while doing expensive work
 *    - Deadlock from wrong lock order
 *    - Forgetting to notify
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - GPU work queue synchronization:
 *   class GPUWorkQueue {
 *       queue<GPUTask> tasks;
 *       mutex mtx;
 *       condition_variable cv;
 *   public:
 *       void addTask(GPUTask task) {
 *           {
 *               lock_guard<mutex> lock(mtx);
 *               tasks.push(task);
 *           }
 *           cv.notify_one();
 *       }
 *
 *       GPUTask getTask() {
 *           unique_lock<mutex> lock(mtx);
 *           cv.wait(lock, [this]{ return !tasks.empty(); });
 *           GPUTask task = tasks.front();
 *           tasks.pop();
 *           return task;
 *       }
 *   };
 *
 * - Waiting for kernel completion:
 *   - One thread launches kernels
 *   - Another thread waits for results
 *   - Condition variable signals completion
 *
 * - Pipeline pattern:
 *   - Stage 1: Data loading (CPU)
 *   - Stage 2: GPU processing
 *   - Stage 3: Result collection
 *   - Condition variables synchronize stages
 *
 * Note: For actual GPU synchronization, use CUDA events/streams,
 * but condition variables coordinate CPU threads managing GPU
 *
 * COMPILATION: g++ -std=c++11 -pthread 03_condition_variables.cpp -o condvar
 * ==================================================================================================
 */

int main() {
    cout << "=== Condition Variables Practice ===" << endl;

    cout << "\n1. Basic Condition Variable:" << endl;
    conditionVariableBasics();

    cout << "\n2. Producer-Consumer Pattern:" << endl;
    // Reset global state
    while (!dataQueue.empty()) dataQueue.pop();
    done = false;
    producerConsumerExample();

    cout << "\n3. Timed Wait:" << endl;
    timedWaitExample();

    return 0;
}
