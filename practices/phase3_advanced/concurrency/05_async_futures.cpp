/*
 * ==================================================================================================
 * Exercise: Async and Futures (Higher-Level Concurrency)
 * ==================================================================================================
 * Difficulty: Intermediate/Advanced | Time: 45-55 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master std::async for task-based parallelism
 * 2. Understand std::future and std::promise
 * 3. Learn std::packaged_task
 * 4. Practice exception propagation across threads
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Asynchronous GPU operations
 * - Parallel task execution for GPU workloads
 * - Result retrieval from async kernels
 * - Exception handling in GPU code
 * ==================================================================================================
 */

#include <iostream>
#include <future>
#include <thread>
#include <chrono>
#include <vector>
#include <numeric>
using namespace std;

/*
 * EXERCISE 1: std::async Basics (10 min)
 */

int computeSum(int a, int b) {
    this_thread::sleep_for(chrono::milliseconds(500));
    return a + b;
}

void asyncBasics() {
    cout << "Starting async computation..." << endl;

    // Launch async task
    future<int> result = async(launch::async, computeSum, 10, 20);

    cout << "Doing other work..." << endl;
    this_thread::sleep_for(chrono::milliseconds(200));

    // Get result (blocks if not ready)
    cout << "Result: " << result.get() << endl;
}

/*
 * EXERCISE 2: async Launch Policies (10 min)
 */

void launchPolicies() {
    // launch::async - Guaranteed new thread
    auto f1 = async(launch::async, []() {
        cout << "Async policy: " << this_thread::get_id() << endl;
        return 42;
    });

    // launch::deferred - Lazy evaluation (no new thread)
    auto f2 = async(launch::deferred, []() {
        cout << "Deferred policy: " << this_thread::get_id() << endl;
        return 100;
    });

    cout << "Main thread: " << this_thread::get_id() << endl;

    // f1 runs in separate thread
    cout << "f1 result: " << f1.get() << endl;

    // f2 runs in calling thread when get() is called
    cout << "f2 result: " << f2.get() << endl;
}

/*
 * EXERCISE 3: std::promise and std::future (15 min)
 */

void producer(promise<int> prom) {
    this_thread::sleep_for(chrono::seconds(1));
    cout << "Producer: Computing..." << endl;
    prom.set_value(42);  // Set the result
}

void promiseFutureExample() {
    promise<int> prom;
    future<int> fut = prom.get_future();

    // Start producer thread
    thread t(producer, move(prom));

    cout << "Waiting for result..." << endl;

    // Wait for result
    int result = fut.get();
    cout << "Got result: " << result << endl;

    t.join();
}

/*
 * EXERCISE 4: std::packaged_task (10 min)
 */

int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

void packagedTaskExample() {
    // Package function for async execution
    packaged_task<int(int)> task(factorial);

    // Get future before moving task
    future<int> result = task.get_future();

    // Run task in separate thread
    thread t(move(task), 5);

    cout << "Factorial(5) = " << result.get() << endl;

    t.join();
}

/*
 * EXERCISE 5: Exception Propagation (10 min)
 */

int divide(int a, int b) {
    if (b == 0) {
        throw runtime_error("Division by zero!");
    }
    return a / b;
}

void exceptionPropagation() {
    // Exception is stored in future
    future<int> result = async(launch::async, divide, 10, 0);

    try {
        int value = result.get();  // Exception thrown here
        cout << "Result: " << value << endl;
    } catch (const exception& e) {
        cout << "Caught exception: " << e.what() << endl;
    }
}

/*
 * EXERCISE 6: Parallel Reduction (10 min)
 */

int parallelSum(const vector<int>& data) {
    const int CHUNK_SIZE = data.size() / 4;
    vector<future<int>> futures;

    // Launch 4 async tasks
    for (int i = 0; i < 4; i++) {
        auto begin = data.begin() + i * CHUNK_SIZE;
        auto end = (i == 3) ? data.end() : begin + CHUNK_SIZE;

        futures.push_back(async(launch::async, [begin, end]() {
            return accumulate(begin, end, 0);
        }));
    }

    // Collect results
    int total = 0;
    for (auto& f : futures) {
        total += f.get();
    }

    return total;
}

void parallelReductionExample() {
    vector<int> data(1000);
    for (int i = 0; i < data.size(); i++) {
        data[i] = i + 1;
    }

    int result = parallelSum(data);
    cout << "Parallel sum: " << result << endl;
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is std::async?
 * A: Higher-level abstraction for launching async tasks
 *    Returns std::future for result retrieval
 *
 * Q2: async vs thread?
 * A: async: Task-based, returns future, automatic thread management
 *    thread: Lower-level, manual thread management, no return value
 *
 * Q3: What are launch policies?
 * A: launch::async - Runs in new thread immediately
 *    launch::deferred - Lazy evaluation, runs when get() called
 *    launch::async | launch::deferred - Implementation decides
 *
 * Q4: What is std::future?
 * A: Object that represents future result of async operation
 *    get() blocks until result is ready
 *
 * Q5: What is std::promise?
 * A: Object that sets a value/exception for associated future
 *    Allows manual control of when result is ready
 *
 * Q6: promise vs async?
 * A: promise: Manual control, set value from anywhere
 *    async: Automatic, function return becomes future value
 *
 * Q7: What is std::packaged_task?
 * A: Wraps callable object to enable async execution
 *    Combines function and future management
 *
 * Q8: Can future.get() be called multiple times?
 * A: No, get() moves the result out
 *    Use shared_future for multiple get() calls
 *
 * Q9: How are exceptions handled?
 * A: Exception in async task is stored in future
 *    Thrown when get() is called
 *    Allows exception propagation across threads
 *
 * Q10: When is future ready?
 * A: Check with future.wait() or future.wait_for()
 *    Don't block unnecessarily
 *
 * Q11: Performance considerations?
 * A: - Thread creation overhead
 *    - Consider thread pool for many small tasks
 *    - Deferred execution avoids unnecessary threads
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Async kernel launches:
 *   auto result = async(launch::async, [&]() {
 *       // Launch CUDA kernel
 *       kernel<<<grid, block, 0, stream>>>(d_data);
 *       cudaStreamSynchronize(stream);
 *       // Copy result back
 *       return processResult();
 *   });
 *
 * - Parallel GPU workload distribution:
 *   vector<future<GPUResult>> results;
 *   for (int gpu = 0; gpu < numGPUs; gpu++) {
 *       results.push_back(async(launch::async, [gpu]() {
 *           cudaSetDevice(gpu);
 *           // Launch kernel on this GPU
 *           return computeOnGPU();
 *       }));
 *   }
 *   for (auto& f : results) {
 *       auto result = f.get();
 *   }
 *
 * - Exception handling for CUDA errors:
 *   future<float*> gpuCompute = async(launch::async, []() {
 *       float* d_result;
 *       cudaError_t err = cudaMalloc(&d_result, size);
 *       if (err != cudaSuccess) {
 *           throw runtime_error(cudaGetErrorString(err));
 *       }
 *       return d_result;
 *   });
 *
 * - Pipeline pattern:
 *   auto stage1 = async(launch::async, loadData);
 *   auto stage2 = async(launch::async, [&]() {
 *       auto data = stage1.get();
 *       return processOnGPU(data);
 *   });
 *
 * COMPILATION: g++ -std=c++11 -pthread 05_async_futures.cpp -o async
 * ==================================================================================================
 */

int main() {
    cout << "=== Async and Futures Practice ===" << endl;

    cout << "\n1. Async Basics:" << endl;
    asyncBasics();

    cout << "\n2. Launch Policies:" << endl;
    launchPolicies();

    cout << "\n3. Promise and Future:" << endl;
    promiseFutureExample();

    cout << "\n4. Packaged Task:" << endl;
    packagedTaskExample();

    cout << "\n5. Exception Propagation:" << endl;
    exceptionPropagation();

    cout << "\n6. Parallel Reduction:" << endl;
    parallelReductionExample();

    return 0;
}
