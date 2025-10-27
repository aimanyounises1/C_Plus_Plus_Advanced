/*
 * ==================================================================================================
 * Exercise: Strategy Pattern
 * ==================================================================================================
 * Difficulty: Intermediate/Advanced | Time: 40-50 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master Strategy pattern
 * 2. Understand algorithm encapsulation
 * 3. Learn runtime algorithm selection
 * 4. Practice composition over inheritance
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Algorithm selection based on data size
 * - Different sorting strategies for GPU
 * - Memory access patterns (coalesced vs non-coalesced)
 * - Kernel selection based on architecture
 * ==================================================================================================
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <string>
using namespace std;

/*
 * EXERCISE 1: Basic Strategy Pattern (15 min)
 */

// Strategy interface
class SortStrategy {
public:
    virtual void sort(vector<int>& data) = 0;
    virtual string getName() const = 0;
    virtual ~SortStrategy() = default;
};

// Concrete Strategies
class BubbleSort : public SortStrategy {
public:
    void sort(vector<int>& data) override {
        int n = data.size();
        for (int i = 0; i < n-1; i++) {
            for (int j = 0; j < n-i-1; j++) {
                if (data[j] > data[j+1]) {
                    swap(data[j], data[j+1]);
                }
            }
        }
    }
    string getName() const override { return "Bubble Sort"; }
};

class QuickSort : public SortStrategy {
public:
    void sort(vector<int>& data) override {
        quickSort(data, 0, data.size() - 1);
    }
    string getName() const override { return "Quick Sort"; }

private:
    void quickSort(vector<int>& arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }

    int partition(vector<int>& arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                swap(arr[i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[high]);
        return i + 1;
    }
};

class STLSort : public SortStrategy {
public:
    void sort(vector<int>& data) override {
        std::sort(data.begin(), data.end());
    }
    string getName() const override { return "STL Sort"; }
};

// Context
class Sorter {
private:
    unique_ptr<SortStrategy> strategy;

public:
    void setStrategy(unique_ptr<SortStrategy> strat) {
        strategy = move(strat);
    }

    void sort(vector<int>& data) {
        if (strategy) {
            cout << "Using " << strategy->getName() << endl;
            strategy->sort(data);
        }
    }
};

/*
 * EXERCISE 2: Strategy with Function Pointers (10 min)
 */

class PaymentStrategy {
public:
    virtual void pay(int amount) = 0;
    virtual ~PaymentStrategy() = default;
};

class CreditCardPayment : public PaymentStrategy {
private:
    string cardNumber;
public:
    CreditCardPayment(const string& card) : cardNumber(card) {}
    void pay(int amount) override {
        cout << "Paid $" << amount << " using Credit Card: " << cardNumber << endl;
    }
};

class PayPalPayment : public PaymentStrategy {
private:
    string email;
public:
    PayPalPayment(const string& e) : email(e) {}
    void pay(int amount) override {
        cout << "Paid $" << amount << " using PayPal: " << email << endl;
    }
};

class BitcoinPayment : public PaymentStrategy {
private:
    string walletAddress;
public:
    BitcoinPayment(const string& wallet) : walletAddress(wallet) {}
    void pay(int amount) override {
        cout << "Paid $" << amount << " using Bitcoin: " << walletAddress << endl;
    }
};

class ShoppingCart {
private:
    unique_ptr<PaymentStrategy> paymentStrategy;
    int totalAmount;

public:
    ShoppingCart() : totalAmount(0) {}

    void setPaymentStrategy(unique_ptr<PaymentStrategy> strategy) {
        paymentStrategy = move(strategy);
    }

    void addItem(int price) {
        totalAmount += price;
    }

    void checkout() {
        if (paymentStrategy) {
            paymentStrategy->pay(totalAmount);
        } else {
            cout << "No payment method selected" << endl;
        }
    }
};

/*
 * EXERCISE 3: Strategy with Lambdas (Modern C++) (15 min)
 */

#include <functional>

class Calculator {
private:
    function<int(int, int)> operation;

public:
    void setOperation(function<int(int, int)> op) {
        operation = op;
    }

    int execute(int a, int b) {
        return operation ? operation(a, b) : 0;
    }
};

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is the Strategy pattern?
 * A: Defines family of algorithms, encapsulates each one, makes them
 *    interchangeable. Lets algorithm vary independently from clients.
 *
 * Q2: Components of Strategy pattern?
 * A: - Strategy: Interface for all supported algorithms
 *    - ConcreteStrategy: Implements algorithm using Strategy interface
 *    - Context: Maintains reference to Strategy object
 *
 * Q3: When to use Strategy pattern?
 * A: - Many related classes differ only in behavior
 *    - Need different variants of an algorithm
 *    - Algorithm uses data client shouldn't know about
 *    - Class defines many behaviors as conditional statements
 *
 * Q4: Strategy vs State pattern?
 * A: Strategy: Client chooses algorithm explicitly
 *    State: Context changes behavior based on internal state automatically
 *
 * Q5: Advantages of Strategy?
 * A: - Eliminates conditional statements
 *    - Encapsulates algorithm variations
 *    - Runtime algorithm selection
 *    - Open/Closed Principle (add new strategies without modifying context)
 *
 * Q6: Disadvantages?
 * A: - Increased number of objects
 *    - Clients must know different strategies
 *    - Communication overhead between Strategy and Context
 *
 * Q7: Strategy vs Template Method?
 * A: Strategy: Composition, runtime flexibility, full algorithm replacement
 *    Template Method: Inheritance, compile-time, partial algorithm override
 *
 * Q8: How to avoid exposing implementation?
 * A: - Strategy and Context share interface to pass data
 *    - Context passes itself as argument to Strategy
 *    - Use callbacks or visitors
 *
 * Q9: Modern C++ alternatives?
 * A: - std::function with lambdas (simpler for simple strategies)
 *    - Template specialization (compile-time)
 *    - Function pointers (C-style)
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Kernel selection: Different kernels for small vs large data
 * - Memory strategy: Choose coalesced/non-coalesced access pattern
 * - Reduction strategy: Different reduction algorithms based on data size
 * - Precision strategy: float vs double vs half precision
 *
 * Example use case:
 * class ReductionStrategy {
 * public:
 *     virtual void reduce(float* d_data, int n, float* result) = 0;
 * };
 *
 * class SequentialReduction : public ReductionStrategy {
 *     void reduce(float* d_data, int n, float* result) override {
 *         // Sequential reduction for small data
 *     }
 * };
 *
 * class ParallelReduction : public ReductionStrategy {
 *     void reduce(float* d_data, int n, float* result) override {
 *         // Parallel reduction for large data
 *     }
 * };
 *
 * class CudaReducer {
 *     unique_ptr<ReductionStrategy> strategy;
 * public:
 *     void setStrategy(unique_ptr<ReductionStrategy> s) {
 *         strategy = move(s);
 *     }
 *     void reduce(float* d_data, int n, float* result) {
 *         if (n < 1000) {
 *             setStrategy(make_unique<SequentialReduction>());
 *         } else {
 *             setStrategy(make_unique<ParallelReduction>());
 *         }
 *         strategy->reduce(d_data, n, result);
 *     }
 * };
 *
 * COMPILATION: g++ -std=c++14 04_strategy.cpp -o strategy
 * ==================================================================================================
 */

int main() {
    cout << "=== Strategy Pattern Practice ===" << endl;

    // Sorting Strategy
    cout << "\n1. Sorting Strategy:" << endl;
    vector<int> data1 = {64, 34, 25, 12, 22, 11, 90};
    vector<int> data2 = data1;
    vector<int> data3 = data1;

    Sorter sorter;

    sorter.setStrategy(make_unique<BubbleSort>());
    sorter.sort(data1);
    cout << "Result: ";
    for (int n : data1) cout << n << " ";
    cout << endl;

    sorter.setStrategy(make_unique<QuickSort>());
    sorter.sort(data2);
    cout << "Result: ";
    for (int n : data2) cout << n << " ";
    cout << endl;

    sorter.setStrategy(make_unique<STLSort>());
    sorter.sort(data3);
    cout << "Result: ";
    for (int n : data3) cout << n << " ";
    cout << endl;

    // Payment Strategy
    cout << "\n2. Payment Strategy:" << endl;
    ShoppingCart cart;
    cart.addItem(100);
    cart.addItem(50);

    cart.setPaymentStrategy(make_unique<CreditCardPayment>("1234-5678-9012-3456"));
    cart.checkout();

    ShoppingCart cart2;
    cart2.addItem(75);
    cart2.setPaymentStrategy(make_unique<PayPalPayment>("user@example.com"));
    cart2.checkout();

    // Lambda Strategy
    cout << "\n3. Strategy with Lambdas:" << endl;
    Calculator calc;

    calc.setOperation([](int a, int b) { return a + b; });
    cout << "5 + 3 = " << calc.execute(5, 3) << endl;

    calc.setOperation([](int a, int b) { return a * b; });
    cout << "5 * 3 = " << calc.execute(5, 3) << endl;

    calc.setOperation([](int a, int b) { return a - b; });
    cout << "5 - 3 = " << calc.execute(5, 3) << endl;

    return 0;
}
