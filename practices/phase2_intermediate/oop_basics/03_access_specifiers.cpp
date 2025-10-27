/*
 * ==================================================================================================
 * Exercise: Access Specifiers in C++ (public, private, protected)
 * ==================================================================================================
 * Difficulty: Beginner/Intermediate | Time: 35-45 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master public, private, and protected access specifiers
 * 2. Understand encapsulation and data hiding
 * 3. Learn proper use of getters and setters
 * 4. Recognize when to use each access level
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Encapsulation fundamental to API design
 * - Protecting internal state prevents bugs
 * - Public interfaces vs implementation details
 * - CUDA APIs use encapsulation extensively
 * ==================================================================================================
 */

#include <iostream>
#include <string>

using namespace std;

/*
 * ==================================================================================================
 * THEORY: Access Specifiers
 * ==================================================================================================
 *
 * PUBLIC: Accessible from anywhere
 * - Interface of the class
 * - Methods that users should call
 * - Usually: member functions, not data
 *
 * PRIVATE: Accessible only within the class
 * - Implementation details
 * - Data members (encapsulation!)
 * - Helper functions
 * - Default for class
 *
 * PROTECTED: Accessible in class and derived classes
 * - Used in inheritance hierarchies
 * - Data/methods for derived classes only
 *
 * ==================================================================================================
 */

/*
 * EXERCISE 1: Basic Encapsulation (10 min)
 */

// TODO 1.1: Create a "BankAccount" class with proper encapsulation
class BankAccount {
private:
    string accountNumber;
    double balance;

public:
    // Constructor
    BankAccount(string accNum, double initialBalance)
        : accountNumber(accNum), balance(initialBalance) {}

    // Getters
    string getAccountNumber() const { return accountNumber; }
    double getBalance() const { return balance; }

    // Methods with validation
    bool deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            return true;
        }
        return false;
    }

    bool withdraw(double amount) {
        if (amount > 0 && balance >= amount) {
            balance -= amount;
            return true;
        }
        return false;
    }
};

// TODO 1.2: Try to access private members directly - observe error!

/*
 * EXERCISE 2: Proper Getters/Setters (10 min)
 */

// TODO 2.1: Create a "Person" class with validation
class Person {
private:
    string name;
    int age;

public:
    // Getters
    string getName() const { return name; }
    int getAge() const { return age; }

    // Setters with validation
    void setName(const string& n) {
        if (!n.empty()) name = n;
    }

    void setAge(int a) {
        if (a >= 0 && a <= 150) age = a;
    }
};

/*
 * EXERCISE 3: Protected Access (15 min)
 */

// TODO 3.1: Create base class with protected members
class Shape {
protected:
    double area;

public:
    virtual double calculateArea() = 0;
    double getArea() const { return area; }
};

class Circle : public Shape {
private:
    double radius;

public:
    Circle(double r) : radius(r) {}

    double calculateArea() override {
        area = 3.14159 * radius * radius;  // Can access protected 'area'
        return area;
    }
};

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: Why use private members?
 * A: Encapsulation - hide implementation, control access, maintain invariants
 *
 * Q2: When to use protected?
 * A: When derived classes need access but external code shouldn't
 *
 * Q3: Difference between struct and class default access?
 * A: struct is public by default, class is private by default
 *
 * Q4: Can you change access of inherited members?
 * A: Yes - public/protected/private inheritance changes access
 *
 * Q5: What are friend functions?
 * A: Functions that can access private/protected members despite access specifiers
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - API design: Public interfaces hide GPU implementation details
 * - Resource handles: Private members prevent misuse
 * - Stream/context objects: Encapsulate GPU state
 *
 * COMPILATION: g++ -std=c++17 03_access_specifiers.cpp -o access
 *
 * LEARNING CHECKLIST:
 * ☐ Understand public/private/protected
 * ☐ Implement proper encapsulation
 * ☐ Write getters/setters with validation
 * ☐ Use protected for inheritance
 * ==================================================================================================
 */

int main() {
    cout << "=== Access Specifiers Practice ===" << endl;

    BankAccount acc("12345", 1000.0);
    acc.deposit(500);
    cout << "Balance: " << acc.getBalance() << endl;

    return 0;
}
