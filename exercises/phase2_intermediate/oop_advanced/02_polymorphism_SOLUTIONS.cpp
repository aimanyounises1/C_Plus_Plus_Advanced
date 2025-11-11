/*
 * ==================================================================================================
 * POLYMORPHISM EXERCISES - COMPLETE SOLUTIONS
 * ==================================================================================================
 * This file contains complete, working solutions to all polymorphism exercises.
 * Use this for reference ONLY after attempting the exercises yourself!
 *
 * Learning is most effective when you struggle with problems first.
 * ==================================================================================================
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>

using namespace std;

/*
 * ==================================================================================================
 * EXERCISE 1: Banking System - SOLUTIONS
 * ==================================================================================================
 */

class Account {
protected:
    string accountHolder;
    double balance;
    string accountNumber;

public:
    Account(const string& holder, const string& accNum, double initialBalance)
        : accountHolder(holder), accountNumber(accNum), balance(initialBalance) {}

    virtual double calculateInterest() const {
        return balance * 0.02;  // 2% interest
    }

    void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
        }
    }

    bool withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }

    double getBalance() const {
        return balance;
    }

    string getAccountNumber() const { return accountNumber; }
    string getAccountHolder() const { return accountHolder; }

    virtual ~Account() {}
};

class SavingsAccount : public Account {
public:
    SavingsAccount(const string& holder, const string& accNum, double initialBalance)
        : Account(holder, accNum, initialBalance) {}

    double calculateInterest() const override {
        return balance * 0.035;  // 3.5% interest
    }
};

class CheckingAccount : public Account {
private:
    double overdraftLimit;

public:
    CheckingAccount(const string& holder, const string& accNum,
                   double initialBalance, double overdraft = 500.0)
        : Account(holder, accNum, initialBalance), overdraftLimit(overdraft) {}

    double calculateInterest() const override {
        return balance * 0.01;  // 1% interest
    }

    bool withdraw(double amount) override {
        if (amount > 0 && (balance - amount) >= -overdraftLimit) {
            balance -= amount;
            return true;
        }
        return false;
    }
};

/*
 * ==================================================================================================
 * EXERCISE 2: Shape Calculator - SOLUTIONS
 * ==================================================================================================
 */

class Shape {
public:
    virtual double getArea() const = 0;  // Pure virtual
    virtual double getPerimeter() const = 0;  // Pure virtual
    virtual string getType() const = 0;  // Pure virtual
    virtual ~Shape() {}
};

class Circle : public Shape {
private:
    double radius;
    static constexpr double PI = 3.14159265359;

public:
    Circle(double r) : radius(r) {}

    double getArea() const override {
        return PI * radius * radius;
    }

    double getPerimeter() const override {
        return 2 * PI * radius;
    }

    string getType() const override { return "Circle"; }
};

class Rectangle : public Shape {
private:
    double width;
    double height;

public:
    Rectangle(double w, double h) : width(w), height(h) {}

    double getArea() const override {
        return width * height;
    }

    double getPerimeter() const override {
        return 2 * (width + height);
    }

    string getType() const override { return "Rectangle"; }
};

class Triangle : public Shape {
private:
    double side1, side2, side3;

public:
    Triangle(double s1, double s2, double s3)
        : side1(s1), side2(s2), side3(s3) {}

    double getArea() const override {
        // Heron's formula: s = (a+b+c)/2, area = sqrt(s*(s-a)*(s-b)*(s-c))
        double s = (side1 + side2 + side3) / 2.0;
        return sqrt(s * (s - side1) * (s - side2) * (s - side3));
    }

    double getPerimeter() const override {
        return side1 + side2 + side3;
    }

    string getType() const override { return "Triangle"; }
};

/*
 * ==================================================================================================
 * EXERCISE 3: Employee Management System - SOLUTIONS
 * ==================================================================================================
 */

class Employee {
protected:
    string name;
    int employeeId;
    double baseSalary;

public:
    Employee(const string& n, int id, double salary)
        : name(n), employeeId(id), baseSalary(salary) {}

    virtual double calculatePay() const {
        return baseSalary;
    }

    virtual string getDetails() const {
        return "Employee: " + name + " (ID: " + to_string(employeeId) + ")";
    }

    string getName() const { return name; }
    int getId() const { return employeeId; }

    virtual ~Employee() {}
};

class Manager : public Employee {
private:
    int teamSize;

public:
    Manager(const string& n, int id, double salary, int team)
        : Employee(n, id, salary), teamSize(team) {}

    double calculatePay() const override {
        return baseSalary * 1.2;  // 20% bonus
    }

    string getDetails() const override {
        return "Manager: " + name + " (ID: " + to_string(employeeId) +
               ", Team: " + to_string(teamSize) + ")";
    }
};

class Engineer : public Employee {
private:
    int projectsCompleted;

public:
    Engineer(const string& n, int id, double salary, int projects)
        : Employee(n, id, salary), projectsCompleted(projects) {}

    double calculatePay() const override {
        return baseSalary + (projectsCompleted * 50.0);  // $50 per project
    }

    string getDetails() const override {
        return "Engineer: " + name + " (ID: " + to_string(employeeId) +
               ", Projects: " + to_string(projectsCompleted) + ")";
    }
};

class Intern : public Employee {
private:
    int monthsRemaining;

public:
    Intern(const string& n, int id, double salary, int months)
        : Employee(n, id, salary), monthsRemaining(months) {}

    double calculatePay() const override {
        return baseSalary * 0.6;  // 60% of base
    }

    string getDetails() const override {
        return "Intern: " + name + " (ID: " + to_string(employeeId) +
               ", Months: " + to_string(monthsRemaining) + ")";
    }
};

/*
 * ==================================================================================================
 * EXERCISE 4: Vehicle Fleet Management - SOLUTIONS
 * ==================================================================================================
 */

class Vehicle {
protected:
    string make;
    string model;
    int year;
    double fuelLevel;  // percentage (0-100)

public:
    Vehicle(const string& mk, const string& mdl, int yr)
        : make(mk), model(mdl), year(yr), fuelLevel(100.0) {}

    virtual double getFuelEfficiency() const {
        return 25.0;  // Base: 25 mpg
    }

    virtual double getMaintenanceCost() const {
        return 500.0;  // Base: $500
    }

    void refuel() {
        fuelLevel = 100.0;
    }

    void drive(double miles) {
        double efficiency = getFuelEfficiency();
        double fuelConsumed = (miles / efficiency);  // gallons consumed
        fuelLevel -= fuelConsumed;  // 1 gallon = 1% for simplicity
        if (fuelLevel < 0) fuelLevel = 0;
    }

    double getFuelLevel() const { return fuelLevel; }
    string getMake() const { return make; }
    string getModel() const { return model; }

    virtual ~Vehicle() {}
};

class Car : public Vehicle {
private:
    int numberOfDoors;

public:
    Car(const string& mk, const string& mdl, int yr, int doors)
        : Vehicle(mk, mdl, yr), numberOfDoors(doors) {}

    double getFuelEfficiency() const override {
        return 35.0;  // 35 mpg
    }

    double getMaintenanceCost() const override {
        return 400.0;  // $400
    }
};

class Truck : public Vehicle {
private:
    double cargoCapacity;

public:
    Truck(const string& mk, const string& mdl, int yr, double capacity)
        : Vehicle(mk, mdl, yr), cargoCapacity(capacity) {}

    double getFuelEfficiency() const override {
        return 18.0;  // 18 mpg
    }

    double getMaintenanceCost() const override {
        return 800.0;  // $800
    }
};

class Motorcycle : public Vehicle {
private:
    string type;

public:
    Motorcycle(const string& mk, const string& mdl, int yr, const string& t)
        : Vehicle(mk, mdl, yr), type(t) {}

    double getFuelEfficiency() const override {
        return 55.0;  // 55 mpg
    }

    double getMaintenanceCost() const override {
        return 200.0;  // $200
    }
};

/*
 * ==================================================================================================
 * BONUS: Function Overloading - SOLUTIONS
 * ==================================================================================================
 */

class MathOperations {
public:
    int multiply(int a, int b) {
        return a * b;
    }

    int multiply(int a, int b, int c) {
        return a * b * c;
    }

    double multiply(double a, double b) {
        return a * b;
    }

    vector<int> multiply(const vector<int>& vec, int scalar) {
        vector<int> result;
        result.reserve(vec.size());
        for (int val : vec) {
            result.push_back(val * scalar);
        }
        return result;
    }
};

/*
 * ==================================================================================================
 * KEY LEARNING POINTS FROM THESE SOLUTIONS:
 * ==================================================================================================
 *
 * 1. VIRTUAL FUNCTIONS:
 *    - Use 'virtual' in base class
 *    - Use 'override' in derived classes for safety
 *    - Always have virtual destructor in polymorphic base classes
 *
 * 2. PURE VIRTUAL (ABSTRACT):
 *    - Syntax: virtual func() = 0;
 *    - Makes class abstract (cannot instantiate)
 *    - Forces derived classes to implement
 *
 * 3. CONST CORRECTNESS:
 *    - Getter functions should be const
 *    - Calculation functions that don't modify state should be const
 *    - Important for working with const objects and references
 *
 * 4. PROTECTED MEMBERS:
 *    - Accessible in derived classes
 *    - Still hidden from outside world
 *    - Balance between encapsulation and inheritance needs
 *
 * 5. CONSTRUCTOR INITIALIZATION LISTS:
 *    - More efficient than assignment in constructor body
 *    - Required for const members and references
 *    - Calls base class constructor first
 *
 * 6. FUNCTION OVERLOADING:
 *    - Same function name, different parameters
 *    - Compile-time polymorphism
 *    - Compiler selects based on arguments
 *
 * ==================================================================================================
 */