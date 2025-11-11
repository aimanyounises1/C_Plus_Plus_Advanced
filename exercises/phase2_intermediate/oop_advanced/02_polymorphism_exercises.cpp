/*
 * ==================================================================================================
 * POLYMORPHISM EXERCISES - Hands-On Coding Challenges
 * ==================================================================================================
 * Complete the following exercises by implementing the required functionality.
 * DO NOT modify function signatures or class names - tests depend on them!
 *
 * After completing, run: ./test_polymorphism to get your score
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
 * EXERCISE 1: Banking System (25 points)
 * ==================================================================================================
 * Create a polymorphic banking system with different account types.
 * Each account type calculates interest differently.
 */

// TODO: Implement the base Account class
class Account {
protected:
    string accountHolder;
    double balance;
    string accountNumber;

public:
    // TODO: Implement constructor
    Account(const string &holder, const string &accNum, double initialBalance) : accountHolder(holder),
        balance(initialBalance), accountNumber(accNum) {
    }

    // TODO: Make this virtual and implement
    virtual double calculateInterest() const {
        // YOUR CODE HERE (base rate: 2%)
        return balance * 0.02;
    }

    // TODO: Implement deposit
    void deposit(double amount) {
        balance += amount;
    }

    // TODO: Implement withdraw (return false if insufficient funds)
    virtual bool withdraw(double amount) {
        // YOUR CODE HERE
        if (amount > balance) return false;
        balance -= amount;
        return balance >= 0;
    }

    // TODO: Make getBalance const and implement
    double getBalance() const {
        // YOUR CODE HERE
        return balance;
    }

    string getAccountNumber() const { return accountNumber; }
    string getAccountHolder() const { return accountHolder; }

    // TODO: Implement virtual destructor
    virtual ~Account() {
    }
};

// TODO: Implement SavingsAccount (interest rate: 3.5%)
class SavingsAccount : public Account {
public:
    SavingsAccount(const string &holder, const string &accNum, double initialBalance)
        : Account(holder, accNum, initialBalance) {
    }

    // TODO: Override calculateInterest
    double calculateInterest() const override {
        // YOUR CODE HERE (3.5% interest rate)
        return balance * 0.035;
    }

    bool withdraw(double amount) override {
        balance -= amount;
        return balance >= 0;
    }

    virtual ~SavingsAccount() {
    }
};

// TODO: Implement CheckingAccount (interest rate: 1%, has overdraft protection)
class CheckingAccount : public Account {
private:
    double overdraftLimit;

public:
    CheckingAccount(const string &holder, const string &accNum,
                    double initialBalance, double overdraft = 500.0)
        : Account(holder, accNum, initialBalance), overdraftLimit(overdraft) {
    }

    // TODO: Override calculateInterest
    double calculateInterest() const override {
        // YOUR CODE HERE (1% interest rate)
        return balance * 0.01;
    }

    // TODO: Override withdraw to allow overdraft
    bool withdraw(double amount) override {
        balance -= amount;
        return balance >= 0;
    }

    virtual ~CheckingAccount() {
    }
};

/*
 * ==================================================================================================
 * EXERCISE 2: Shape Calculator (25 points)
 * ==================================================================================================
 * Implement different shapes with area and perimeter calculations.
 */

// TODO: Implement abstract Shape base class
class Shape {
public:
    // TODO: Make these pure virtual functions
    virtual double getArea() const {
        return 0.0;
    }

    virtual double getPerimeter() const {
        return 0.0;
    }

    virtual string getType() const {
        return "Unknown";
    }

    virtual ~Shape() {
    }
};

// TODO: Implement Circle
class Circle : public Shape {
private:
    double radius;

public:
    Circle(double r) : radius(r) {
    }

    // TODO: Implement area = π * r²
    double getArea() const override {
        // YOUR CODE HERE
        return M_PI * pow(radius, 2);
    }

    // TODO: Implement perimeter = 2 * π * r
    double getPerimeter() const override {
        // YOUR CODE HERE
        return 2 * M_PI * radius;
    }

    string getType() const override { return "Circle"; }
};

// TODO: Implement Rectangle
class Rectangle : public Shape {
private:
    double width;
    double height;

public:
    Rectangle(double w, double h) : width(w), height(h) {
    }

    // TODO: Implement area = width * height
    double getArea() const override {
        // YOUR CODE HERE
        return width * height;
    }

    // TODO: Implement perimeter = 2 * (width + height)
    double getPerimeter() const override {
        // YOUR CODE HERE
        return 2 * (width + height);
    }

    string getType() const override { return "Rectangle"; }
};

// TODO: Implement Triangle
class Triangle : public Shape {
private:
    double side1, side2, side3;

public:
    Triangle(double s1, double s2, double s3)
        : side1(s1), side2(s2), side3(s3) {
    }

    // TODO: Implement area using Heron's formula
    // s = (a+b+c)/2, area = sqrt(s*(s-a)*(s-b)*(s-c))
    double getArea() const override {
        // YOUR CODE HERE
        double s = (side1 + side2 + side3) / 2;
        return sqrt(s * (s - side1) * (s - side2) * (s - side3));
    }

    // TODO: Implement perimeter = side1 + side2 + side3
    double getPerimeter() const override {
        // YOUR CODE HERE
        return side1 + side2 + side3;
    }

    string getType() const override { return "Triangle"; }
};

/*
 * ==================================================================================================
 * EXERCISE 3: Employee Management System (25 points)
 * ==================================================================================================
 * Create a polymorphic employee hierarchy with different salary calculations.
 */

// TODO: Implement Employee base class
class Employee {
protected:
    string name;
    int employeeId;
    double baseSalary;

public:
    Employee(const string &n, int id, double salary)
        : name(n), employeeId(id), baseSalary(salary) {
    }

    // TODO: Implement virtual calculatePay
    virtual double calculatePay() const {
        // YOUR CODE HERE (return baseSalary)
        return baseSalary;
    }

    // TODO: Implement virtual getDetails
    virtual string getDetails() const {
        // YOUR CODE HERE (return name and ID)
        return getName() + " " + to_string(employeeId);
    }

    string getName() const { return name; }
    int getId() const { return employeeId; }

    virtual ~Employee() {
    }
};

// TODO: Implement Manager (gets 20% bonus)
class Manager : public Employee {
private:
    int teamSize;

public:
    Manager(const string &n, int id, double salary, int team)
        : Employee(n, id, salary), teamSize(team) {
    }

    // TODO: Override calculatePay (baseSalary * 1.2)
    double calculatePay() const override {
        // YOUR CODE HERE
        return baseSalary * 1.2;
    }

    string getDetails() const override {
        // YOUR CODE HERE (include team size)
        return "";
    }
};

// TODO: Implement Engineer (gets $50 per project)
class Engineer : public Employee {
private:
    int projectsCompleted;

public:
    Engineer(const string &n, int id, double salary, int projects)
        : Employee(n, id, salary), projectsCompleted(projects) {
    }

    // TODO: Override calculatePay (baseSalary + projects * 50)
    double calculatePay() const override {
        // YOUR CODE HERE
        return baseSalary + projectsCompleted * 50;
    }

    string getDetails() const override {
        // YOUR CODE HERE (include projects completed)
        return "";
    }
};

// TODO: Implement Intern (gets 60% of base salary)
class Intern : public Employee {
private:
    int monthsRemaining;

public:
    Intern(const string &n, int id, double salary, int months)
        : Employee(n, id, salary), monthsRemaining(months) {
    }

    // TODO: Override calculatePay (baseSalary * 0.6)
    double calculatePay() const override {
        // YOUR CODE HERE
        return baseSalary * 0.6;
    }

    string getDetails() const override {
        // YOUR CODE HERE (include months remaining)
        return getName() + " " + to_string(employeeId) + to_string(monthsRemaining);
    }
};

/*
 * ==================================================================================================
 * EXERCISE 4: Vehicle Fleet Management (25 points)
 * ==================================================================================================
 * Implement a vehicle fleet with different vehicle types and fuel consumption calculations.
 */

// TODO: Implement Vehicle base class
class Vehicle {
protected:
    string make;
    string model;
    int year;
    double fuelLevel; // percentage (0-100)

public:
    Vehicle(const string &mk, const string &mdl, int yr)
        : make(mk), model(mdl), year(yr), fuelLevel(100.0) {
    }

    // TODO: Implement virtual getFuelEfficiency (base: 25 mpg)
    virtual double getFuelEfficiency() const {
        // YOUR CODE HERE
        return 0.0;
    }

    // TODO: Implement virtual getMaintenanceCost (base: $500)
    virtual double getMaintenanceCost() const {
        // YOUR CODE HERE
        return 0.0;
    }

    // TODO: Implement refuel
    void refuel() {
        // YOUR CODE HERE (set fuelLevel to 100)
    }

    // TODO: Implement drive (reduces fuel based on distance and efficiency)
    void drive(double miles) {
        // YOUR CODE HERE
        // Calculate fuel consumed based on miles and efficiency
        // 1 gallon = 1% fuel for simplicity
    }

    double getFuelLevel() const { return fuelLevel; }
    string getMake() const { return make; }
    string getModel() const { return model; }

    virtual ~Vehicle() {
    }
};

// TODO: Implement Car (35 mpg, $400 maintenance)
class Car : public Vehicle {
private:
    int numberOfDoors;

public:
    Car(const string &mk, const string &mdl, int yr, int doors)
        : Vehicle(mk, mdl, yr), numberOfDoors(doors) {
    }

    // TODO: Override getFuelEfficiency
    double getFuelEfficiency() const override {
        // YOUR CODE HERE (35 mpg)
        return 0.0;
    }

    // TODO: Override getMaintenanceCost
    double getMaintenanceCost() const override {
        // YOUR CODE HERE ($400)
        return 0.0;
    }
};

// TODO: Implement Truck (18 mpg, $800 maintenance)
class Truck : public Vehicle {
private:
    double cargoCapacity;

public:
    Truck(const string &mk, const string &mdl, int yr, double capacity)
        : Vehicle(mk, mdl, yr), cargoCapacity(capacity) {
    }

    // TODO: Override getFuelEfficiency
    double getFuelEfficiency() const override {
        // YOUR CODE HERE (18 mpg)
        return 0.0;
    }

    // TODO: Override getMaintenanceCost
    double getMaintenanceCost() const override {
        // YOUR CODE HERE ($800)
        return 0.0;
    }
};

// TODO: Implement Motorcycle (55 mpg, $200 maintenance)
class Motorcycle : public Vehicle {
private:
    string type; // "sport", "cruiser", etc.

public:
    Motorcycle(const string &mk, const string &mdl, int yr, const string &t)
        : Vehicle(mk, mdl, yr), type(t) {
    }

    // TODO: Override getFuelEfficiency
    double getFuelEfficiency() const override {
        // YOUR CODE HERE (55 mpg)
        return 0.0;
    }

    // TODO: Override getMaintenanceCost
    double getMaintenanceCost() const override {
        // YOUR CODE HERE ($200)
        return 0.0;
    }
};

/*
 * ==================================================================================================
 * BONUS EXERCISE: Function Overloading (Compile-time Polymorphism) (10 bonus points)
 * ==================================================================================================
 */

// TODO: Implement MathOperations class with function overloading
class MathOperations {
public:
    // TODO: Implement multiply for two integers
    int multiply(int a, int b) {
        // YOUR CODE HERE
        return 0;
    }

    // TODO: Implement multiply for three integers
    int multiply(int a, int b, int c) {
        // YOUR CODE HERE
        return 0;
    }

    // TODO: Implement multiply for two doubles
    double multiply(double a, double b) {
        // YOUR CODE HERE
        return 0.0;
    }

    // TODO: Implement multiply for vector (element-wise with scalar)
    vector<int> multiply(const vector<int> &vec, int scalar) {
        // YOUR CODE HERE
        return {};
    }
};

/*
 * ==================================================================================================
 * END OF EXERCISES
 * ==================================================================================================
 *
 * Total Possible Score: 110 points (100 + 10 bonus)
 *
 * Grading Scale:
 * 90-100+: Excellent (A)
 * 80-89:   Good (B)
 * 70-79:   Satisfactory (C)
 * 60-69:   Needs Improvement (D)
 * < 60:    Incomplete (F)
 *
 * To test your solutions, compile and run:
 * g++ -std=c++17 02_polymorphism_exercises.cpp ../../../tests/test_polymorphism.cpp -o test_polymorphism
 * ./test_polymorphism
 * ==================================================================================================
 */
