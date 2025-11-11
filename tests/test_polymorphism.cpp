/*
 * ==================================================================================================
 * POLYMORPHISM TEST SUITE
 * ==================================================================================================
 * This file tests your polymorphism exercise implementations and provides a score.
 * DO NOT MODIFY THIS FILE!
 * ==================================================================================================
 */

#include "../exercises/phase2_intermediate/oop_advanced/02_polymorphism_exercises.cpp"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <sstream>

using namespace std;

// Test result tracking
int totalTests = 0;
int passedTests = 0;
int totalPoints = 0;
int earnedPoints = 0;

// Color codes for terminal output
const string GREEN = "\033[1;32m";
const string RED = "\033[1;31m";
const string YELLOW = "\033[1;33m";
const string BLUE = "\033[1;34m";
const string RESET = "\033[0m";
const string BOLD = "\033[1m";

void printHeader(const string& title) {
    cout << "\n" << BLUE << BOLD << "╔══════════════════════════════════════════════════════════════╗" << RESET << endl;
    cout << BLUE << BOLD << "║ " << setw(60) << left << title << " ║" << RESET << endl;
    cout << BLUE << BOLD << "╚══════════════════════════════════════════════════════════════╝" << RESET << endl;
}

void testCase(const string& testName, bool passed, int points) {
    totalTests++;
    totalPoints += points;

    if (passed) {
        passedTests++;
        earnedPoints += points;
        cout << GREEN << "✓ " << RESET << testName << " (" << points << " points)" << endl;
    } else {
        cout << RED << "✗ " << RESET << testName << " (" << points << " points)" << endl;
    }
}

bool doubleEquals(double a, double b, double epsilon = 0.01) {
    return abs(a - b) < epsilon;
}

/*
 * ==================================================================================================
 * EXERCISE 1 TESTS: Banking System (25 points)
 * ==================================================================================================
 */
void testBankingSystem() {
    printHeader("Exercise 1: Banking System (25 points)");

    try {
        // Test 1: Account creation and basic operations (5 points)
        Account acc("John Doe", "ACC001", 1000.0);
        testCase("Account creation", acc.getBalance() == 1000.0, 2);

        // Test 2: Deposit (3 points)
        acc.deposit(500.0);
        testCase("Deposit functionality", acc.getBalance() == 1500.0, 3);

        // Test 3: Withdraw (3 points)
        bool success = acc.withdraw(300.0);
        testCase("Withdraw functionality", success && acc.getBalance() == 1200.0, 3);

        // Test 4: Insufficient funds (2 points)
        success = acc.withdraw(2000.0);
        testCase("Insufficient funds check", !success && acc.getBalance() == 1200.0, 2);

        // Test 5: Base interest calculation (3 points)
        double interest = acc.calculateInterest();
        testCase("Base account interest (2%)", doubleEquals(interest, 24.0), 3);

        // Test 6: SavingsAccount interest (4 points)
        SavingsAccount savings("Jane Smith", "SAV001", 1000.0);
        double savingsInterest = savings.calculateInterest();
        testCase("Savings account interest (3.5%)", doubleEquals(savingsInterest, 35.0), 4);

        // Test 7: CheckingAccount interest (3 points)
        CheckingAccount checking("Bob Wilson", "CHK001", 1000.0, 500.0);
        double checkingInterest = checking.calculateInterest();
        testCase("Checking account interest (1%)", doubleEquals(checkingInterest, 10.0), 3);

        // Test 8: Overdraft protection (4 points)
        checking.withdraw(1200.0);  // Should succeed with overdraft
        testCase("Overdraft protection", checking.getBalance() == -200.0, 4);

    } catch (const exception& e) {
        cout << RED << "Exception caught: " << e.what() << RESET << endl;
        testCase("Banking System Exception Test", false, 25);
    }
}

/*
 * ==================================================================================================
 * EXERCISE 2 TESTS: Shape Calculator (25 points)
 * ==================================================================================================
 */
void testShapeCalculator() {
    printHeader("Exercise 2: Shape Calculator (25 points)");

    try {
        // Test 1: Circle area (4 points)
        Circle circle(5.0);
        testCase("Circle area calculation", doubleEquals(circle.getArea(), 78.54, 0.1), 4);

        // Test 2: Circle perimeter (4 points)
        testCase("Circle perimeter calculation", doubleEquals(circle.getPerimeter(), 31.42, 0.1), 4);

        // Test 3: Rectangle area (3 points)
        Rectangle rect(4.0, 6.0);
        testCase("Rectangle area calculation", doubleEquals(rect.getArea(), 24.0), 3);

        // Test 4: Rectangle perimeter (3 points)
        testCase("Rectangle perimeter calculation", doubleEquals(rect.getPerimeter(), 20.0), 3);

        // Test 5: Triangle area (Heron's formula) (5 points)
        Triangle triangle(3.0, 4.0, 5.0);  // Right triangle
        testCase("Triangle area (Heron's formula)", doubleEquals(triangle.getArea(), 6.0), 5);

        // Test 6: Triangle perimeter (3 points)
        testCase("Triangle perimeter calculation", doubleEquals(triangle.getPerimeter(), 12.0), 3);

        // Test 7: Polymorphic behavior (3 points)
        Shape* shapes[] = {&circle, &rect, &triangle};
        double totalArea = 0;
        for (int i = 0; i < 3; i++) {
            totalArea += shapes[i]->getArea();
        }
        testCase("Polymorphic shape processing", doubleEquals(totalArea, 108.54, 0.1), 3);

    } catch (const exception& e) {
        cout << RED << "Exception caught: " << e.what() << RESET << endl;
        testCase("Shape Calculator Exception Test", false, 25);
    }
}

/*
 * ==================================================================================================
 * EXERCISE 3 TESTS: Employee Management System (25 points)
 * ==================================================================================================
 */
void testEmployeeManagement() {
    printHeader("Exercise 3: Employee Management System (25 points)");

    try {
        // Test 1: Base employee pay (3 points)
        Employee emp("Alice", 1001, 5000.0);
        testCase("Base employee salary", doubleEquals(emp.calculatePay(), 5000.0), 3);

        // Test 2: Manager bonus (5 points)
        Manager mgr("Bob", 2001, 8000.0, 5);
        testCase("Manager bonus (20%)", doubleEquals(mgr.calculatePay(), 9600.0), 5);

        // Test 3: Engineer project bonus (5 points)
        Engineer eng("Charlie", 3001, 7000.0, 10);
        testCase("Engineer project bonus", doubleEquals(eng.calculatePay(), 7500.0), 5);

        // Test 4: Intern reduced pay (5 points)
        Intern intern("Diana", 4001, 3000.0, 6);
        testCase("Intern reduced pay (60%)", doubleEquals(intern.calculatePay(), 1800.0), 5);

        // Test 5: Polymorphic payroll calculation (7 points)
        Employee* employees[] = {&emp, &mgr, &eng, &intern};
        double totalPayroll = 0;
        for (int i = 0; i < 4; i++) {
            totalPayroll += employees[i]->calculatePay();
        }
        testCase("Polymorphic payroll calculation", doubleEquals(totalPayroll, 23900.0), 7);

    } catch (const exception& e) {
        cout << RED << "Exception caught: " << e.what() << RESET << endl;
        testCase("Employee Management Exception Test", false, 25);
    }
}

/*
 * ==================================================================================================
 * EXERCISE 4 TESTS: Vehicle Fleet Management (25 points)
 * ==================================================================================================
 */
void testVehicleFleet() {
    printHeader("Exercise 4: Vehicle Fleet Management (25 points)");

    try {
        // Test 1: Car fuel efficiency (3 points)
        Car car("Toyota", "Camry", 2023, 4);
        testCase("Car fuel efficiency (35 mpg)", doubleEquals(car.getFuelEfficiency(), 35.0), 3);

        // Test 2: Car maintenance cost (3 points)
        testCase("Car maintenance cost", doubleEquals(car.getMaintenanceCost(), 400.0), 3);

        // Test 3: Truck fuel efficiency (3 points)
        Truck truck("Ford", "F-150", 2023, 1000.0);
        testCase("Truck fuel efficiency (18 mpg)", doubleEquals(truck.getFuelEfficiency(), 18.0), 3);

        // Test 4: Truck maintenance cost (3 points)
        testCase("Truck maintenance cost", doubleEquals(truck.getMaintenanceCost(), 800.0), 3);

        // Test 5: Motorcycle fuel efficiency (3 points)
        Motorcycle bike("Harley", "Street 750", 2023, "cruiser");
        testCase("Motorcycle fuel efficiency (55 mpg)", doubleEquals(bike.getFuelEfficiency(), 55.0), 3);

        // Test 6: Motorcycle maintenance cost (3 points)
        testCase("Motorcycle maintenance cost", doubleEquals(bike.getMaintenanceCost(), 200.0), 3);

        // Test 7: Refuel functionality (2 points)
        car.drive(35.0);  // Use some fuel
        car.refuel();
        testCase("Refuel functionality", doubleEquals(car.getFuelLevel(), 100.0), 2);

        // Test 8: Drive and fuel consumption (5 points)
        Car testCar("Honda", "Accord", 2023, 4);
        testCar.drive(35.0);  // Should use approximately 1 gallon = 1% fuel
        testCase("Drive and fuel consumption", car.getFuelLevel() <= 99.5, 5);

    } catch (const exception& e) {
        cout << RED << "Exception caught: " << e.what() << RESET << endl;
        testCase("Vehicle Fleet Exception Test", false, 25);
    }
}

/*
 * ==================================================================================================
 * BONUS TESTS: Function Overloading (10 points)
 * ==================================================================================================
 */
void testFunctionOverloading() {
    printHeader("Bonus: Function Overloading (10 points)");

    try {
        MathOperations math;

        // Test 1: Multiply two integers (2 points)
        testCase("Multiply two integers", math.multiply(5, 3) == 15, 2);

        // Test 2: Multiply three integers (2 points)
        testCase("Multiply three integers", math.multiply(2, 3, 4) == 24, 2);

        // Test 3: Multiply two doubles (3 points)
        testCase("Multiply two doubles", doubleEquals(math.multiply(2.5, 4.0), 10.0), 3);

        // Test 4: Multiply vector by scalar (3 points)
        vector<int> vec = {1, 2, 3, 4};
        vector<int> result = math.multiply(vec, 3);
        bool vectorCorrect = (result.size() == 4 && result[0] == 3 &&
                             result[1] == 6 && result[2] == 9 && result[3] == 12);
        testCase("Multiply vector by scalar", vectorCorrect, 3);

    } catch (const exception& e) {
        cout << RED << "Exception caught: " << e.what() << RESET << endl;
        testCase("Function Overloading Exception Test", false, 10);
    }
}

/*
 * ==================================================================================================
 * MAIN TEST RUNNER
 * ==================================================================================================
 */
void printSummary() {
    cout << "\n" << BOLD << "═══════════════════════════════════════════════════════════════" << RESET << endl;
    cout << BOLD << "                       TEST SUMMARY                             " << RESET << endl;
    cout << BOLD << "═══════════════════════════════════════════════════════════════" << RESET << endl;

    double percentage = (totalPoints > 0) ? (100.0 * earnedPoints / totalPoints) : 0.0;

    cout << "\nTests Passed: " << passedTests << "/" << totalTests << endl;
    cout << "Points Earned: " << earnedPoints << "/" << totalPoints << endl;
    cout << "Percentage: " << fixed << setprecision(1) << percentage << "%" << endl;

    // Grade determination
    string grade, status, color;
    if (percentage >= 90) {
        grade = "A";
        status = "Excellent!";
        color = GREEN;
    } else if (percentage >= 80) {
        grade = "B";
        status = "Good!";
        color = GREEN;
    } else if (percentage >= 70) {
        grade = "C";
        status = "Satisfactory";
        color = YELLOW;
    } else if (percentage >= 60) {
        grade = "D";
        status = "Needs Improvement";
        color = YELLOW;
    } else {
        grade = "F";
        status = "Incomplete";
        color = RED;
    }

    cout << "\n" << color << BOLD << "Grade: " << grade << " - " << status << RESET << endl;

    // Detailed breakdown
    cout << "\n" << BOLD << "Score Breakdown:" << RESET << endl;
    cout << "  Banking System:         Exercise 1 (25 points)" << endl;
    cout << "  Shape Calculator:       Exercise 2 (25 points)" << endl;
    cout << "  Employee Management:    Exercise 3 (25 points)" << endl;
    cout << "  Vehicle Fleet:          Exercise 4 (25 points)" << endl;
    cout << "  Function Overloading:   Bonus     (10 points)" << endl;

    cout << "\n" << BOLD << "═══════════════════════════════════════════════════════════════" << RESET << endl;

    if (percentage < 100) {
        cout << "\n" << YELLOW << "Keep practicing! Review the failed tests and try again." << RESET << endl;
    } else {
        cout << "\n" << GREEN << "Perfect score! Excellent work on polymorphism!" << RESET << endl;
    }
}

int main() {
    cout << BOLD << "\n╔═══════════════════════════════════════════════════════════════╗" << RESET << endl;
    cout << BOLD << "║       POLYMORPHISM EXERCISES - AUTOMATED TEST SUITE           ║" << RESET << endl;
    cout << BOLD << "╚═══════════════════════════════════════════════════════════════╝" << RESET << endl;

    // Run all tests
    testBankingSystem();
    testShapeCalculator();
    testEmployeeManagement();
    testVehicleFleet();
    testFunctionOverloading();

    // Print final summary
    printSummary();

    return (earnedPoints == totalPoints) ? 0 : 1;
}