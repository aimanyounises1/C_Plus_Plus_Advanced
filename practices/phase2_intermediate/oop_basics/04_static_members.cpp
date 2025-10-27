/*
 * ==================================================================================================
 * Exercise: Static Members in C++
 * ==================================================================================================
 * Difficulty: Intermediate | Time: 40-50 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Understand static member variables (class-level data)
 * 2. Master static member functions
 * 3. Learn static initialization and lifetime
 * 4. Apply static members for shared state
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Singleton pattern for GPU context management
 * - Shared counters/statistics across objects
 * - Thread-safe static initialization (C++11)
 * - Static registration patterns
 * ==================================================================================================
 */

#include <iostream>
#include <string>

using namespace std;

/*
 * ==================================================================================================
 * THEORY: Static Members
 * ==================================================================================================
 *
 * STATIC MEMBER VARIABLE:
 * - Shared by ALL objects of the class
 * - Only ONE copy exists (not per-object)
 * - Must be defined outside class
 * - Lifetime: Entire program duration
 *
 * STATIC MEMBER FUNCTION:
 * - Can be called without an object
 * - Can only access static members
 * - No 'this' pointer
 * - Syntax: ClassName::functionName()
 *
 * ==================================================================================================
 */

/*
 * EXERCISE 1: Static Member Variables (15 min)
 */

// TODO 1.1: Create a "Counter" class with static count
class Counter {
private:
    static int totalCount;  // Declaration
    int id;

public:
    Counter() {
        id = ++totalCount;
        cout << "Counter " << id << " created. Total: " << totalCount << endl;
    }

    ~Counter() {
        --totalCount;
        cout << "Counter " << id << " destroyed. Total: " << totalCount << endl;
    }

    static int getCount() {  // Static function
        return totalCount;
    }
};

// TODO: Define static member outside class (required!)
// int Counter::totalCount = 0;

// TODO 1.2: Test static member
// Create multiple Counter objects
// Call Counter::getCount() without an object

/*
 * EXERCISE 2: Singleton Pattern (15 min)
 */

// TODO 2.1: Implement Singleton class
class DatabaseConnection {
private:
    static DatabaseConnection* instance;
    string connectionString;

    // Private constructor
    DatabaseConnection(const string& conn) : connectionString(conn) {
        cout << "Database connected: " << conn << endl;
    }

public:
    // Delete copy constructor and assignment
    DatabaseConnection(const DatabaseConnection&) = delete;
    DatabaseConnection& operator=(const DatabaseConnection&) = delete;

    static DatabaseConnection* getInstance(const string& conn = "localhost") {
        if (instance == nullptr) {
            instance = new DatabaseConnection(conn);
        }
        return instance;
    }

    void query(const string& sql) {
        cout << "Executing: " << sql << endl;
    }
};

// TODO: Define static instance pointer
// DatabaseConnection* DatabaseConnection::instance = nullptr;

// TODO 2.2: Test singleton
// Try to get multiple instances - should return same object

/*
 * EXERCISE 3: Static Const Members (10 min)
 */

// TODO 3.1: Create class with static constants
class MathConstants {
public:
    static const double PI;
    static const double E;
    static constexpr double SQRT2 = 1.41421356;  // constexpr can be initialized in-class
};

// TODO: Define static const members
// const double MathConstants::PI = 3.14159265;
// const double MathConstants::E = 2.71828182;

/*
 * EXERCISE 4: Object Counting (15 min)
 */

// TODO 4.1: Track object creation/destruction
class Tracker {
private:
    static int objectsCreated;
    static int objectsAlive;
    int id;

public:
    Tracker() {
        id = ++objectsCreated;
        ++objectsAlive;
    }

    ~Tracker() {
        --objectsAlive;
    }

    static void printStats() {
        cout << "Total created: " << objectsCreated << endl;
        cout << "Currently alive: " << objectsAlive << endl;
    }
};

// TODO: Define statics
// int Tracker::objectsCreated = 0;
// int Tracker::objectsAlive = 0;

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is a static member variable?
 * A: Shared by all objects, one copy per class (not per object), class-level data
 *
 * Q2: How do you access a static member?
 * A: ClassName::member or object.member (both work, but ClassName:: is clearer)
 *
 * Q3: Can static functions access non-static members?
 * A: No! They have no 'this' pointer, so they can't access instance members
 *
 * Q4: Why must static members be defined outside the class?
 * A: Declaration in class doesn't allocate memory. Definition provides storage
 *
 * Q5: What is the Singleton pattern?
 * A: Ensure only ONE instance of a class exists. Use private constructor + static instance
 *
 * Q6: When is a static member initialized?
 * A: Before main() starts (static initialization phase), or on first use (static local)
 *
 * Q7: Are static members thread-safe?
 * A: C++11+: Static local initialization is thread-safe. Static members need manual synchronization
 *
 * Q8: What is the difference between static member and global variable?
 * A: Static member has class scope, access control, and is tied to class semantics
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * ==================================================================================================
 *
 * 1. GPU Context Management: Singleton pattern for cudaContext, cuBLAS handles
 * 2. Device Query: Static functions to query GPU properties without object
 * 3. Shared Statistics: Track total GPU memory allocated across all objects
 * 4. Thread Safety: Static initialization for GPU runtime (thread-safe in C++11+)
 * 5. Factory Registration: Static registration of kernel types
 *
 * ==================================================================================================
 * COMPILATION: g++ -std=c++17 04_static_members.cpp -o static
 *
 * LEARNING CHECKLIST:
 * ☐ Declare and define static members
 * ☐ Use static functions without objects
 * ☐ Implement Singleton pattern
 * ☐ Understand static initialization
 * ☐ Use static for shared state
 * ☐ Know thread safety considerations
 * ==================================================================================================
 */

int main() {
    cout << "=== Static Members Practice ===" << endl;

    // Test your implementations here

    return 0;
}
