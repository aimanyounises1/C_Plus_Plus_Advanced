# Advanced Exercises Guide - "From Scratch" Mode

## What's Different?

### Basic Exercises (e.g., Polymorphism)
```cpp
// WE GIVE YOU:
class Account {
protected:
    string accountHolder;
    double balance;
public:
    Account(const string& holder, double initialBalance) {
        // YOUR CODE HERE  ← You just fill in the body
    }
};
```
**You only implement function bodies.**

### Advanced Exercises (NEW!)
```cpp
// WE GIVE YOU:
/*
 * Create a class "Account" with:
 * - Protected members: accountHolder (string), balance (double)
 * - Constructor that takes holder and initialBalance
 * - Virtual method: calculateInterest()
 */

// YOUR CODE HERE  ← You write EVERYTHING!
```
**You design and implement the entire class from scratch!**

---

## Why This Approach?

In real-world coding:
- ❌ You DON'T get pre-written class skeletons
- ❌ You DON'T get function signatures handed to you
- ✅ You MUST know syntax (constructor format, inheritance, etc.)
- ✅ You MUST design class structure yourself
- ✅ You MUST understand concepts deeply

This tests **true understanding**, not just filling in blanks!

---

## Available Advanced Exercises

### 1. **Virtual Functions Advanced** (120 points)
**File:** `exercises/phase2_intermediate/oop_advanced/03_virtual_functions_advanced.cpp`

**Challenges:**
- **Problem 1:** Plugin System - Design abstract Algorithm class with 3 implementations
- **Problem 2:** GPU Resource Management - Implement virtual destructors correctly (avoid memory leaks!)
- **Problem 3:** Compression Strategy - Use `final` keyword to prevent overriding
- **Bonus:** Multiple Inheritance - Implement class inheriting from 2 interfaces

**What You'll Learn:**
- Virtual destructor importance (memory management)
- When to use `final` keyword
- Multiple inheritance syntax
- Polymorphic plugin architecture

**Test:**
```bash
./run_test.sh virtual_functions_advanced
```

---

### 2. **Abstract Classes Advanced** (120 points)
**File:** `exercises/phase2_intermediate/oop_advanced/04_abstract_classes_advanced.cpp`

**Challenges:**
- **Problem 1:** Database Driver Architecture - Design abstract driver for MySQL, PostgreSQL, MongoDB
- **Problem 2:** Graphics Renderer - Template Method Pattern with `final` render algorithm
- **Problem 3:** Stream Processing - Interface Segregation Principle (multiple small interfaces)
- **Bonus:** Factory Pattern - Abstract factory for creating themed UI widgets

**What You'll Learn:**
- Abstract class design
- Template Method Pattern
- Interface Segregation Principle
- Abstract Factory Pattern
- When to use pure virtual vs regular virtual

**Test:**
```bash
./run_test.sh abstract_classes_advanced
```

---

## How to Approach These Exercises

### Step 1: Read the Requirements CAREFULLY
```
Create ABSTRACT class "DatabaseDriver" with:
- Pure virtual: bool connect(const string& connectionString)
- Protected member: bool isConnected
```

Translate to code:
```cpp
class DatabaseDriver {
protected:
    bool isConnected;
public:
    virtual bool connect(const string& connectionString) = 0;
    virtual ~DatabaseDriver() {}
};
```

### Step 2: Remember C++ Syntax Differences

#### From Java:
```java
// Java
public abstract class Foo extends Bar implements IBaz {
    public abstract void method();
}
```

#### To C++:
```cpp
// C++
class Foo : public Bar, public IBaz {
    virtual void method() = 0;
};
```

#### Key Differences:
| Feature | Java | C++ |
|---------|------|-----|
| Inheritance | `extends` | `: public` |
| Interface | `implements` | `: public` (same syntax!) |
| Abstract method | `abstract void m()` | `virtual void m() = 0` |
| Override | `@Override` | `override` (keyword) |
| Final method | `final void m()` | `void m() final` |
| Final class | `final class C` | `class C final` |
| Virtual destructor | N/A | `virtual ~Class() {}` (CRITICAL!) |

### Step 3: Write the Code

**Start with the base/abstract class:**
```cpp
class Algorithm {
public:
    virtual string getName() const = 0;  // Pure virtual
    virtual void execute() = 0;
    virtual ~Algorithm() {}  // Virtual destructor!
};
```

**Then implement derived classes:**
```cpp
class MatrixMultiply : public Algorithm {
public:
    string getName() const override { return "Matrix Multiplication"; }
    void execute() override { /* implementation */ }
};
```

### Step 4: Test Incrementally

Don't write everything at once!

```bash
# Write Problem 1 only
./run_test.sh virtual_functions_advanced

# See which tests pass/fail
# Fix failures
# Move to Problem 2
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Forgetting Virtual Destructor
```cpp
// ❌ WRONG - Memory leak!
class Base {
public:
    ~Base() {}  // NOT virtual!
};

class Derived : public Base {
private:
    int* data;
public:
    Derived() : data(new int[100]) {}
    ~Derived() { delete[] data; }
};

Base* ptr = new Derived();
delete ptr;  // Only ~Base() called, ~Derived() NOT called = LEAK!
```

```cpp
// ✅ CORRECT
class Base {
public:
    virtual ~Base() {}  // Virtual!
};
// Now delete ptr calls both destructors!
```

### Pitfall 2: Wrong Inheritance Syntax
```cpp
// ❌ WRONG
class Derived extends Base { };  // Java syntax!

// ❌ WRONG
class Derived : Base { };  // Missing 'public'!

// ✅ CORRECT
class Derived : public Base { };
```

### Pitfall 3: Forgetting `const` on Getters
```cpp
// ❌ WRONG
class Foo {
    string getName() { return name; }  // Missing const!
};

const Foo foo;
foo.getName();  // ERROR! Can't call non-const method on const object!

// ✅ CORRECT
class Foo {
    string getName() const { return name; }  // const!
};
```

### Pitfall 4: Pure Virtual Without `= 0`
```cpp
// ❌ WRONG
virtual void method();  // Not pure virtual, needs implementation!

// ✅ CORRECT (for pure virtual)
virtual void method() = 0;  // Pure virtual

// ✅ CORRECT (for regular virtual with default)
virtual void method() { /* default implementation */ }
```

### Pitfall 5: Multiple Inheritance Syntax
```cpp
// ❌ WRONG (Java thinking)
class Packet implements ISerializable, ILoggable { };

// ✅ CORRECT (C++)
class Packet : public ISerializable, public ILoggable { };
// Same syntax for classes and interfaces!
```

---

## Testing Strategy

### 1. Compile First
```bash
./run_test.sh virtual_functions_advanced
```

**If compilation fails:**
- Read error messages carefully
- Common issues:
  - Missing semicolon after class definition
  - Missing `override` keyword
  - Wrong function signature
  - Missing virtual destructor

### 2. Check Test Output
```
✓ MatrixMultiply creation (2 points)
✗ MatrixMultiply getName (2 points)
```

The test name tells you EXACTLY what failed!

### 3. Fix One Problem at a Time
Don't try to fix everything at once. Fix one test, re-run, repeat.

---

## Grading

Both advanced exercises: **120 points** (100 required + 20 bonus)

### Grade Scale:
- **90-100+**: A (Excellent - Production ready!)
- **80-89**: B (Good - Minor issues)
- **70-79**: C (Satisfactory - More practice needed)
- **60-69**: D (Needs work - Review concepts)
- **<60**: F (Incomplete - Study theory again)

---

## Study Resources

Before starting:
1. **Review theory:** `practices/phase2_intermediate/oop_advanced/03_virtual_functions.cpp`
2. **Read interview questions** at the end of practice files
3. **Understand concepts** before coding

Key concepts to master:
- Virtual functions & vtable mechanism
- Pure virtual vs regular virtual
- Virtual destructors (WHY they're critical)
- `override` keyword (compile-time safety)
- `final` keyword (prevent override/inheritance)
- Multiple inheritance syntax
- Abstract classes vs interfaces (C++ convention)

---

## Tips for Success

### 1. Write Headers First
```cpp
// Start with class declaration
class MyClass {
public:
    // Declare all methods
private:
    // Declare all members
};

// Then implement methods
```

### 2. Use Constructor Initialization Lists
```cpp
// ✅ GOOD
MyClass(int x, string s) : memberX(x), memberS(s) {}

// ❌ AVOID (less efficient)
MyClass(int x, string s) {
    memberX = x;
    memberS = s;
}
```

### 3. Test Each Problem Separately
Don't write all 4 problems at once! Test incrementally.

### 4. Read Test Errors Carefully
```
Expected: "MySQL"
Actual: "mysql"
```
Case matters!

### 5. Use the Practice Files
If stuck, review similar examples in `practices/` directory.

---

## Next Steps

After completing both advanced exercises with A grades:

1. ✅ Review solutions for optimization opportunities
2. ✅ Try implementing bonus problems for extra points
3. ✅ Create your own exercises for other topics
4. ✅ Move to next chapter (Operator Overloading, Move Semantics, etc.)

---

## Need Help?

1. **Read requirements again** - Most answers are in the problem description
2. **Check syntax** - Review C++ vs Java differences above
3. **Compiler errors** - Read them carefully, they often tell you exactly what's wrong
4. **Test failures** - The test name describes what it's testing
5. **Study theory** - Go back to practice files and review concepts

**Remember:** Struggling is part of learning! The challenge makes you a better programmer.

---

## Example: Complete Workflow

```bash
# 1. Read the exercise file
cat exercises/phase2_intermediate/oop_advanced/03_virtual_functions_advanced.cpp

# 2. Start with Problem 1
# Write the Algorithm abstract class
# Write MatrixMultiply implementation

# 3. Test
./run_test.sh virtual_functions_advanced

# Output:
# ✓ MatrixMultiply creation (2 points)
# ✓ MatrixMultiply getName (2 points)
# ✗ MatrixMultiply getMemory (2 points)
# ^ Fix this!

# 4. Fix the getRequiredMemoryMB() method
# Re-test

# 5. Repeat until Problem 1 passes completely

# 6. Move to Problem 2, repeat process

# 7. Final test - aim for 100+ points!
```

---

**Good luck! You're coding like a professional now - no training wheels!**