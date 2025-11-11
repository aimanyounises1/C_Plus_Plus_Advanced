# Exercise System - Quick Start Guide

Welcome to your automated C++ learning and testing system!

## What You Get

✓ **Hands-on exercises** - Real coding challenges, not just examples
✓ **Automated testing** - Instant feedback on your solutions
✓ **Score tracking** - See exactly how well you're doing (0-110 points)
✓ **Detailed feedback** - Know which tests pass/fail
✓ **Solution reference** - Check correct implementations when stuck

---

## Quick Start (3 Steps)

### Step 1: Open the Exercise File
```bash
# For polymorphism:
open exercises/phase2_intermediate/oop_advanced/02_polymorphism_exercises.cpp
```

Or use your favorite editor:
```bash
vim exercises/phase2_intermediate/oop_advanced/02_polymorphism_exercises.cpp
code exercises/phase2_intermediate/oop_advanced/02_polymorphism_exercises.cpp
```

### Step 2: Complete the TODOs
Look for comments like:
```cpp
// TODO: Implement constructor
Account(const string& holder, const string& accNum, double initialBalance) {
    // YOUR CODE HERE
}
```

Fill in your implementation!

### Step 3: Run the Tests
**Easy way (recommended):**
```bash
./run_test.sh polymorphism
```

**Manual way:**
```bash
g++ -std=c++17 \
    exercises/phase2_intermediate/oop_advanced/02_polymorphism_exercises.cpp \
    tests/test_polymorphism.cpp \
    -o test_polymorphism

./test_polymorphism
```

---

## Understanding Your Results

### Output Format
```
╔══════════════════════════════════════════════════════════════╗
║ Exercise 1: Banking System (25 points)                      ║
╚══════════════════════════════════════════════════════════════╝

✓ Account creation (2 points)
✓ Deposit functionality (3 points)
✗ Withdraw functionality (3 points)
✗ Insufficient funds check (2 points)
...
```

### Grade Scale
- **A (90-100+)**: Excellent - You've mastered this topic!
- **B (80-89)**: Good - Minor improvements needed
- **C (70-79)**: Satisfactory - Review failed sections
- **D (60-69)**: Needs work - Revisit the theory
- **F (<60)**: Incomplete - Study and try again

---

## Current Topics

### Polymorphism (110 points total)

**Exercise 1: Banking System (25 pts)**
- Implement Account, SavingsAccount, CheckingAccount classes
- Learn: Virtual functions, inheritance, method overriding

**Exercise 2: Shape Calculator (25 pts)**
- Implement Circle, Rectangle, Triangle classes
- Learn: Pure virtual functions, abstract classes

**Exercise 3: Employee Management (25 pts)**
- Implement Manager, Engineer, Intern classes
- Learn: Polymorphic behavior, dynamic dispatch

**Exercise 4: Vehicle Fleet (25 pts)**
- Implement Car, Truck, Motorcycle classes
- Learn: Complex polymorphic hierarchies

**Bonus: Function Overloading (10 pts)**
- Implement compile-time polymorphism
- Learn: Function overloading with different signatures

---

## Tips for Success

### 1. Study First
Before coding, review:
```
practices/phase2_intermediate/oop_advanced/02_polymorphism.cpp
```

### 2. Test Frequently
Don't wait until everything is done. Test after each exercise!

### 3. Read Error Messages
```
✗ Withdraw functionality (3 points)
```
This tells you WHAT failed. Check that function's logic.

### 4. Check Your Math
Many tests check calculated values:
```cpp
// 3.5% interest on $1000 should be $35
testCase("Savings interest", doubleEquals(interest, 35.0), 4);
```

### 5. Use const Correctly
```cpp
double getBalance() const { return balance; }
//                  ^^^^^ - Don't forget this!
```

### 6. Stuck? Check Solutions
```
exercises/phase2_intermediate/oop_advanced/02_polymorphism_SOLUTIONS.cpp
```
**But try on your own first!**

---

## Common Issues

### Compilation Errors

**Problem**: `undefined reference to...`
**Solution**: Make sure you compiled both exercise and test files together

**Problem**: `error: cannot declare variable 'x' to be of abstract type`
**Solution**: You need to implement all pure virtual functions

### Test Failures

**Problem**: "Account creation" test fails
**Solution**: Check your constructor - it should initialize all member variables

**Problem**: Math-related tests fail
**Solution**: Use a calculator! 3.5% = 0.035, not 0.35

**Problem**: All tests fail
**Solution**: You might not have implemented the function bodies (still returning 0 or default values)

---

## File Structure

```
C_Plus_Plus_Advanced/
├── practices/                       # Study materials (theory + examples)
│   └── phase2_intermediate/
│       └── oop_advanced/
│           └── 02_polymorphism.cpp
│
├── exercises/                       # Your coding challenges
│   ├── README.md                    # Detailed guide
│   └── phase2_intermediate/
│       └── oop_advanced/
│           ├── 02_polymorphism_exercises.cpp    # Fill this in
│           └── 02_polymorphism_SOLUTIONS.cpp    # Reference solutions
│
├── tests/                           # Automated test files
│   └── test_polymorphism.cpp
│
├── run_test.sh                      # Easy test runner script
└── EXERCISE_QUICK_START.md          # This file
```

---

## Workflow Example

```bash
# 1. Study the theory
cat practices/phase2_intermediate/oop_advanced/02_polymorphism.cpp

# 2. Open exercise in your editor
vim exercises/phase2_intermediate/oop_advanced/02_polymorphism_exercises.cpp

# 3. Work on Exercise 1 (Banking System)
# ... implement Account, SavingsAccount, CheckingAccount ...

# 4. Test
./run_test.sh polymorphism

# Output shows:
# ✓ Account creation (2 points)
# ✓ Deposit functionality (3 points)
# ✗ Withdraw functionality (3 points)
# Score: 5/25 for Exercise 1

# 5. Fix withdraw function
# ... fix the logic ...

# 6. Test again
./run_test.sh polymorphism

# Output shows all of Exercise 1 passing!
# ✓ All Exercise 1 tests passed (25/25)

# 7. Move on to Exercise 2...
```

---

## Need Help?

1. **Read the theory first** - Most answers are in the practice files
2. **Check function signatures** - Make sure they match exactly (including `const`)
3. **Review test output** - It tells you what's expected
4. **Debug incrementally** - Fix one test at a time
5. **Check solutions** - But only after trying yourself!

---

## Progress Tracking

Keep track of your scores:

| Date | Exercise | Score | Grade | Status |
|------|----------|-------|-------|--------|
| 2025-01-10 | Polymorphism | 85/110 | B | In progress |
| | | | | |

---

## Next Steps

Once you achieve 90+ on polymorphism:
1. Review the interview questions in the practice file
2. Implement the bonus exercises for extra points
3. More exercises coming soon for other topics!

**Happy Coding! 🚀**