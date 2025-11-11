# C++ Learning Exercises with Automated Testing

This directory contains hands-on coding exercises with automated testing and scoring.

## How to Use This System

### 1. Study the Theory
First, review the learning material in the `practices/` directory:
- Example: `/practices/phase2_intermediate/oop_advanced/02_polymorphism.cpp`
- These files contain theory, examples, and interview questions

### 2. Complete the Exercises
Open the corresponding exercise file:
- Example: `/exercises/phase2_intermediate/oop_advanced/02_polymorphism_exercises.cpp`
- Read the instructions carefully
- Fill in the code where you see `// YOUR CODE HERE`
- **DO NOT** modify function signatures, class names, or test-related code

### 3. Test Your Solutions

#### Compile and Run Tests:
```bash
# Navigate to your project directory
cd /Users/aimanyounis/CLionProjects/C_Plus_Plus_Advanced

# Compile the test (for polymorphism example)
g++ -std=c++17 \
    exercises/phase2_intermediate/oop_advanced/02_polymorphism_exercises.cpp \
    tests/test_polymorphism.cpp \
    -o test_polymorphism

# Run the test
./test_polymorphism
```

### 4. Review Your Score
The test will output:
- ✓ Passed tests (in green)
- ✗ Failed tests (in red)
- Total points earned
- Final grade (A-F)
- Detailed breakdown by exercise

### 5. Iterate
- Review failed tests
- Fix your implementations
- Re-run tests until you achieve 100%

## Grading Scale

- **90-100+ points**: A (Excellent)
- **80-89 points**: B (Good)
- **70-79 points**: C (Satisfactory)
- **60-69 points**: D (Needs Improvement)
- **< 60 points**: F (Incomplete)

## Available Exercises

### Phase 2: Intermediate
- **Polymorphism** (02_polymorphism_exercises.cpp)
  - Banking System (25 pts)
  - Shape Calculator (25 pts)
  - Employee Management (25 pts)
  - Vehicle Fleet (25 pts)
  - Bonus: Function Overloading (10 pts)
  - **Total: 110 points**

*More exercises coming soon!*

## Tips for Success

1. **Read the theory first** - Understanding concepts before coding is crucial
2. **Test frequently** - Run tests after completing each exercise
3. **Read error messages** - Failed tests show what's expected vs. what you provided
4. **Use const correctness** - Pay attention to const qualifiers in function signatures
5. **Check edge cases** - Think about boundary conditions (negative numbers, empty inputs, etc.)
6. **Don't modify tests** - The test file should remain unchanged

## Troubleshooting

### Compilation Errors
- Check that you've implemented all required functions
- Verify function signatures match exactly
- Ensure you're using C++17 or later

### Linking Errors
- Make sure you're compiling both the exercise and test files together
- Use the full path or navigate to the project root

### Test Failures
- Read the test name carefully - it tells you what's being tested
- Check your logic against the requirements in comments
- Use a calculator to verify expected values

## Getting Help

If you're stuck:
1. Review the theory in the corresponding practice file
2. Check the interview questions section for insights
3. Look at similar examples in the codebase
4. Break down the problem into smaller steps

## Progress Tracking

Keep a log of your scores:
```
Date       | Exercise        | Score  | Grade
-----------|-----------------|--------|-------
2025-01-10 | Polymorphism    | 85/110 | B
```

Good luck and happy coding!