/*
 * Exercise: Operators
 * Difficulty: Beginner
 * Time: 35-45 minutes
 * Topics: Arithmetic, relational, logical, bitwise, assignment operators, operator precedence
 *
 * LEARNING OBJECTIVES:
 * - Master all C++ operators
 * - Understand operator precedence
 * - Practice bitwise operations (critical for GPU programming!)
 * - Learn compound assignment operators
 *
 * INTERVIEW RELEVANCE:
 * - Bitwise operations are frequently asked in technical interviews
 * - Understanding precedence prevents bugs
 * - GPU programming extensively uses bitwise operations for optimization
 * - Nvidia interviews often include bit manipulation problems
 */

#include <iostream>
#include <iomanip>
#include <bitset>

int main() {
    std::cout << "=== Operators Exercises ===\n\n";

    // ========================================================================
    // EXERCISE 1: Arithmetic Operators (5 min)
    // ========================================================================
    std::cout << "Exercise 1: Arithmetic Operators\n";
    std::cout << "--------------------------------\n";

    // TODO 1.1: Declare two integers a=10, b=3
    int a = 10;
    int b = 3;

    // TODO 1.2: Perform and print: addition, subtraction, multiplication, division, modulo
    // Example: std::cout << "a + b = " << (a + b) << "\n";
    std::cout << "a + b = " << (a + b) << "\n";
    std::cout << "a - b = " << (a - b) << "\n";
    std::cout << "a * b = " << (a + b) << "\n";
    std::cout << "a / b = " << (a / b) << "\n";

    // TODO 1.3: What happens with division? (integer vs float) the result will be integer
    // Calculate 10/3 as int and as double
    std::cout << "a / b = " << (a / b) << "\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 2: Increment/Decrement Operators (5 min)
    // ========================================================================
    std::cout << "Exercise 2: Increment/Decrement\n";
    std::cout << "-------------------------------\n";

    int x = 5;

    // TODO 2.1: Post-increment (x++)
    // Print x, then x++, then x again to see the difference
    std::cout << "x++ = " << x++ << std::endl;
    x = --x;

    // TODO 2.2: Pre-increment (++x)
    // Reset x to 5, then print ++x and x
    x = ++x;

    // TODO 2.3: Demonstrate the difference
    int y = 10;
    int result1 = y++; // Post: use then increment
    int result2 = ++y; // Pre: increment then use
    // Print both results and y
    std::cout << "result1 = " << result1 << std::endl;
    std::cout << "result2 = " << result2 << std::endl;

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 3: Relational Operators (5 min)
    // ========================================================================
    std::cout << "Exercise 3: Relational Operators\n";
    std::cout << "--------------------------------\n";

    int num1 = 15, num2 = 20;

    // TODO 3.1: Test all relational operators
    // Print results of: ==, !=, <, >, <=, >=
    // Example: std::cout << "num1 == num2: " << (num1 == num2) << "\n";
    std::cout << "num1 ==  num2 = " << (num1 == num2) << std::endl;
    std::cout << "num1 <= num2 = " << (num1 <= num2) << std::endl;
    std::cout << "num1 >=  num2 = " << (num1 >= num2) << std::endl;

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 4: Logical Operators (5 min)
    // ========================================================================
    std::cout << "Exercise 4: Logical Operators\n";
    std::cout << "-----------------------------\n";

    bool isStudent = true;
    bool hasID = false;

    // TODO 4.1: AND operator (&&)
    // Can enter building if student AND has ID
    std::cout << "And operator " << (isStudent && hasID) << std::endl;

    // TODO 4.2: OR operator (||)
    // Can get discount if student OR senior (create bool isSenior)
    std::cout << "Or operator " << (isStudent || hasID) << std::endl;


    // TODO 4.3: NOT operator (!)
    // Invert a boolean
    std::cout << "Or operator " << !(isStudent || hasID) << std::endl;


    // TODO 4.4: Complex expression
    // (isStudent && hasID) || isSenior
    bool isSenior = true;
    std::cout << "Complex expression" << ((isSenior && isStudent) || isSenior) << std::endl;

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 5: Bitwise Operators (10 min) - IMPORTANT FOR GPU!
    // ========================================================================
    std::cout << "Exercise 5: Bitwise Operators\n";
    std::cout << "-----------------------------\n";

    unsigned int a1 = 12; // Binary: 1100
    unsigned int b1 = 10; // Binary: 1010

    // TODO 5.1: Print binary representations
    std::cout << "a = " << a1 << " (binary: " << std::bitset<8>(a1) << ")\n";
    std::cout << "b = " << b1 << " (binary: " << std::bitset<8>(b1) << ")\n\n";

    // TODO 5.2: AND operator (&)
    // Result: 1100 & 1010 = 1000 (8)
    std::cout << "And operator on unassigned = " << (a1 & b1) << std::endl;

    // TODO 5.3: OR operator (|)
    // Result: 1100 | 1010 = 1110 (14)
    std::cout << "Or operator = " << (a1 | b1) << std::endl;

    // TODO 5.4: XOR operator (^)
    // Result: 1100 ^ 1010 = 0110 (6)
    std::cout << "XOR operator = " << (a1 ^ b1) << std::endl;


    // TODO 5.5: NOT operator (~)
    // Inverts all bits

    std::cout << " NOT operator =" << (~a1) << std::endl;
    std::cout << " NOT operator =" << (~b1) << std::endl;

    // TODO 5.6: Left shift (<<)
    // Multiplies by 2^n
    // Example: 12 << 1 = 24 (multiply by 2)
    std::cout << (a1 << b1) << std::endl;

    // TODO 5.7: Right shift (>>)
    // Divides by 2^n
    // Example: 12 >> 1 = 6 (divide by 2)


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 6: Assignment Operators (5 min)
    // ========================================================================
    std::cout << "Exercise 6: Assignment Operators\n";
    std::cout << "--------------------------------\n";

    int value = 100;

    // TODO 6.1: Compound assignment operators
    // +=, -=, *=, /=, %=
    // Example: value += 10 (same as value = value + 10)


    // TODO 6.2: Bitwise compound assignments
    // &=, |=, ^=, <<=, >>=


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 7: Operator Precedence (5 min)
    // ========================================================================
    std::cout << "Exercise 7: Operator Precedence\n";
    std::cout << "-------------------------------\n";

    // TODO 7.1: Predict the result, then verify
    int result = 2 + 3 * 4; // What is the result? (14 or 20?)


    // TODO 7.2: Use parentheses to change order
    int result_2 = (2 + 3) * 4;


    // TODO 7.3: Complex expression
    bool complex = 5 > 3 && 10 < 20 || 2 == 2;


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 8: Practical Applications (10 min)
    // ========================================================================
    std::cout << "Exercise 8: Practical Applications\n";
    std::cout << "----------------------------------\n";

    // TODO 8.1: Check if a number is even using modulo


    // TODO 8.2: Check if a number is a power of 2 using bitwise
    // Hint: Power of 2 has only one bit set
    // n & (n-1) == 0 for powers of 2
    unsigned int num = 16;


    // TODO 8.3: Swap two numbers using XOR (no temp variable)
    int p = 5, q = 10;
    std::cout << "Before: p=" << p << ", q=" << q << "\n";
    // Swap using: p ^= q; q ^= p; p ^= q;


    std::cout << "After: p=" << p << ", q=" << q << "\n";

    // TODO 8.4: Extract specific bits
    // Get the 3rd bit (0-indexed) of a number
    unsigned int data = 0b11010110; // Binary literal
    // Extract bit 3 (which is 0): (data >> 3) & 1


    // TODO 8.5: Set a specific bit
    // Set bit 2 to 1: data | (1 << 2)


    // TODO 8.6: Clear a specific bit
    // Clear bit 4: data & ~(1 << 4)


    std::cout << "\n";

    // ========================================================================
    // CHALLENGE EXERCISES (Optional - 15 min)
    // ========================================================================
    std::cout << "Challenge Exercises\n";
    std::cout << "-------------------\n";

    // CHALLENGE 1: Count the number of 1s in binary representation
    // Example: 13 (1101) has three 1s
    unsigned int number = 29; // Binary: 11101
    // Hint: Loop and check each bit, or use Brian Kernighan's algorithm
    auto binary = std::bitset<8>(number);
    std::cout << "Number of 1s is = " << binary.count() << std::endl;
    int count = 0;
    // Loop version
    for (auto i = 0; i < binary.size(); ++i) {
        if (binary[i] == 1) {
            count++;
        }
    }
    std::cout << "The number of the 1s  = " << count << std::endl;
    // CHALLENGE 2: Reverse the bits of a number
    // Example: 0b00001100 becomes 0b00110000
    unsigned char byte = 0b00001100;
    std::string bits_reversed = std::bitset<8>(byte).to_string();
    bits_reversed.reserve(bits_reversed.size());
    std::cout << "bits_reversed = "  << bits_reversed << std::endl;
    // CHALLENGE 3: Check if two numbers have opposite signs
    // Hint: Use XOR on the sign bits
    auto converted_byte = static_cast<unsigned char>(std::stoi(bits_reversed , nullptr, 2));

    std::cout << "The integers are equal ? = "  <<  (converted_byte == static_cast<int>(byte));

    // CHALLENGE 4: Find the only non-repeating element
    // Given array where every element appears twice except one
    // Example: [2, 3, 5, 3, 2] → 5 is non-repeating
    // Hint: XOR all elements (duplicates cancel out)
    int arr[] = {4, 2, 7, 2, 4};
    int n = 5;
    int start = arr[0];
    for (const int val : arr) start ^= val;
    std::cout << "The non repeating number is = " << start << std::endl;
    std::cout << "\n";

    // ========================================================================
    // COMMON INTERVIEW QUESTIONS
    // ========================================================================
    /*
     * Q1: What's the difference between & and &&?
     * A: & is bitwise AND (operates on bits)
     *    && is logical AND (operates on boolean values, short-circuits)
     *
     * Q2: What is operator precedence?
     * A: Order in which operators are evaluated.
     *    * and / before + and -
     *    Use parentheses to be explicit!
     *
     * Q3: How do you check if a number is a power of 2?
     * A: (n & (n-1)) == 0
     *    Powers of 2 have only one bit set: 8 = 1000, 8-1 = 0111
     *    8 & 7 = 1000 & 0111 = 0000
     *
     * Q4: Why use bitwise operations? bitwise operations are faster than arithmetic operations (programming , embedded systems, GPU kernels)
     * A: Much faster than arithmetic operations
     *    Essential in low-level programming, embedded systems, GPU kernels
     *    Used for flags, masks, compression
     *
     * Q5: What does x ^= x do? it will set the number to zero.
     * A: Sets x to 0. Any number XOR itself equals 0.
     *
     * Q6: How to swap without a temp variable? a = 1001 b = 0110 --> 0000
     * A: Using XOR: a ^= b; b ^= a; a ^= b; a = 0000 -> 1001 || a ^= b --> 0110
     *    Or arithmetic: a = a + b; b = a - b; a = a - b;
     */

    return 0;
}

/*
 * BITWISE OPERATIONS FOR GPU PROGRAMMING:
 * =======================================
 * Bitwise operations are EXTREMELY important in CUDA programming:
 *
 * 1. Thread Index Calculations:
 *    threadIdx.x & 31 → Get lane ID in warp (fast modulo 32)
 *
 * 2. Memory Alignment:
 *    addr & ~0x7F → Align to 128-byte boundary
 *
 * 3. Warp-Level Programming:
 *    Ballot, shuffle operations use bit masks
 *
 * 4. Atomic Operations:
 *    Often implement using compare-and-swap with bit manipulation
 *
 * 5. Flag Management:
 *    Efficiently pack multiple boolean flags into single int
 *
 * COMPILATION:
 * ============
 * g++ -std=c++17 -Wall -Wextra 02_operators.cpp -o operators
 * ./operators
 *
 * LEARNING CHECKLIST:
 * ==================
 * ☐ Understand all arithmetic operators
 * ☐ Know pre vs post increment
 * ☐ Can use relational operators
 * ☐ Master logical operators and short-circuiting
 * ☐ Understand all bitwise operators (CRITICAL!)
 * ☐ Know operator precedence
 * ☐ Can use bit manipulation for practical problems
 * ☐ Understand GPU relevance of bitwise operations
 *
 * NEXT STEPS:
 * ===========
 * - Move to 03_control_flow.cpp
 * - Practice more bit manipulation problems (LeetCode)
 * - Study how CUDA uses bitwise operations
 * - Learn about __popc() and other CUDA bit intrinsics
 */
