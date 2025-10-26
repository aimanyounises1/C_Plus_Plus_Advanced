/*
 * Exercise: Variables and Data Types
 * Difficulty: Beginner
 * Time: 30-40 minutes
 * Topics: int, float, double, char, bool, string, type conversion, sizeof
 *
 * LEARNING OBJECTIVES:
 * - Understand fundamental C++ data types
 * - Learn type sizes and ranges
 * - Practice type conversion (implicit and explicit)
 * - Use sizeof operator
 *
 * INTERVIEW RELEVANCE:
 * - Understanding data types is fundamental
 * - Type conversion bugs are common interview questions
 * - Memory size awareness is critical for optimization (Nvidia cares about this!)
 */

#include <iostream>
#include <string>
#include <limits>
#include <iomanip>

int main() {
    std::cout << "=== Variables and Data Types Exercises ===\n\n";

    // ========================================================================
    // EXERCISE 1: Basic Variable Declaration (5 min)
    // ========================================================================
    std::cout << "Exercise 1: Basic Variable Declaration\n";
    std::cout << "---------------------------------------\n";

    // TODO 1.1: Declare an integer variable 'age' and set it to 25


    // TODO 1.2: Declare a float variable 'height' in meters (e.g., 1.75)


    // TODO 1.3: Declare a double variable 'pi' with value 3.14159265359


    // TODO 1.4: Declare a char variable 'grade' with value 'A'


    // TODO 1.5: Declare a bool variable 'isStudent' and set it to true


    // TODO 1.6: Declare a string variable 'name' with your name


    // TODO 1.7: Print all variables with descriptive labels
    // Example: std::cout << "Age: " << age << std::endl;



    std::cout << "\n";

    // ========================================================================
    // EXERCISE 2: Data Type Sizes (5 min)
    // ========================================================================
    std::cout << "Exercise 2: Data Type Sizes\n";
    std::cout << "----------------------------\n";

    // TODO 2.1: Use sizeof() to print the size of each data type
    // Example: std::cout << "Size of int: " << sizeof(int) << " bytes\n";

    // Print sizes of: int, long, long long, float, double, char, bool






    std::cout << "\n";

    // ========================================================================
    // EXERCISE 3: Type Ranges (5 min)
    // ========================================================================
    std::cout << "Exercise 3: Type Ranges\n";
    std::cout << "-----------------------\n";

    // TODO 3.1: Use std::numeric_limits to print min and max values
    // Example: std::cout << "int min: " << std::numeric_limits<int>::min() << "\n";

    // Print ranges for: int, unsigned int, long, float, double





    std::cout << "\n";

    // ========================================================================
    // EXERCISE 4: Type Conversion (10 min)
    // ========================================================================
    std::cout << "Exercise 4: Type Conversion\n";
    std::cout << "---------------------------\n";

    // TODO 4.1: Implicit conversion (automatic)
    int intValue = 42;
    // Convert intValue to double and store in doubleValue

    // Print both values


    // TODO 4.2: Explicit conversion (casting)
    double preciseValue = 3.14159;
    // Use static_cast to convert to int and store in truncatedValue

    // Print both - notice data loss!


    // TODO 4.3: Integer division problem
    int numerator = 5;
    int denominator = 2;
    // Calculate result (will be 2, not 2.5!)

    // Now fix it by casting to double

    // Print both results



    // TODO 4.4: Overflow example
    unsigned char smallNumber = 255;
    // Add 1 to it - what happens? (wraps to 0)

    // Print the result


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 5: Const and Auto (5 min)
    // ========================================================================
    std::cout << "Exercise 5: Const and Auto\n";
    std::cout << "--------------------------\n";

    // TODO 5.1: Declare a const variable that cannot be changed


    // TODO 5.2: Try to change it (uncomment next line - should give error)
    // constValue = 100;

    // TODO 5.3: Use auto keyword for type inference
    auto autoInt = 42;           // Type is int
    auto autoDouble = 3.14;      // Type is double
    auto autoString = "Hello";   // Type is const char*

    // Print types using typeid (advanced)
    std::cout << "autoInt type: " << typeid(autoInt).name() << "\n";
    std::cout << "autoDouble type: " << typeid(autoDouble).name() << "\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 6: Practical Application (10 min)
    // ========================================================================
    std::cout << "Exercise 6: Practical Application\n";
    std::cout << "----------------------------------\n";

    // TODO 6.1: Temperature Converter
    // Write code to convert Celsius to Fahrenheit
    // Formula: F = C * 9.0/5.0 + 32.0
    double celsius = 25.0;


    // std::cout << celsius << "°C = " << /* your result */ << "°F\n";

    // TODO 6.2: Calculate BMI (Body Mass Index)
    // Formula: BMI = weight_kg / (height_m * height_m)
    double weightKg = 70.0;
    double heightM = 1.75;


    // std::cout << "BMI: " << /* your result */ << "\n";

    // TODO 6.3: Time conversion
    // Convert total seconds to hours, minutes, seconds
    int totalSeconds = 3665;  // 1 hour, 1 minute, 5 seconds




    std::cout << totalSeconds << " seconds = ";
    // Print in format: "1h 1m 5s"

    std::cout << "\n";

    // ========================================================================
    // CHALLENGE EXERCISES (Optional - 15 min)
    // ========================================================================
    std::cout << "Challenge Exercises\n";
    std::cout << "-------------------\n";

    // CHALLENGE 1: Swap two variables without using a third variable
    // Hint: Use arithmetic or XOR
    int a = 10, b = 20;
    std::cout << "Before swap: a=" << a << ", b=" << b << "\n";

    // Your swap code here (no third variable!)


    std::cout << "After swap: a=" << a << ", b=" << b << "\n";

    // CHALLENGE 2: Check for integer overflow before it happens
    // Given two large integers, check if their sum would overflow
    int max = std::numeric_limits<int>::max();
    int large1 = max - 10;
    int large2 = 20;

    // Check if large1 + large2 would overflow (without actually doing it!)
    // Hint: large1 + large2 > MAX means large1 > MAX - large2



    // CHALLENGE 3: Determine if system is big-endian or little-endian
    // Hint: Store an int, then read it byte by byte



    std::cout << "\n";

    // ========================================================================
    // COMMON INTERVIEW QUESTIONS
    // ========================================================================
    /*
     * Q1: What's the difference between int and long?
     * A: Size and range. int is typically 4 bytes, long is 4 or 8 bytes.
     *    Use sizeof() and std::numeric_limits to check.
     *
     * Q2: What happens when you assign a float to an int?
     * A: Truncation (not rounding!). 3.9 becomes 3, not 4.
     *
     * Q3: Why use double over float?
     * A: Double has more precision (15-17 digits vs 6-7).
     *    Important for scientific computing and financial calculations.
     *
     * Q4: What is type promotion?
     * A: When mixed types are used in operations, smaller types are
     *    promoted to larger types (int + double = double).
     *
     * Q5: What's the difference between signed and unsigned?
     * A: Signed can be negative, unsigned cannot but has larger positive range.
     *    unsigned int: 0 to 4,294,967,295
     *    signed int: -2,147,483,648 to 2,147,483,647
     */

    return 0;
}

/*
 * COMPILATION AND RUNNING:
 * ========================
 * g++ -std=c++17 -Wall -Wextra 01_variables_datatypes.cpp -o variables
 * ./variables
 *
 * EXPECTED OUTPUT:
 * After completing all exercises, you should see:
 * - All variable values printed correctly
 * - Data type sizes (int: 4 bytes, double: 8 bytes, etc.)
 * - Type ranges (int max: 2147483647, etc.)
 * - Conversion examples showing implicit/explicit casting
 * - Practical applications (temperature, BMI, time conversion)
 * - Challenge results (swapped values, overflow detection, endianness)
 *
 * LEARNING CHECKLIST:
 * ☐ Can declare variables of different types
 * ☐ Understand sizeof and type sizes
 * ☐ Know type ranges (min/max values)
 * ☐ Can perform type conversions
 * ☐ Aware of truncation and overflow issues
 * ☐ Use const and auto appropriately
 * ☐ Apply knowledge to practical problems
 *
 * NEXT STEPS:
 * - Move to 02_operators.cpp
 * - Review type conversion carefully (common source of bugs!)
 * - Practice explaining these concepts (interview preparation)
 */
