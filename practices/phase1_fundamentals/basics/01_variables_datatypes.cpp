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
using namespace std;

int main() {
    std::cout << "=== Variables and Data Types Exercises ===\n\n";

    // ========================================================================
    // EXERCISE 1: Basic Variable Declaration (5 min)
    // ========================================================================
    std::cout << "Exercise 1: Basic Variable Declaration\n";
    std::cout << "---------------------------------------\n";

    // TODO 1.1: Declare an integer variable 'age' and set it to 25
    int age = 25;

    // TODO 1.2: Declare a float variable 'height' in meters (e.g., 1.75)
    float height = 1.75;

    // TODO 1.3: Declare a double variable 'pi' with value 3.14159265359
    double pi = 3.14159265359;

    // TODO 1.4: Declare a char variable 'grade' with value 'A'
    char grade = 'A';

    // TODO 1.5: Declare a bool variable 'isStudent' and set it to true
    bool isStudent = true;

    // TODO 1.6: Declare a string variable 'name' with your name
    std::string name = "Aiman";

    // TODO 1.7: Print all variables with descriptive labels
    // Example: std::cout << "Age: " << age << std::endl;
    std::cout << "My age is :" << age << std::endl;

    // ========================================================================
    // EXERCISE 2: Data Type Sizes (5 min)
    // ========================================================================
    std::cout << "Exercise 2: Data Type Sizes\n";
    std::cout << "----------------------------\n";
    // TODO 2.1: Use sizeof() to print the size of each data type
    // Example: std::cout << "Size of int: " << sizeof(int) << " bytes\n";
    std::cout << "integer is : " << sizeof(int) << " bytes" << std::endl;
    // Print sizes of: int, long, long long, float, double, char, bool
    std::cout << "float is : " << sizeof(float) << "bytes" << std::endl;
    std::cout << "long is : " << sizeof(long) << "bytes" << std::endl;
    std::cout << "long long is : " << sizeof(long long) << "bytes" << std::endl;
    std::cout << "double is : " << sizeof(double) << "bytes" << std::endl;
    std::cout << "char is : " << sizeof(char) << "bytes" << std::endl;
    std::cout << "bool is : " << sizeof(bool) << "bytes" << std::endl;


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 3: Type Ranges (5 min)
    // ========================================================================
    std::cout << "Exercise 3: Type Ranges\n";
    std::cout << "-----------------------\n";

    // TODO 3.1: Use std::numeric_limits to print min and max values
    // Example: std::cout << "int min: " << std::numeric_limits<int>::min() << "\n";

    // Print ranges for: int, unsigned int, long, float, double
    std::cout << "The min of each built data type" << std::endl;
    std::cout << "integer min : " << std::numeric_limits<int>::min() << std::endl;
    std::cout << "long min : " << std::numeric_limits<long>::min() << std::endl;
    std::cout << "double min : " << std::numeric_limits<double>::min() << std::endl;
    std::cout << "float min : " << std::numeric_limits<float>::min() << std::endl;

    // Printing the ranges for max  int, long, double, float
    std::cout << "The max of each built in data type" << std::endl;
    std::cout << "double max : " << std::numeric_limits<double>::max() << std::endl;
    std::cout << "float max :" << std::numeric_limits<float>::max() << std::endl;
    std::cout << "long max " << std::numeric_limits<long>::max() << std::endl;
    std::cout << "\n";

    // ========================================================================
    // EXERCISE 4: Type Conversion (10 min)
    // ========================================================================
    std::cout << "Exercise 4: Type Conversion\n";
    std::cout << "---------------------------\n";

    // TODO 4.1: Implicit conversion (automatic)
    int intValue = 42;
    // Convert intValue to double and store in doubleValue
    double int_to_double = intValue;
    // Print both values
    std::cout << "intValue is : " << intValue << std::endl;
    std::cout << "int_to_double after conversion" << int_to_double << std::endl;

    // TODO 4.2: Explicit conversion (casting)
    double preciseValue = 3.14159;
    // Use static_cast to convert to int and store in truncatedValue
    int intPreciseValue = static_cast<int>(preciseValue);
    // Print both - notice data loss!
    std::cout << "int preciseValue : " << intPreciseValue << std::endl;
    std::cout << "double preciseValue" << preciseValue << std::endl;
    // TODO 4.3: Integer division problem
    int numerator = 5;
    int denominator = 2;
    // Calculate result (will be 2, not 2.5!)
    // Now fix it by casting to double
    cout << "numerator/dnominator = " << numerator / denominator << endl;
    // Print both results
    cout << "denominator is = " << denominator << endl;
    cout << "numerator is = " << numerator << endl;

    // TODO 4.4: Overflow example
    unsigned char smallNumber = 255;
    // Add 1 to it - what happens? (wraps to 0)
    smallNumber = +smallNumber;
    // Print the result
    cout << "smallNumber is = " << smallNumber << endl;

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 5: Const and Auto (5 min)
    // ========================================================================
    std::cout << "Exercise 5: Const and Auto\n";
    std::cout << "--------------------------\n";

    // TODO 5.1: Declare a const variable that cannot be changed
    const int constValue = 250;

    // TODO 5.2: Try to change it (uncomment next line - should give error)
    // constValue = 100;

    // TODO 5.3: Use auto keyword for type inference
    auto autoInt = 42; // Type is int
    auto autoDouble = 3.14; // Type is double
    auto autoString = "Hello"; // Type is const char*

    // Print types using typeid (advanced)
    std::cout << "autoInt type: " << typeid(autoInt).name() << "\n";
    std::cout << "autoDouble type: " << typeid(autoDouble).name() << "\n";
    cout << "autoString type: " << typeid(autoString).name() << endl;
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
    double fahrenheit = 25 * 9.0/5.0 + 32.0;

    std::cout << celsius << "°C = " << fahrenheit << "°F\n";

    // TODO 6.2: Calculate BMI (Body Mass Index)
    // Formula: BMI = weight_kg / (height_m * height_m)
    double weightKg = 70.0;
    double heightM = 1.75;
    cout << "BMI =" << weightKg / (heightM * heightM) << endl;

    // std::cout << "BMI: " << /* your result */ << "\n";

    // TODO 6.3: Time conversion
    // Convert total seconds to hours, minutes, seconds
    int totalSeconds = 3665; // 1 hour, 1 minute, 5 seconds


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
    int swp = a;
    a = b;
    b = swp;

    std::cout << "After swap: a=" << a << ", b=" << b << "\n";

    // CHALLENGE 2: Check for integer overflow before it happens
    // Given two large integers, check if their sum would overflow
    int max = std::numeric_limits<int>::max();
    int large1 = max - 10;
    int large2 = 20;
    (max == large1 + large2) ? cout << "overflow" : cout << "no overflow failure" << endl;
    // Check if large1 + large2 would overflow (without actually doing it!)
    // Hint: large1 + large2 > MAX means large1 > MAX - large2


    // CHALLENGE 3: Determine if system is big-endian or little-endian
    // Hint: Store an int, then read it byte by byte


    std::cout << "\n";

    // ========================================================================
    // COMMON INTERVIEW QUESTIONS
    // ========================================================================
    /*
     * Q1: What's the difference between int and long? int is a 4 bytes and long is 8 bytes this can be know from sizeof() function in cpp
     * A: Size and range. int is typically 4 bytes, long is 4 or 8 bytes.
     *    Use sizeof() and std::numeric_limits to check.
     *
     * Q2: What happens when you assign a float to an int?
     *  Answer    >>>  if you assign a decimal number of float to int truncation so it will remove the decimal part
     *
     * A: Truncation (not rounding!). 3.9 becomes 3, not 4.
     *
     * Q3: Why use double over float? because double has 64 digits and can represent the decimal part of number more precisely.
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
