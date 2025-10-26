/*
 * Exercise: Enumerations
 * Difficulty: Beginner
 * Time: 30-40 minutes
 * Topics: enum, enum class, type safety, underlying types
 *
 * LEARNING OBJECTIVES:
 * - Understand traditional enums (C-style)
 * - Master enum class (C++11 scoped enums)
 * - Learn underlying types and type safety
 * - Practice using enums in switch statements
 * - Understand when to use enums vs constants
 *
 * INTERVIEW RELEVANCE:
 * - Enums are used for state machines, flags, error codes
 * - Understanding scoped vs unscoped is important
 * - Enum class prevents common bugs
 * - Used extensively in APIs and configuration
 * - GPU kernels use enums for kernel launch configuration
 */

#include <iostream>
#include <string>

int main() {
    std::cout << "=== Enumerations Exercises ===\n\n";

    // ========================================================================
    // EXERCISE 1: Traditional Enums (C-style) (5 min)
    // ========================================================================
    std::cout << "Exercise 1: Traditional Enums\n";
    std::cout << "------------------------------\n";

    // TODO 1.1: Define an enum for days of the week
    // enum DayOfWeek {
    //     MONDAY,    // 0
    //     TUESDAY,   // 1
    //     WEDNESDAY, // 2
    //     THURSDAY,  // 3
    //     FRIDAY,    // 4
    //     SATURDAY,  // 5
    //     SUNDAY     // 6
    // };


    // TODO 1.2: Create a variable and assign a value
    // DayOfWeek today = WEDNESDAY;


    // TODO 1.3: Print the value (it's just an integer!)
    // std::cout << "Today is day: " << today << "\n";


    // TODO 1.4: Enum with explicit values
    // enum ErrorCode {
    //     SUCCESS = 0,
    //     FILE_NOT_FOUND = 404,
    //     ACCESS_DENIED = 403,
    //     INTERNAL_ERROR = 500
    // };


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 2: Problems with Traditional Enums (5 min)
    // ========================================================================
    std::cout << "Exercise 2: Traditional Enum Problems\n";
    std::cout << "--------------------------------------\n";

    // TODO 2.1: Name collision (enums pollute namespace)
    // enum Color { RED, GREEN, BLUE };
    // enum TrafficLight { RED, YELLOW, GREEN };  // ERROR! RED and GREEN already defined


    // TODO 2.2: Implicit conversion to int (not type-safe)
    // Color c = RED;
    // int x = c;           // OK (should this be allowed?)
    // c = 42;              // ERROR (good!)
    // c = Color(42);       // OK (bad! no range checking)


    // TODO 2.3: Can't forward declare traditional enums easily


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 3: Enum Class (C++11 Scoped Enums) (10 min)
    // ========================================================================
    std::cout << "Exercise 3: Enum Class\n";
    std::cout << "----------------------\n";

    // TODO 3.1: Define an enum class
    enum class Color {
        Red,
        Green,
        Blue
    };

    enum class TrafficLight {
        Red,     // No collision! Scoped to TrafficLight
        Yellow,
        Green
    };

    // TODO 3.2: Use enum class values (must use scope)
    Color c = Color::Red;
    TrafficLight light = TrafficLight::Green;

    std::cout << "Color: Red = " << static_cast<int>(c) << "\n";
    std::cout << "Light: Green = " << static_cast<int>(light) << "\n";

    // TODO 3.3: No implicit conversion to int (type safe!)
    // int x = c;              // ERROR! Can't convert
    // int x = static_cast<int>(c);  // OK, explicit


    // TODO 3.4: Can't compare different enum classes
    // if (c == light) { }    // ERROR! Different types


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 4: Underlying Types (5 min)
    // ========================================================================
    std::cout << "Exercise 4: Underlying Types\n";
    std::cout << "----------------------------\n";

    // TODO 4.1: Specify underlying type (default is int)
    enum class Status : char {  // Uses char instead of int
        Idle = 'I',
        Running = 'R',
        Stopped = 'S'
    };

    enum class LargeEnum : uint64_t {
        First = 0,
        Last = 10000000000ULL  // Needs more than 32 bits
    };

    std::cout << "sizeof(Status): " << sizeof(Status) << " bytes\n";
    std::cout << "sizeof(LargeEnum): " << sizeof(LargeEnum) << " bytes\n";

    // TODO 4.2: Why specify underlying type?
    /*
     * Reasons to specify:
     * - Control memory usage (use smaller types)
     * - Ensure specific size for serialization/network protocols
     * - Interface with C code or hardware registers
     * - Forward declaration (must know size)
     */

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 5: Enums in Switch Statements (10 min)
    // ========================================================================
    std::cout << "Exercise 5: Switch Statements\n";
    std::cout << "------------------------------\n";

    // TODO 5.1: Use enum in switch
    Color myColor = Color::Blue;

    switch (myColor) {
        case Color::Red:
            std::cout << "Color is Red\n";
            break;
        case Color::Green:
            std::cout << "Color is Green\n";
            break;
        case Color::Blue:
            std::cout << "Color is Blue\n";
            break;
        // No default needed if all cases covered
    }

    // TODO 5.2: Switch without default (compiler warning if cases missing)
    // Good practice: let compiler warn about missing cases


    // TODO 5.3: Traditional enum vs enum class in switch
    // Traditional enum: can use just name (RED)
    // Enum class: must use scope (Color::Red)


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 6: Practical Applications (10 min)
    // ========================================================================
    std::cout << "Exercise 6: Practical Applications\n";
    std::cout << "-----------------------------------\n";

    // TODO 6.1: State machine using enum
    enum class GameState {
        MainMenu,
        Playing,
        Paused,
        GameOver
    };

    GameState state = GameState::MainMenu;

    // Simulate state transitions
    std::cout << "Starting game...\n";
    state = GameState::Playing;

    switch (state) {
        case GameState::MainMenu:
            std::cout << "Show main menu\n";
            break;
        case GameState::Playing:
            std::cout << "Game is running\n";
            break;
        case GameState::Paused:
            std::cout << "Game is paused\n";
            break;
        case GameState::GameOver:
            std::cout << "Game over!\n";
            break;
    }

    // TODO 6.2: Error codes
    enum class ResultCode {
        Success,
        InvalidInput,
        OutOfMemory,
        FileNotFound,
        PermissionDenied
    };


    // TODO 6.3: GPU kernel configuration
    enum class MemoryType {
        Global,
        Shared,
        Constant,
        Texture
    };


    // TODO 6.4: Flags (bit flags using enum)
    enum class FilePermissions : unsigned int {
        None = 0,
        Read = 1 << 0,   // 0001
        Write = 1 << 1,  // 0010
        Execute = 1 << 2 // 0100
    };

    // Combine flags using bitwise OR
    unsigned int perms = static_cast<unsigned int>(FilePermissions::Read) |
                         static_cast<unsigned int>(FilePermissions::Write);

    std::cout << "Permissions: " << perms << " (Read + Write)\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 7: Helper Functions for Enums (5 min)
    // ========================================================================
    std::cout << "Exercise 7: Helper Functions\n";
    std::cout << "----------------------------\n";

    // TODO 7.1: Convert enum to string
    // std::string toString(Color c) {
    //     switch (c) {
    //         case Color::Red: return "Red";
    //         case Color::Green: return "Green";
    //         case Color::Blue: return "Blue";
    //         default: return "Unknown";
    //     }
    // }


    // TODO 7.2: Convert string to enum
    // Color fromString(const std::string& str) {
    //     if (str == "Red") return Color::Red;
    //     if (str == "Green") return Color::Green;
    //     if (str == "Blue") return Color::Blue;
    //     return Color::Red;  // Default
    // }


    // TODO 7.3: Increment enum (for iteration)
    // Color& operator++(Color& c) {
    //     c = static_cast<Color>(static_cast<int>(c) + 1);
    //     return c;
    // }


    std::cout << "\n";

    // ========================================================================
    // CHALLENGE EXERCISES (Optional - 10 min)
    // ========================================================================
    std::cout << "Challenge Exercises\n";
    std::cout << "-------------------\n";

    // CHALLENGE 1: Implement a type-safe flags system using enum class
    // Overload operators |, &, ~ for bitwise operations


    // CHALLENGE 2: Create an enum for all CUDA error codes
    // enum class CudaError {
    //     Success,
    //     InvalidValue,
    //     OutOfMemory,
    //     ...
    // };


    // CHALLENGE 3: Implement a finite state machine
    // Track state transitions and validate them


    // CHALLENGE 4: Create an enum with reflection
    // Generate string names automatically (use macro or template)


    std::cout << "\n";

    return 0;
}

// ========================================================================
// COMMON INTERVIEW QUESTIONS
// ========================================================================
/*
 * Q1: What's the difference between enum and enum class?
 * A: enum (unscoped):
 *    - Names are in enclosing scope (can cause name collisions)
 *    - Implicitly converts to int
 *    - Can't forward declare without underlying type
 *    - Example: enum Color { RED, GREEN };
 *    - Use: Color c = RED;
 *
 *    enum class (scoped):
 *    - Names are scoped to enum (no collisions)
 *    - No implicit conversion to int (type safe)
 *    - Can forward declare
 *    - Example: enum class Color { Red, Green };
 *    - Use: Color c = Color::Red;
 *
 *    Prefer enum class for new code!
 *
 * Q2: When should you use enums vs constants?
 * A: Use enums when:
 *    - Values are mutually exclusive (only one at a time)
 *    - Represent a fixed set of options
 *    - Want type safety
 *    - Need switch statement exhaustiveness checking
 *
 *    Use constants when:
 *    - Independent values
 *    - Mathematical constants (pi, e)
 *    - Configuration values
 *    - May need different types
 *
 * Q3: How do you specify the underlying type of an enum?
 * A: enum class MyEnum : uint8_t {  // Use uint8_t instead of default int
 *        Value1,
 *        Value2
 *    };
 *
 *    Why specify:
 *    - Control size (memory optimization)
 *    - Serialization (know exact byte size)
 *    - Interface with hardware/network protocols
 *    - Forward declaration (compiler needs to know size)
 *
 * Q4: Can you forward declare an enum?
 * A: Traditional enum: Only if you specify underlying type
 *    enum Color : int;  // Forward declaration
 *    enum Color : int { RED, GREEN, BLUE };  // Definition
 *
 *    Enum class: Can forward declare (underlying type is int by default)
 *    enum class Color;  // Forward declaration
 *    enum class Color { Red, Green, Blue };  // Definition
 *
 * Q5: How do you implement bit flags with enums?
 * A: Use powers of 2 (bit positions):
 *
 *    enum class Flags : unsigned int {
 *        None = 0,
 *        Flag1 = 1 << 0,  // 0001
 *        Flag2 = 1 << 1,  // 0010
 *        Flag3 = 1 << 2,  // 0100
 *        Flag4 = 1 << 3   // 1000
 *    };
 *
 *    Combine: flags = Flag1 | Flag2;
 *    Check: if (flags & Flag1)
 *    Toggle: flags ^= Flag1;
 *
 * Q6: What are the advantages of enum class over #define?
 * A: enum class advantages:
 *    - Type safety (can't mix different enums)
 *    - Scoped names (no global pollution)
 *    - Compiler can check exhaustiveness in switch
 *    - Better debugging (debugger shows names)
 *    - Part of the type system
 *
 *    #define disadvantages:
 *    - Just text replacement
 *    - No type checking
 *    - Global scope
 *    - Can't use in switch for exhaustiveness checking
 *
 * Q7: How do you iterate over an enum?
 * A: Enums don't have built-in iteration. You must:
 *
 *    1. Define First and Last sentinels:
 *       enum class Color { First, Red = First, Green, Blue, Last = Blue };
 *
 *    2. Cast and increment:
 *       for (int i = static_cast<int>(Color::First);
 *            i <= static_cast<int>(Color::Last);
 *            i++) {
 *           Color c = static_cast<Color>(i);
 *       }
 *
 *    3. Or use an array of values:
 *       constexpr Color all_colors[] = {Color::Red, Color::Green, Color::Blue};
 *
 * Q8: Can you have methods in an enum?
 * A: No, enums can't have member functions in C++.
 *
 *    Workarounds:
 *    - Free functions that take enum as parameter
 *    - Namespace with functions and enum
 *    - Use a class with static constants instead
 *
 *    Example:
 *    namespace Color {
 *        enum class Type { Red, Green, Blue };
 *        std::string toString(Type c) { ... }
 *    }
 */

/*
 * ENUMS IN GPU PROGRAMMING:
 * ==========================
 *
 * 1. Kernel Launch Configuration:
 *    enum class BlockSize : int {
 *        Size128 = 128,
 *        Size256 = 256,
 *        Size512 = 512
 *    };
 *
 *    kernel<<<numBlocks, static_cast<int>(BlockSize::Size256)>>>();
 *
 * 2. Memory Type Selection:
 *    enum class MemSpace {
 *        Global,
 *        Shared,
 *        Constant
 *    };
 *
 * 3. Error Handling:
 *    enum class CudaStatus {
 *        Success,
 *        MemoryError,
 *        InvalidValue,
 *        LaunchFailure
 *    };
 *
 * 4. Compute Capability:
 *    enum class ComputeCapability {
 *        CC_70 = 70,  // Volta
 *        CC_75 = 75,  // Turing
 *        CC_80 = 80,  // Ampere
 *        CC_86 = 86,  // Ampere
 *        CC_89 = 89   // Lovelace
 *    };
 *
 * 5. Optimization Hints:
 *    enum class CachePreference {
 *        PreferNone,
 *        PreferShared,
 *        PreferL1,
 *        PreferEqual
 *    };
 *
 *    cudaDeviceSetCacheConfig(
 *        static_cast<cudaFuncCache>(CachePreference::PreferShared)
 *    );
 *
 * COMPILATION:
 * ============
 * g++ -std=c++17 -Wall -Wextra 02_enumerations.cpp -o enums
 * ./enums
 *
 * LEARNING CHECKLIST:
 * ===================
 * ☐ Understand traditional enums and their limitations
 * ☐ Master enum class (scoped enums)
 * ☐ Know how to specify underlying types
 * ☐ Can use enums in switch statements
 * ☐ Understand enum class type safety
 * ☐ Know when to use enums vs constants
 * ☐ Can implement bit flags with enums
 * ☐ Understand scoping rules
 *
 * NEXT STEPS:
 * ===========
 * - Move to 03_file_io.cpp
 * - Study enum class best practices
 * - Learn about std::variant (type-safe union)
 * - Explore enum reflection libraries
 * - Understand bit manipulation for flags
 */
