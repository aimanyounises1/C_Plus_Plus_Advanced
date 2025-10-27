/*
 * ==================================================================================================
 * Exercise: std::optional, std::variant, std::any (C++17)
 * ==================================================================================================
 * Difficulty: Intermediate/Advanced | Time: 45-55 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master std::optional for nullable values
 * 2. Understand std::variant for type-safe unions
 * 3. Learn std::any for type erasure
 * 4. Practice modern vocabulary types
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - Optional for GPU query results
 * - Variant for polymorphic GPU data
 * - Safe error handling
 * - Modern C++ idioms
 * ==================================================================================================
 */

#include <iostream>
#include <optional>
#include <variant>
#include <any>
#include <string>
using namespace std;

/*
 * EXERCISE 1: std::optional (15 min)
 */

// Function that may not return a value
optional<int> findIndex(const string& str, char ch) {
    for (size_t i = 0; i < str.length(); i++) {
        if (str[i] == ch) return i;
    }
    return nullopt;  // or {}
}

void optionalExample() {
    auto result = findIndex("hello", 'l');

    if (result) {  // or result.has_value()
        cout << "Found at index: " << *result << endl;  // or result.value()
    } else {
        cout << "Not found" << endl;
    }

    // value_or provides default
    cout << "Index: " << findIndex("hello", 'x').value_or(-1) << endl;
}

/*
 * EXERCISE 2: std::variant (15 min)
 */

void variantExample() {
    variant<int, double, string> data;

    data = 42;
    cout << "int: " << get<int>(data) << endl;

    data = 3.14;
    cout << "double: " << get<double>(data) << endl;

    data = "hello";
    cout << "string: " << get<string>(data) << endl;

    // Check which type is active
    if (holds_alternative<string>(data)) {
        cout << "Currently holds string" << endl;
    }

    // Visit pattern
    visit([](auto&& arg) {
        using T = decay_t<decltype(arg)>;
        if constexpr (is_same_v<T, int>) cout << "int: " << arg << endl;
        else if constexpr (is_same_v<T, double>) cout << "double: " << arg << endl;
        else cout << "string: " << arg << endl;
    }, data);
}

/*
 * EXERCISE 3: std::any (10 min)
 */

void anyExample() {
    any value;

    value = 42;
    cout << "int: " << any_cast<int>(value) << endl;

    value = 3.14;
    cout << "double: " << any_cast<double>(value) << endl;

    value = string("hello");
    cout << "string: " << any_cast<string>(value) << endl;

    // Check type
    if (value.type() == typeid(string)) {
        cout << "Currently holds string" << endl;
    }
}

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is std::optional?
 * A: Represents value that may or may not exist (like nullable, but type-safe)
 *
 * Q2: optional vs pointer?
 * A: optional: Value semantics, no heap, explicit "no value"
 *    pointer: Can be null, but less clear intent
 *
 * Q3: What is std::variant?
 * A: Type-safe union - holds one of several types at compile time
 *
 * Q4: variant vs union?
 * A: variant: Type-safe, RAII, knows active type
 *    union: Unsafe, manual tracking
 *
 * Q5: What is std::any?
 * A: Type erasure - holds any type, checked at runtime
 *
 * Q6: When to use each?
 * A: optional: Nullable values
 *    variant: Fixed set of types
 *    any: Dynamic typing (rare, prefer variant)
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - Optional for device query results
 * - Variant for multi-architecture kernels
 * - Error handling without exceptions
 *
 * COMPILATION: g++ -std=c++17 03_optional_variant_any.cpp -o vocab
 * ==================================================================================================
 */

int main() {
    cout << "=== Modern Vocabulary Types Practice ===" << endl;

    optionalExample();
    variantExample();
    anyExample();

    return 0;
}
