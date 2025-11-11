#!/bin/bash

# ==================================================================================================
# Test Runner Script
# ==================================================================================================
# This script compiles and runs tests for your C++ exercises
# Usage: ./run_test.sh <exercise_name>
# Example: ./run_test.sh polymorphism
# ==================================================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if exercise name is provided
if [ $# -eq 0 ]; then
    print_error "No exercise name provided"
    echo "Usage: ./run_test.sh <exercise_name>"
    echo ""
    echo "Available exercises:"
    echo "  Basic Exercises:"
    echo "    - polymorphism"
    echo ""
    echo "  Advanced Exercises (from scratch!):"
    echo "    - virtual_functions_advanced"
    echo "    - abstract_classes_advanced"
    echo ""
    echo "Example: ./run_test.sh polymorphism"
    echo "Example: ./run_test.sh virtual_functions_advanced"
    exit 1
fi

EXERCISE=$1
PROJECT_ROOT="/Users/aimanyounis/CLionProjects/C_Plus_Plus_Advanced"

# Navigate to project root
cd "$PROJECT_ROOT"

# Handle different exercises
case $EXERCISE in
    polymorphism)
        print_info "Compiling Polymorphism Exercise..."

        EXERCISE_FILE="exercises/phase2_intermediate/oop_advanced/02_polymorphism_exercises.cpp"
        TEST_FILE="tests/test_polymorphism.cpp"
        OUTPUT="test_polymorphism"

        if [ ! -f "$EXERCISE_FILE" ]; then
            print_error "Exercise file not found: $EXERCISE_FILE"
            exit 1
        fi

        if [ ! -f "$TEST_FILE" ]; then
            print_error "Test file not found: $TEST_FILE"
            exit 1
        fi

        # Compile
        g++ -std=c++17 "$EXERCISE_FILE" "$TEST_FILE" -o "$OUTPUT" 2>&1

        if [ $? -eq 0 ]; then
            print_success "Compilation successful!"
            echo ""
            print_info "Running tests..."
            echo ""
            "./$OUTPUT"
            TEST_RESULT=$?
            echo ""

            if [ $TEST_RESULT -eq 0 ]; then
                print_success "All tests passed! Perfect score!"
            else
                print_warning "Some tests failed. Review the output above and fix your implementations."
            fi
        else
            print_error "Compilation failed. Check the errors above."
            exit 1
        fi
        ;;

    virtual_functions_advanced)
        print_info "Compiling Advanced Virtual Functions Exercise (FROM SCRATCH!)..."

        EXERCISE_FILE="exercises/phase2_intermediate/oop_advanced/03_virtual_functions_advanced.cpp"
        TEST_FILE="tests/test_virtual_functions_advanced.cpp"
        OUTPUT="test_virtual_advanced"

        if [ ! -f "$EXERCISE_FILE" ]; then
            print_error "Exercise file not found: $EXERCISE_FILE"
            exit 1
        fi

        if [ ! -f "$TEST_FILE" ]; then
            print_error "Test file not found: $TEST_FILE"
            exit 1
        fi

        # Compile
        g++ -std=c++17 "$EXERCISE_FILE" "$TEST_FILE" -o "$OUTPUT" 2>&1

        if [ $? -eq 0 ]; then
            print_success "Compilation successful!"
            echo ""
            print_info "Running tests..."
            echo ""
            "./$OUTPUT"
            TEST_RESULT=$?
            echo ""

            if [ $TEST_RESULT -eq 0 ]; then
                print_success "All tests passed! Perfect score!"
            else
                print_warning "Some tests failed. Review the output above and fix your implementations."
            fi
        else
            print_error "Compilation failed. Check the errors above."
            exit 1
        fi
        ;;

    abstract_classes_advanced)
        print_info "Compiling Advanced Abstract Classes Exercise (FROM SCRATCH!)..."

        EXERCISE_FILE="exercises/phase2_intermediate/oop_advanced/04_abstract_classes_advanced.cpp"
        TEST_FILE="tests/test_abstract_classes_advanced.cpp"
        OUTPUT="test_abstract_advanced"

        if [ ! -f "$EXERCISE_FILE" ]; then
            print_error "Exercise file not found: $EXERCISE_FILE"
            exit 1
        fi

        if [ ! -f "$TEST_FILE" ]; then
            print_error "Test file not found: $TEST_FILE"
            exit 1
        fi

        # Compile
        g++ -std=c++17 "$EXERCISE_FILE" "$TEST_FILE" -o "$OUTPUT" 2>&1

        if [ $? -eq 0 ]; then
            print_success "Compilation successful!"
            echo ""
            print_info "Running tests..."
            echo ""
            "./$OUTPUT"
            TEST_RESULT=$?
            echo ""

            if [ $TEST_RESULT -eq 0 ]; then
                print_success "All tests passed! Perfect score!"
            else
                print_warning "Some tests failed. Review the output above and fix your implementations."
            fi
        else
            print_error "Compilation failed. Check the errors above."
            exit 1
        fi
        ;;

    *)
        print_error "Unknown exercise: $EXERCISE"
        echo ""
        echo "Available exercises:"
        echo "  Basic:"
        echo "    - polymorphism"
        echo ""
        echo "  Advanced (from scratch!):"
        echo "    - virtual_functions_advanced"
        echo "    - abstract_classes_advanced"
        exit 1
        ;;
esac

print_info "Test run complete."