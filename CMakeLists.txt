cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(C_Plus_Plus_Advanced LANGUAGES CXX)

# Set C++ standard to C++20 for modern features (ranges, coroutines, etc.)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Check for CUDA support
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # CUDA flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

    # Set architecture (adjust based on your GPU)
    # Common options: sm_70 (V100), sm_75 (RTX 2000), sm_80 (A100), sm_86 (RTX 3000)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    endif()

    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    set(CUDA_AVAILABLE TRUE)
else()
    message(STATUS "CUDA not found - CUDA examples will be skipped")
    message(STATUS "To compile CUDA code:")
    message(STATUS "  - Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
    message(STATUS "  - Or use cloud GPU: see CLOUD_GPU_SETUP.md")
    set(CUDA_AVAILABLE FALSE)
endif()

# Platform-specific settings
if(APPLE)
    message(STATUS "macOS detected")
    message(STATUS "Note: CUDA is not supported on macOS")
    message(STATUS "See MACOS_SETUP.md for cloud GPU options")
endif()

# Find threading library
find_package(Threads REQUIRED)

# Add subdirectories
add_subdirectory(solutions)

# Uncomment the line below AFTER filling in practice template files
# See practices/README_PRACTICES.md for instructions
# add_subdirectory(practices)

# Print configuration summary
message(STATUS "")
message(STATUS "========================================")
message(STATUS "Configuration Summary")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Available: ${CUDA_AVAILABLE}")
if(CUDA_AVAILABLE)
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
endif()
message(STATUS "========================================")
message(STATUS "")

# Print build instructions
message(STATUS "Build Instructions:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make -j$(nproc)")
message(STATUS "")
