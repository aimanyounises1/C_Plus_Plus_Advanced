# Solutions subdirectory
cmake_minimum_required(VERSION 4.0)
project(practices)
# C++ Examples (always compiled - these have actual code)
add_subdirectory(cpp_examples)

# CUDA Solutions (only if CUDA available)
if(CUDA_AVAILABLE)
    message(STATUS "Building CUDA solutions...")
    add_subdirectory(phase5_cuda)
    add_subdirectory(algorithms)
else()
    message(STATUS "Skipping CUDA solutions (CUDA not available)")
    message(STATUS "See MACOS_SETUP.md or CLOUD_GPU_SETUP.md for cloud GPU options")
endif()

# Note about practice files
message(STATUS "")
message(STATUS "Note: Practice files in /practices/ are templates (TODO placeholders)")
message(STATUS "Fill them in with your implementations, then they can be compiled")
message(STATUS "Working examples are in /solutions/ directory")
