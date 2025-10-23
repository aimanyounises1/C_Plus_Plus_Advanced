# Solutions subdirectory

if(CUDA_AVAILABLE)
    message(STATUS "Building CUDA solutions...")
    add_subdirectory(phase5_cuda)
    add_subdirectory(algorithms)
else()
    message(STATUS "Skipping CUDA solutions (CUDA not available)")
endif()

# Add C++ only solutions if needed
# add_subdirectory(cpp_solutions)
