# CUDA Solutions

# Vector Addition with optimizations
add_executable(vector_add 01_vector_addition_optimized.cu)
set_target_properties(vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Matrix Multiplication with optimizations
add_executable(matmul 02_matrix_multiplication_optimized.cu)
set_target_properties(matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Parallel Reduction
add_executable(reduction 03_parallel_reduction.cu)
set_target_properties(reduction PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Installation (optional)
install(TARGETS vector_add matmul reduction
    RUNTIME DESTINATION bin
)

message(STATUS "CUDA examples configured:")
message(STATUS "  - vector_add: Vector addition with 3 optimization levels")
message(STATUS "  - matmul: Matrix multiplication with 4 optimization levels")
message(STATUS "  - reduction: Parallel reduction with 6 implementations")
