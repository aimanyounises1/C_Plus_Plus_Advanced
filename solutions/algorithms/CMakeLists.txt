# GPU Algorithm Implementations

# Two Sum GPU implementation
add_executable(two_sum_gpu 01_two_sum_gpu.cu)
set_target_properties(two_sum_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Add more algorithm implementations as they're created
# add_executable(parallel_sort parallel_sort.cu)
# add_executable(bfs_gpu bfs_gpu.cu)

message(STATUS "GPU algorithm examples configured:")
message(STATUS "  - two_sum_gpu: Two Sum with CPU vs GPU comparison")
