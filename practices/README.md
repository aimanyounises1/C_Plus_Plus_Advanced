# C++ to Nvidia Practice Files

This directory contains organized practice files following the [Nvidia C++ Roadmap](../NVIDIA_CPP_ROADMAP.md).

## Directory Structure

### Phase 1: Fundamentals (2-3 months)
```
phase1_fundamentals/
├── basics/              # Basic syntax, control flow, functions
├── pointers_memory/     # Pointers, references, dynamic memory
├── functions/           # Function overloading, recursion, headers
└── data_structures/     # Structures, enums, file I/O
```

### Phase 2: Intermediate (3-4 months)
```
phase2_intermediate/
├── oop_basics/          # Classes, objects, constructors
├── oop_advanced/        # Inheritance, polymorphism, virtual functions
├── stl/                 # Containers, iterators, algorithms, lambdas
└── memory_management/   # RAII, smart pointers, move semantics
```

### Phase 3: Advanced (3-4 months)
```
phase3_advanced/
├── templates/           # Function/class templates, specialization
├── modern_cpp/          # C++11/14/17/20 features
├── concurrency/         # Threads, mutexes, atomics, async
└── design_patterns/     # Common design patterns
```

### Phase 4: Performance & Optimization (2-3 months)
```
phase4_performance/
├── profiling/           # Benchmarking, profiling tools
├── optimization/        # Compiler opts, loop opts, branch prediction
├── simd/                # SIMD intrinsics and vectorization
└── cache_optimization/  # Cache-friendly code, alignment, prefetching
```

### Phase 5: GPU Programming & CUDA (3-6 months) ⭐ CRITICAL FOR NVIDIA
```
phase5_cuda/
├── gpu_basics/          # GPU architecture fundamentals
├── cuda_basics/         # Basic CUDA programming
├── cuda_advanced/       # Shared memory, reductions, warp primitives
├── cuda_optimization/   # Profiling, occupancy, multi-GPU
└── cuda_libraries/      # cuBLAS, cuDNN, Thrust, TensorRT
```

### Phase 6: Specialized Topics (2-4 months)
```
phase6_specialized/
├── graphics/            # OpenGL, Vulkan, shaders, ray tracing
├── deep_learning/       # Neural networks, ML optimization
├── systems_programming/ # Linux system calls, DMA, PCIe
└── software_engineering/# CMake, testing, debugging, docs
```

### Phase 7: Projects & Portfolio
```
phase7_projects/
├── cuda_raytracer/      # CUDA-accelerated ray tracer
├── physics_simulation/  # GPU particle physics
├── ml_inference_engine/ # Neural network inference
├── custom_allocator/    # High-performance allocator
├── data_structures/     # Custom implementations
└── mini_projects/       # Smaller focused projects
```

## How to Use This Repository

### 1. Follow the Roadmap
Start with Phase 1 and progress sequentially. Each phase builds on previous knowledge.

### 2. Practice Files
Each `.cpp` or `.cu` file contains:
- **Comment header** describing the topic
- **TODO** section explaining what to practice
- Empty implementation for you to fill

### 3. Compilation

**For C++ files:**
```bash
g++ -std=c++20 -O2 filename.cpp -o output
```

**For CUDA files:**
```bash
nvcc -std=c++17 -O2 filename.cu -o output
```

### 4. Project Work
Major projects in `phase7_projects/` have their own README files with:
- Project overview
- Goals and requirements
- Key files to create
- Learning outcomes

### 5. Track Your Progress
- Complete files in order within each phase
- Comment your code thoroughly
- Write tests for your implementations
- Compare your solutions with standard library implementations

## Recommended Practice Schedule

### Daily Practice (Minimum)
- 1-2 hours of focused practice
- Complete 1-2 practice files per day
- Review and refactor previous code

### Weekly Goals
- Complete 1 topic area (e.g., all OOP basics files)
- Write at least 1 test suite
- Profile and optimize 1 piece of code

### Monthly Milestones
- Complete 1 phase per month (adjust based on complexity)
- Complete at least 1 mini-project
- Review and consolidate knowledge

## Important Notes

### CUDA Development
- Phase 5 (CUDA) is **CRITICAL** for Nvidia positions
- Spend extra time on CUDA optimization
- Build impressive CUDA projects for your portfolio
- Study Nsight profiling tools thoroughly

### Performance Focus
- Always benchmark your implementations
- Compare against standard libraries
- Profile before and after optimizations
- Document performance improvements

### Portfolio Building
- Keep your best implementations
- Write clear documentation
- Add performance benchmarks
- Create demo videos/screenshots

## Resources

### Compilation Tools
- **g++/clang++**: C++ compilers
- **nvcc**: CUDA compiler
- **CMake**: Build system

### Development Tools
- **VSCode/CLion**: IDEs
- **gdb/lldb**: Debuggers
- **cuda-gdb**: CUDA debugger
- **Nsight Compute/Systems**: CUDA profilers

### Learning Resources
- See main roadmap for books and courses
- Nvidia CUDA samples: `/usr/local/cuda/samples`
- cppreference.com for C++ reference
- CUDA Programming Guide (official docs)

## Tips for Success

1. **Code Daily**: Consistency is key
2. **Understand, Don't Memorize**: Focus on concepts
3. **Write Tests**: Verify your implementations
4. **Optimize**: Always look for performance improvements
5. **Document**: Explain your design decisions
6. **Build Projects**: Theory + Practice = Mastery
7. **Profile Everything**: Measure, don't guess
8. **Focus on CUDA**: It's essential for Nvidia

## Progression Path

```
Phase 1 (Fundamentals) → Phase 2 (Intermediate) → Phase 3 (Advanced)
                                    ↓
Phase 4 (Performance) ← Phase 5 (CUDA) ⭐ ← Phase 6 (Specialized)
                                    ↓
                        Phase 7 (Projects & Portfolio)
```

## Getting Help

- Read compiler error messages carefully
- Use debuggers to understand behavior
- Consult documentation
- Study example implementations
- Join C++ and CUDA communities

---

**Remember**: The goal isn't just to complete files, but to deeply understand the concepts and build production-quality code. Take your time, practice deliberately, and focus on mastery.

Good luck on your journey to Nvidia!
