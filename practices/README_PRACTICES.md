# Practice Files Guide

## ⚠️ Important: These Are Templates!

All files in this `practices/` directory are **empty templates** - they contain only TODO comments, not actual code.

### What You See:
```cpp
// Practice: Variables and Data Types
// Topics: int, float, double, char, bool, type conversion
// TODO: Practice declaring and using different data types
```

### What You Need to Do:
Replace the TODO with actual working code:
```cpp
// Practice: Variables and Data Types
// Topics: int, float, double, char, bool, type conversion

#include <iostream>

int main() {
    // Practice declaring variables
    int age = 25;
    double height = 5.9;
    char grade = 'A';
    bool isPassing = true;

    std::cout << "Age: " << age << std::endl;
    std::cout << "Height: " << height << std::endl;
    std::cout << "Grade: " << grade << std::endl;
    std::cout << "Passing: " << isPassing << std::endl;

    return 0;
}
```

---

## How to Use These Practice Files

### Step 1: Choose a Phase

Start with Phase 1 (fundamentals) if you're learning C++, or jump to Phase 5 (CUDA) if you know C++ already.

**Recommended order:**
1. Phase 1: C++ Fundamentals
2. Phase 2: Intermediate C++
3. Phase 3: Advanced C++
4. Phase 4: Performance
5. Phase 5: CUDA (Critical for Nvidia interviews!)
6. Phase 6: Specialized topics
7. Phase 7: Projects

### Step 2: Fill In the Code

Open a practice file and implement what the TODO describes:

```bash
# Open in your favorite editor
code practices/phase1_fundamentals/basics/01_variables_datatypes.cpp

# Or use vim
vim practices/phase1_fundamentals/basics/01_variables_datatypes.cpp
```

### Step 3: Compile and Test

**Option A: Manual compilation**
```bash
# For C++ files
g++ -std=c++17 -O2 practices/phase1_fundamentals/basics/01_variables_datatypes.cpp -o test
./test

# For CUDA files (requires CUDA Toolkit)
nvcc -std=c++17 -O3 practices/phase5_cuda/cuda_basics/01_hello_cuda.cu -o test
./test
```

**Option B: Using CMake** (After filling in files)
```bash
# 1. Rename the template
cd practices
mv CMakeLists.txt.template CMakeLists.txt

# 2. Edit CMakeLists.txt and uncomment the sections you want to build

# 3. Edit root CMakeLists.txt and uncomment:
# add_subdirectory(practices)

# 4. Rebuild
cd ..
rm -rf build && mkdir build && cd build
cmake ..
make
```

---

## Working Examples

Instead of empty templates, you can look at **working examples** in the `solutions/` directory:

```
solutions/
├── cpp_examples/
│   └── 01_hello_world.cpp  ← Working C++ example
├── phase5_cuda/
│   ├── 01_vector_addition_optimized.cu  ← Production-quality CUDA
│   ├── 02_matrix_multiplication_optimized.cu
│   └── 03_parallel_reduction.cu
├── algorithms/
│   └── 01_two_sum_gpu.cu  ← Algorithm implementation
└── interview_prep/
    └── NVIDIA_TECHNICAL_QUESTIONS.md  ← Interview Q&As
```

**Study these first!** They show proper implementations.

---

## Practice File Organization

### Phase 1: Fundamentals (15 files)
```
practices/phase1_fundamentals/
├── basics/ (5 files)
│   ├── 01_variables_datatypes.cpp
│   ├── 02_operators.cpp
│   ├── 03_control_flow.cpp
│   ├── 04_functions.cpp
│   └── 05_arrays_strings.cpp
├── pointers_memory/ (4 files)
├── functions/ (3 files)
└── data_structures/ (3 files)
```

### Phase 2: Intermediate (19 files)
- OOP basics (5 files)
- OOP advanced (5 files)
- STL (5 files)
- Memory management (4 files)

### Phase 3: Advanced (19 files)
- Templates (5 files)
- Modern C++ (5 files)
- Concurrency (5 files)
- Design patterns (4 files)

### Phase 4: Performance (14 files)
- Profiling (3 files)
- Optimization (4 files)
- Cache optimization (4 files)
- SIMD (3 files)

### Phase 5: CUDA (31 files) ⭐ CRITICAL FOR NVIDIA
- GPU basics (3 files)
- CUDA basics (7 files)
- CUDA advanced (8 files)
- CUDA optimization (7 files)
- CUDA libraries (6 files)

### Phase 6: Specialized (18 files)
- Graphics (5 files)
- Deep learning (5 files)
- Systems programming (4 files)
- Software engineering (4 files)

### Phase 7: Projects (8 mini + 5 major)
- Mini projects (8 .cpp files)
- Major projects (5 README specs)

---

## Recommended Learning Approach

### Approach 1: Study Solutions First (Recommended for Interviews)

1. **Start with working examples:**
   ```bash
   # Study these production-quality implementations
   cat solutions/phase5_cuda/01_vector_addition_optimized.cu
   cat solutions/phase5_cuda/02_matrix_multiplication_optimized.cu
   cat solutions/phase5_cuda/03_parallel_reduction.cu
   ```

2. **Understand optimizations:** See how each version improves performance

3. **Practice by modifying:** Change parameters, add features, experiment

4. **Build your own:** Use solutions as references for your projects

### Approach 2: Fill In Templates (Good for Learning)

1. **Choose a topic:** Start with what you need to learn

2. **Implement from scratch:** Don't look at solutions immediately

3. **Compare:** Check your implementation against solutions

4. **Iterate:** Improve based on what you learned

### Approach 3: Hybrid (Best for Time-Constrained)

1. **Phase 1-4:** Fill in templates for C++ practice

2. **Phase 5 (CUDA):** Study solutions directly (most important for Nvidia!)

3. **Phase 6-7:** Selective - only topics relevant to your role

---

## For Nvidia Interview Prep

### What to Focus On:

**High Priority (Must Know):**
- ✅ Solutions in `phase5_cuda/` - Study these thoroughly
- ✅ Interview questions in `interview_prep/`
- ✅ GPU architecture concepts

**Medium Priority:**
- C++ fundamentals (Phase 1-2)
- Memory management (Phase 2)
- Concurrency (Phase 3)
- Performance optimization (Phase 4)

**Lower Priority (Nice to Have):**
- Graphics (Phase 6) - unless applying for graphics role
- Systems programming (Phase 6)
- Mini projects (Phase 7)

### Timeline (4-6 Weeks):

**Week 1-2:** Phase 5 solutions + interview questions
**Week 3-4:** Algorithm problems + GPU architecture
**Week 5:** Build 1-2 projects from Phase 7
**Week 6:** Mock interviews + polish portfolio

---

## Common Questions

### Q: "Do I need to fill in all 124 template files?"
**A:** No! Focus on working solutions in `/solutions/` directory. Templates are optional learning exercises.

### Q: "Can I compile the templates as-is?"
**A:** No, they're just TODOs. You must add actual code first.

### Q: "Which files should I prioritize?"
**A:** Study `/solutions/phase5_cuda/` files - these are production-quality CUDA examples showing optimizations. Most important for Nvidia interviews!

### Q: "How do I add CMake for my filled-in practices?"
**A:** See instructions at top of `CMakeLists.txt.template` in this directory.

### Q: "Where are the answers to the templates?"
**A:** The `/solutions/` directory contains production-quality implementations. Use these as references.

---

## Build Status

### Currently Configured (Working):
✅ `solutions/cpp_examples/hello_world` - C++ example
✅ `solutions/phase5_cuda/*` - CUDA optimizations (if CUDA available)
✅ `solutions/algorithms/*` - GPU algorithms (if CUDA available)

### Not Configured (Templates):
⚠️ All files in `practices/` - Empty templates, no code

### To Enable Practices:
1. Fill in template files with actual code
2. Rename `CMakeLists.txt.template` → `CMakeLists.txt`
3. Uncomment sections in `practices/CMakeLists.txt`
4. Uncomment `add_subdirectory(practices)` in root `CMakeLists.txt`
5. Rebuild with CMake

---

## Learning Resources

- **Working examples:** `/solutions/` directory
- **Interview prep:** `/solutions/interview_prep/`
- **Roadmap:** `/NVIDIA_CPP_ROADMAP.md`
- **Resources:** `/LEARNING_RESOURCES.md`
- **Mac setup:** `/MACOS_SETUP.md`
- **Cloud GPU:** `/CLOUD_GPU_SETUP.md`
- **CLion IDE:** `/CLION_SETUP.md`

---

## Summary

**Practice files** = Empty TODO templates for learning
**Solution files** = Production-quality working code

**For interview prep:** Study solutions first, practice templates optional

**To compile practices:** Fill them in with actual code, then use CMake template

**Focus:** Phase 5 (CUDA) solutions are most critical for Nvidia interviews!

---

Good luck with your learning and interview prep! 🚀
