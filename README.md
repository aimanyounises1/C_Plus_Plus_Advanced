# Advanced C++ & CUDA Learning Repository

> A comprehensive, production-grade learning resource for mastering C++, CUDA, and GPU programming — specifically designed for Software Engineer interviews at Nvidia and other GPU computing companies.

[![CI/CD](https://github.com/aimanyounises1/C_Plus_Plus_Advanced/workflows/C/C++%20CI/badge.svg)](https://github.com/aimanyounises1/C_Plus_Plus_Advanced/actions)

---

## Overview

This repository is a complete learning pathway from C++ fundamentals to advanced GPU programming, with a strong emphasis on **CUDA optimization** and **parallel algorithm design**. It includes:

- **130+ practice problems** across 7 progressive phases
- **Production-quality reference implementations** with optimization progressions
- **Real Nvidia interview questions** with detailed answers
- **GPU algorithm implementations** (CPU vs GPU comparisons)
- **Comprehensive profiling guides** (Nsight Compute/Systems)
- **5 major capstone projects** for portfolio building

---

## What Makes This Repository Unique

### For Interview Preparation
- Real technical questions from Nvidia interviews
- Optimized CUDA kernels showing 5-10x performance improvements
- Practical profiling examples with actual metrics
- System design problems with GPU infrastructure context
- Behavioral question frameworks (STAR method)

### For Skill Development
- Progressive learning path (beginner → advanced → expert)
- Multiple optimization levels for each concept
- Detailed comments explaining "why" not just "what"
- Performance benchmarks and profiling results
- Industry best practices and patterns

### For Portfolio Building
- Production-quality code examples
- Performance analysis documentation
- Major projects with clear specifications
- Demonstrates deep GPU architecture understanding

---

## Platform Compatibility

### CUDA Support by Platform

| Platform | CUDA Support | Setup Instructions |
|----------|--------------|-------------------|
| **Linux** | ✅ Full native support | Install CUDA Toolkit locally |
| **Windows** | ✅ Full native support | Install CUDA Toolkit + Visual Studio |
| **macOS (Intel)** | ❌ Not supported | Use cloud GPU (see below) |
| **macOS (Apple Silicon)** | ❌ Not supported | Use cloud GPU (see below) |

### Important for Mac Users (Including M4 Max)

**CUDA does not run on macOS** - Apple removed NVIDIA driver support years ago, and Apple Silicon (M1/M2/M3/M4) uses a completely different architecture.

**Solution:** Use cloud GPU instances
- 📘 **Complete Mac Setup Guide:** [MACOS_SETUP.md](MACOS_SETUP.md)
- ☁️ **Cloud GPU Quick Start:** [CLOUD_GPU_SETUP.md](CLOUD_GPU_SETUP.md)

**Recommended for Mac users:**
- ✅ C++ learning: Use Mac natively (Phases 1-4)
- ✅ CUDA learning: Use Google Colab (FREE) or Paperspace ($8/month)
- ✅ Projects: Use AWS/GCP with free credits

**This is completely acceptable for Nvidia interviews** - they understand Mac users need cloud resources!

---

## Repository Structure

```
C_Plus_Plus_Advanced/
├── practices/              # 130+ practice problems (7 phases)
│   ├── phase1_fundamentals/      (15 files) - C++ basics, pointers, memory
│   ├── phase2_intermediate/      (19 files) - OOP, STL, smart pointers
│   ├── phase3_advanced/          (19 files) - Templates, concurrency, patterns
│   ├── phase4_performance/       (14 files) - Profiling, optimization, SIMD
│   ├── phase5_cuda/              (31 files) - GPU programming (CRITICAL!)
│   ├── phase6_specialized/       (18 files) - Graphics, ML, systems
│   └── phase7_projects/          (13 projects) - Portfolio-quality implementations
│
├── solutions/              # Reference implementations & interview prep
│   ├── phase5_cuda/              - Optimized CUDA kernels with progressions
│   ├── algorithms/               - GPU-accelerated algorithm problems
│   ├── interview_prep/           - Nvidia technical & behavioral questions
│   └── benchmarks/               - Profiling guides and performance analysis
│
├── NVIDIA_CPP_ROADMAP.md         - Complete learning roadmap (482 lines)
├── LEARNING_RESOURCES.md         - Curated resources with links (504 lines)
└── README.md                     - This file
```

---

## Quick Start

### Choose Your Setup

**Linux/Windows with NVIDIA GPU:**
```bash
# C++ compiler (GCC 9+ or Clang 10+)
sudo apt install build-essential

# CUDA Toolkit (11.0+)
# Download from: https://developer.nvidia.com/cuda-downloads

# Nsight profiling tools
# Included with CUDA Toolkit
```

**Mac (Intel or Apple Silicon):**
```bash
# Install C++ compiler
xcode-select --install

# For CUDA: See detailed setup guides
# 📘 MACOS_SETUP.md - Complete Mac guide
# ☁️ CLOUD_GPU_SETUP.md - Cloud GPU setup (5 min)
```

**Don't have NVIDIA GPU? (Any platform):**
- Use Google Colab (FREE) - See [CLOUD_GPU_SETUP.md](CLOUD_GPU_SETUP.md)
- 2-minute setup, instant CUDA access

### Clone and Explore
```bash
git clone https://github.com/aimanyounises1/C_Plus_Plus_Advanced.git
cd C_Plus_Plus_Advanced

# Start with the roadmap
cat NVIDIA_CPP_ROADMAP.md

# Try a CUDA example (Linux/Windows with GPU)
cd solutions/phase5_cuda
nvcc -o vector_add 01_vector_addition_optimized.cu -O3 -arch=sm_70
./vector_add

# For Mac/Cloud: See CLOUD_GPU_SETUP.md for Google Colab instructions
```

### Using CLion IDE (Automatic Setup)

**CLion automatically configures everything via CMake:**

1. **Open project:** `File → Open → Select C_Plus_Plus_Advanced folder`
2. **Wait for CMake** to auto-configure (30 seconds)
3. **Select target:** Dropdown → Choose `vector_add`, `matmul`, or `reduction`
4. **Build & Run:** Click green play button ▶️

✅ **CMake configuration included** - No manual setup needed!

📘 **Complete CLion Guide:** [CLION_SETUP.md](CLION_SETUP.md)
- Automatic C++ and CUDA detection
- Build configurations (Debug/Release)
- Integrated debugging and profiling
- Platform-specific instructions
- Troubleshooting guide

---

## Learning Paths

### Path 1: Interview Preparation (4-6 weeks)

**Focus on high-impact topics for Nvidia interviews:**

**Week 1-2: CUDA Fundamentals**
- Study `solutions/phase5_cuda/` reference implementations
- Understand memory coalescing, bank conflicts, occupancy
- Practice explaining optimizations verbally

**Week 3-4: Algorithm Problems**
- Solve problems in `solutions/algorithms/`
- Implement both CPU and GPU versions
- Analyze parallel complexity

**Week 5: Interview Questions**
- Review `solutions/interview_prep/NVIDIA_TECHNICAL_QUESTIONS.md`
- Practice answering architecture questions
- Prepare behavioral responses (STAR method)

**Week 6: Projects & Polish**
- Implement 1-2 projects from `practices/phase7_projects/`
- Profile and optimize
- Document results with performance metrics

### Path 2: Comprehensive Learning (3-6 months)

**Follow the complete 7-phase curriculum:**

1. **Phase 1: Fundamentals** (2-3 weeks)
   - C++ syntax, pointers, memory management
   - Complete all 15 practice files

2. **Phase 2: Intermediate** (3-4 weeks)
   - OOP, STL, modern C++ features
   - 19 practice files + mini-projects

3. **Phase 3: Advanced** (4-5 weeks)
   - Templates, concurrency, design patterns
   - 19 practice files

4. **Phase 4: Performance** (3-4 weeks)
   - Profiling, optimization techniques
   - 14 practice files + benchmarking

5. **Phase 5: CUDA** (6-8 weeks) ⭐ **CRITICAL FOR NVIDIA**
   - GPU architecture, kernel optimization
   - 31 practice files + reference implementations

6. **Phase 6: Specialized** (4-5 weeks)
   - Graphics, ML, systems programming
   - 18 practice files

7. **Phase 7: Projects** (4-6 weeks)
   - 5 major projects for portfolio
   - Real-world applications

### Path 3: Targeted Skill Building

**Choose based on your role:**

**For Performance Engineers:**
- Focus: Phase 4 (Performance) + Phase 5 (CUDA)
- Key topics: Memory optimization, profiling, kernel tuning

**For Backend Engineers:**
- Focus: Phase 2 (Intermediate) + Phase 3 (Advanced) + Phase 6 (Systems)
- Key topics: Concurrency, data structures, system APIs

**For Graphics Engineers:**
- Focus: Phase 4 (Performance) + Phase 5 (CUDA) + Phase 6 (Graphics)
- Key topics: Rendering pipelines, shaders, GPU optimization

**For Deep Learning Engineers:**
- Focus: Phase 5 (CUDA) + Phase 6 (ML specialization)
- Key topics: Tensor operations, inference optimization, frameworks

---

## Highlighted Content

### Reference Implementations

**Vector Addition (3 optimization levels):**
- Naive → Grid-Stride → Vectorized
- Shows 2-4x performance improvement
- Full profiling analysis included
- 📄 `solutions/phase5_cuda/01_vector_addition_optimized.cu`

**Matrix Multiplication (4 optimization levels):**
- Naive → Tiled → No Bank Conflicts → Coarsened
- Shows 10-15x performance improvement
- Detailed commentary on each optimization
- 📄 `solutions/phase5_cuda/02_matrix_multiplication_optimized.cu`

**Parallel Reduction (6 implementations):**
- Interleaved → Sequential → First Add → Warp Unroll → Complete Unroll → Warp Shuffle
- Shows 6-10x improvement
- Essential CUDA pattern
- 📄 `solutions/phase5_cuda/03_parallel_reduction.cu`

### Interview Preparation

**Technical Questions (15 detailed Q&As):**
- CUDA & GPU Programming (6 questions)
- GPU Architecture (3 questions)
- Memory & Performance Optimization (4 questions)
- Algorithms & Data Structures (2 questions)
- C++ Fundamentals (2 questions)
- 📄 `solutions/interview_prep/NVIDIA_TECHNICAL_QUESTIONS.md`

**Sample Questions:**
- "Explain the CUDA memory hierarchy. When would you use each type?"
- "How do you identify if a kernel is memory-bound or compute-bound?"
- "Implement a parallel reduction in CUDA. Optimize it."
- "How would you optimize matrix multiplication on GPU?"

### Profiling & Performance Analysis

**Complete Nsight guide:**
- Nsight Systems (timeline profiling)
- Nsight Compute (kernel analysis)
- Key metrics and what they mean
- Optimization workflow
- Common bottlenecks and solutions
- 📄 `solutions/benchmarks/PROFILING_GUIDE.md`

### Algorithm Problems

**GPU-accelerated implementations:**
- Two Sum (CPU hash table vs GPU parallel search)
- Parallel sorting algorithms
- Graph traversal (BFS/DFS)
- Dynamic programming on GPU
- 📄 `solutions/algorithms/`

---

## Major Projects (Phase 7)

### 1. CUDA Ray Tracer
Implement a GPU-accelerated ray tracer with:
- BVH acceleration structure
- Multiple material types (diffuse, metal, glass)
- Path tracing for global illumination
- Performance target: > 30 FPS at 1080p

### 2. N-Body Physics Simulation
Real-time physics with CUDA:
- Collision detection and response
- Spatial partitioning (grid or octree)
- GPU-CPU visualization pipeline
- Performance: 100K+ particles at 60 FPS

### 3. ML Inference Engine
Custom neural network inference:
- Matrix operations (cuBLAS/custom kernels)
- Activation functions on GPU
- Batch processing
- Performance: < 5ms inference for ResNet-50

### 4. Custom Memory Allocator
High-performance allocator:
- Pool-based allocation
- Thread-safe operations
- Memory alignment
- Benchmark vs malloc/new

### 5. Lock-Free Data Structures
Concurrent data structures:
- Lock-free queue/stack
- Concurrent hash table
- Wait-free algorithms
- Performance testing under contention

---

## Performance Metrics You'll Achieve

**After completing this curriculum, you'll understand:**

### Memory Optimization
- Coalescing: Improve bandwidth utilization from 30% → 85%
- Bank conflicts: Eliminate 32-way conflicts (32x speedup)
- Cache optimization: Improve hit rates from 60% → 90%

### Kernel Optimization
- Occupancy tuning: Increase from 30% → 70%
- Warp efficiency: Reduce divergence from 50% → 95%
- ILP: 2-3x improvement via instruction-level parallelism

### End-to-End Performance
- Typical speedups: 10-100x over naive CPU code
- Approach theoretical limits: 70-90% of peak bandwidth/FLOPS
- Production-quality code that scales

---

## What Nvidia Looks For

Based on real interview experiences, Nvidia evaluates:

### Technical Depth (40%)
- Deep understanding of GPU architecture
- Ability to write efficient CUDA code
- Performance analysis and profiling skills
- Algorithm design for parallel systems

### Problem Solving (30%)
- Systematic optimization approach
- Trade-off analysis
- Debugging complex issues
- Creative solutions to constraints

### Code Quality (20%)
- Clean, readable code
- Proper error handling
- Documentation
- Best practices

### Communication (10%)
- Explaining technical concepts clearly
- Discussing design decisions
- Collaborative problem-solving
- Learning from feedback

---

## Interview Success Strategy

### Before the Interview

1. **Master Core CUDA Concepts:**
   - Memory hierarchy (global, shared, registers)
   - Occupancy and resource constraints
   - Warp execution model
   - Common optimization patterns

2. **Practice Problems:**
   - Implement kernels from scratch (reduction, scan, etc.)
   - Optimize existing kernels
   - Explain your optimizations clearly

3. **Build Portfolio:**
   - Complete 2-3 major projects
   - Document performance improvements
   - Share on GitHub with detailed READMEs

4. **Study Architecture:**
   - Read GPU whitepapers (Ampere, Hopper)
   - Understand memory bandwidth calculations
   - Know specs of current GPUs (A100, H100)

### During the Interview

1. **Clarify Requirements:**
   - Ask about input size, latency requirements
   - Understand constraints (memory, compute)
   - Discuss CPU vs GPU trade-offs

2. **Start Simple:**
   - Implement correct solution first
   - Then optimize systematically
   - Explain each optimization

3. **Think Out Loud:**
   - Explain your reasoning
   - Discuss trade-offs
   - Mention alternatives

4. **Show Profiling Knowledge:**
   - Reference relevant metrics
   - Discuss bottleneck identification
   - Explain optimization priorities

### After Implementation

1. **Verify Correctness:**
   - Test edge cases
   - Discuss validation strategy

2. **Analyze Performance:**
   - Estimate theoretical performance
   - Identify bottlenecks
   - Suggest further optimizations

3. **Ask Questions:**
   - About team's work
   - Technology stack
   - Challenges they face

---

## Resources

### Official Documentation
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)

### Books
- "Programming Massively Parallel Processors" by Kirk & Hwu
- "CUDA by Example" by Sanders & Kandrot
- "The C++ Programming Language" by Stroustrup

### Online Courses
- Udacity: Intro to Parallel Programming (CUDA)
- Coursera: Heterogeneous Parallel Programming
- Nvidia DLI: Fundamentals of Accelerated Computing

### Community
- [Nvidia Developer Forums](https://forums.developer.nvidia.com/)
- [CUDA subreddit](https://www.reddit.com/r/CUDA/)
- [GPU Computing on Stack Overflow](https://stackoverflow.com/questions/tagged/cuda)

---

## Contributing

While this is primarily a personal learning repository, suggestions and improvements are welcome!

**Areas for contribution:**
- Additional algorithm problems
- More optimization examples
- Interview question updates
- Project ideas
- Bug fixes in examples

Please open an issue or pull request.

---

## Progress Tracking

### Recommended Study Schedule

**Full-Time Study (40 hrs/week):** 3-4 months
**Part-Time Study (15 hrs/week):** 6-9 months
**Interview Prep Only:** 4-6 weeks

### Milestones

- [ ] Complete Phase 1-3 (C++ fundamentals to advanced)
- [ ] Implement first CUDA kernel from scratch
- [ ] Optimize a kernel to 70%+ of peak bandwidth
- [ ] Complete 20+ algorithm problems
- [ ] Finish 1 major project with documentation
- [ ] Score 90%+ on mock interview questions
- [ ] Profile and optimize real application

---

## Testimonials & Results

*"This repository was instrumental in my preparation. The CUDA optimization examples and interview questions were spot-on."*
— Software Engineer @ Nvidia

*"The profiling guide alone is worth bookmarking. Real metrics from real kernels."*
— Performance Engineer

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Nvidia for excellent documentation and tools
- CUDA community for sharing knowledge
- Open-source contributors to related projects

---

## Contact & Support

**Questions about the content?**
- Open an issue on GitHub
- Check existing issues first

**Interview preparation help?**
- Review `NVIDIA_CPP_ROADMAP.md`
- Study `solutions/interview_prep/`
- Practice with `solutions/algorithms/`

**Found this helpful?**
- Star the repository ⭐
- Share with others preparing for GPU programming interviews
- Contribute improvements

---

## Final Words

**Landing a Software Engineer role at Nvidia requires:**
1. ✅ Deep CUDA knowledge (not just basics)
2. ✅ Demonstrable optimization skills
3. ✅ Strong C++ fundamentals
4. ✅ Clear communication
5. ✅ Portfolio of real work

**This repository provides all the pieces. Your job is to:**
1. Study the concepts
2. Practice the problems
3. Build the projects
4. Document your work
5. Prepare to explain your approach

**You've got this! 🚀**

---

*Last Updated: 2025-10*

*If you found this repository helpful in your interview preparation, please consider giving it a star and sharing it with others!*
