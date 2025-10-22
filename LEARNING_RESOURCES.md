# C++ to Nvidia Learning Resources

A curated collection of learning resources, courses, tutorials, and tools to support your journey from C++ fundamentals to Nvidia employment.

---

## Table of Contents
1. [Getting Started](#getting-started)
2. [Phase 1: C++ Fundamentals](#phase-1-c-fundamentals)
3. [Phase 2: Intermediate C++](#phase-2-intermediate-c)
4. [Phase 3: Advanced C++](#phase-3-advanced-c)
5. [Phase 4: Performance & Optimization](#phase-4-performance--optimization)
6. [Phase 5: GPU Programming & CUDA](#phase-5-gpu-programming--cuda)
7. [Phase 6: Specialized Topics](#phase-6-specialized-topics)
8. [Practice Platforms](#practice-platforms)
9. [Tools & Software](#tools--software)
10. [Community & Forums](#community--forums)
11. [Nvidia-Specific Resources](#nvidia-specific-resources)

---

## Getting Started

### Development Environment Setup

**IDEs & Editors:**
- [Visual Studio Code](https://code.visualstudio.com/) - Free, cross-platform
  - [C/C++ Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)
  - [CMake Tools Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)
- [CLion](https://www.jetbrains.com/clion/) - Professional IDE (free for students)
- [Visual Studio](https://visualstudio.microsoft.com/) - Windows, Community Edition free

**Compilers:**
- [GCC](https://gcc.gnu.org/) - GNU Compiler Collection
- [Clang](https://clang.llvm.org/) - LLVM-based compiler
- [MSVC](https://visualstudio.microsoft.com/vs/features/cplusplus/) - Microsoft Visual C++

**Build Systems:**
- [CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
- [CMake Documentation](https://cmake.org/documentation/)

---

## Phase 1: C++ Fundamentals

### Free Online Courses

**Comprehensive Tutorials:**
- [learncpp.com](https://www.learncpp.com/) - Excellent free comprehensive C++ tutorial
- [C++ Tutorial - GeeksforGeeks](https://www.geeksforgeeks.org/c-plus-plus/)
- [C++ Language Tutorial - cplusplus.com](https://cplusplus.com/doc/tutorial/)
- [Microsoft C++ Documentation](https://learn.microsoft.com/en-us/cpp/)

**Video Courses:**
- [C++ Tutorial for Beginners - freeCodeCamp](https://www.youtube.com/watch?v=vLnPwxZdW4Y) - 4-hour comprehensive video
- [C++ Programming Course - Caleb Curry](https://www.youtube.com/playlist?list=PL_c9BZzLwBRJVJsIfe97ey45V4LP_HXiG)
- [C++ Full Course - Bro Code](https://www.youtube.com/watch?v=-TkoO8Z07hI)

**Interactive Learning:**
- [SoloLearn C++](https://www.sololearn.com/learn/courses/c-plus-plus-introduction) - Mobile-friendly
- [Codecademy Learn C++](https://www.codecademy.com/learn/learn-c-plus-plus) - Interactive (free tier available)

### Reference Materials
- [cppreference.com](https://en.cppreference.com/) - **Essential reference** for C++ standard library
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) - Best practices

### Books
- "C++ Primer" (5th Edition) by Stanley Lippman, Josée Lajoie, Barbara E. Moo
- "Programming: Principles and Practice Using C++" by Bjarne Stroustrup
- "A Tour of C++" (3rd Edition) by Bjarne Stroustrup

---

## Phase 2: Intermediate C++

### Object-Oriented Programming

**Tutorials:**
- [OOP in C++ - GeeksforGeeks](https://www.geeksforgeeks.org/object-oriented-programming-in-cpp/)
- [C++ Classes and Objects - learncpp.com](https://www.learncpp.com/cpp-tutorial/introduction-to-object-oriented-programming/)
- [OOP C++ Tutorial - Programiz](https://www.programiz.com/cpp-programming/object-class)

**Video Courses:**
- [Object Oriented Programming (OOP) in C++ Course](https://www.youtube.com/watch?v=wN0x9eZLix4)
- [C++ OOP - The Cherno](https://www.youtube.com/playlist?list=PLlrATfBNZ98dudnM48yfGUldqGD0S4FFb)

### STL (Standard Template Library)

**Resources:**
- [STL Tutorial - cplusplus.com](https://cplusplus.com/reference/stl/)
- [C++ STL - GeeksforGeeks](https://www.geeksforgeeks.org/the-c-standard-template-library-stl/)
- [STL Containers - cppreference](https://en.cppreference.com/w/cpp/container)
- [STL Algorithms - cppreference](https://en.cppreference.com/w/cpp/algorithm)

**Video Tutorials:**
- [C++ STL Tutorial - Bo Qian](https://www.youtube.com/playlist?list=PL5jc9xFGsL8G3y3ywuFSvOuNm3GjBwdkb)

### Books
- "Effective C++" (3rd Edition) by Scott Meyers
- "Effective STL" by Scott Meyers
- "C++ Standard Library" by Nicolai M. Josuttis

---

## Phase 3: Advanced C++

### Templates & Modern C++

**Online Resources:**
- [C++ Templates Tutorial - learncpp.com](https://www.learncpp.com/cpp-tutorial/template-classes/)
- [Modern C++ Features - GitHub](https://github.com/AnthonyCalandra/modern-cpp-features)
- [C++11/14/17/20 Features - cppreference](https://en.cppreference.com/w/cpp/compiler_support)
- [C++20 Overview](https://en.cppreference.com/w/cpp/20)

**Video Courses:**
- [C++ Templates - The Cherno](https://www.youtube.com/watch?v=I-hZkUa9mIs)
- [CppCon Talks](https://www.youtube.com/user/CppCon) - Advanced C++ conference talks
- [Back to Basics: Templates - CppCon](https://www.youtube.com/results?search_query=cppcon+templates)

### Concurrency & Multithreading

**Tutorials:**
- [C++ Concurrency - cppreference](https://en.cppreference.com/w/cpp/thread)
- [Multithreading Tutorial - learncpp.com](https://www.learncpp.com/cpp-tutorial/introduction-to-multithreading/)
- [C++ Threading - ModernesCpp](https://www.modernescpp.com/index.php/category/multithreading)

**Video Resources:**
- [Threading in C++ - The Cherno](https://www.youtube.com/watch?v=wXBcwHwIt0w)
- [CppCon: Multithreading](https://www.youtube.com/results?search_query=cppcon+multithreading)

**Courses:**
- [C++ Concurrency in Action - Manning (book + video)](https://www.manning.com/books/c-plus-plus-concurrency-in-action-second-edition)

### Books
- "Effective Modern C++" by Scott Meyers
- "C++ Concurrency in Action" (2nd Edition) by Anthony Williams
- "C++ Templates: The Complete Guide" (2nd Edition) by Vandevoorde, Josuttis, and Gregor
- "The C++ Programming Language" (4th Edition) by Bjarne Stroustrup

---

## Phase 4: Performance & Optimization

### Performance Analysis & Optimization

**Documentation:**
- [Agner Fog's Optimization Manuals](https://www.agner.org/optimize/) - **Must-read** for optimization
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [Compiler Explorer (Godbolt)](https://godbolt.org/) - See assembly output

**Video Resources:**
- [CppCon: Performance](https://www.youtube.com/results?search_query=cppcon+performance)
- [Performance Analysis - Chandler Carruth](https://www.youtube.com/watch?v=fHNmRkzxHWs)
- [Cache-Friendly Code - CppCon](https://www.youtube.com/results?search_query=cppcon+cache)

**Tools:**
- [Valgrind](https://valgrind.org/) - Memory profiling
- [perf](https://perf.wiki.kernel.org/index.php/Main_Page) - Linux profiling
- [Google Benchmark](https://github.com/google/benchmark) - Microbenchmarking library
- [gprof](https://sourceware.org/binutils/docs/gprof/) - GNU profiler

### SIMD Programming

**Resources:**
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [SIMD for C++ Developers](https://www.intel.com/content/www/us/en/developer/articles/technical/simd-for-c-developers.html)
- [CppCon: SIMD](https://www.youtube.com/results?search_query=cppcon+simd)

### Courses
- [Performance Engineering of Software Systems - MIT OCW](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/)

### Books
- "Optimizing C++" by Agner Fog (free PDF)
- "Computer Architecture: A Quantitative Approach" by Hennessy & Patterson
- "Systems Performance" by Brendan Gregg

---

## Phase 5: GPU Programming & CUDA

### Official Nvidia Resources (Start Here!)

**Documentation:**
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/) - **Primary resource**
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples) - Official code examples

**Tools:**
- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- [Nsight Compute](https://developer.nvidia.com/nsight-compute) - Kernel profiler
- [Nsight Systems](https://developer.nvidia.com/nsight-systems) - System-wide profiler

### Free Online Courses

**Video Courses:**
- [Intro to Parallel Programming - Udacity (Nvidia)](https://www.udacity.com/course/intro-to-parallel-programming--cs344) - **Highly recommended**
- [CUDA Crash Course - CoffeeBeforeArch](https://www.youtube.com/playlist?list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU)
- [CUDA Programming - Nvidia Developer](https://www.youtube.com/c/NVIDIADeveloper)

**University Courses:**
- [Heterogeneous Parallel Programming - Coursera (UIUC)](https://www.coursera.org/learn/heterogeneous-parallel-programming)
- [GPU Programming - Caltech](http://courses.cms.caltech.edu/cs179/)

### Tutorials & Guides

**Beginner:**
- [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [CUDA Refresher Series - Nvidia Blog](https://developer.nvidia.com/blog/tag/cuda-refresher/)
- [CUDA Tutorial - CodeProject](https://www.codeproject.com/Articles/3009/Introduction-to-CUDA)

**Advanced:**
- [Nvidia Developer Blog](https://developer.nvidia.com/blog/) - Latest techniques
- [Nvidia GTC On-Demand](https://www.nvidia.com/en-us/on-demand/) - Conference talks
- [Optimizing CUDA Applications - Nvidia](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#optimization)

### CUDA Libraries

**Documentation:**
- [cuBLAS](https://docs.nvidia.com/cuda/cublas/) - Linear algebra
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/) - Deep learning primitives
- [Thrust](https://docs.nvidia.com/cuda/thrust/) - Parallel algorithms
- [CUB](https://nvlabs.github.io/cub/) - Reusable CUDA primitives
- [cuFFT](https://docs.nvidia.com/cuda/cufft/) - Fast Fourier Transform
- [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/) - Inference optimization

### Books
- "Programming Massively Parallel Processors" (4th Edition) by Hwu, Kirk, and Hajj - **Best CUDA book**
- "CUDA by Example" by Sanders & Kandrot - Beginner-friendly
- "Professional CUDA C Programming" by Cheng, Grossman, and McKercher
- "CUDA Programming" by Shane Cook

### Community
- [Nvidia Developer Forums](https://forums.developer.nvidia.com/)
- [CUDA subreddit](https://www.reddit.com/r/CUDA/)

---

## Phase 6: Specialized Topics

### Computer Graphics

**OpenGL:**
- [LearnOpenGL](https://learnopengl.com/) - **Best OpenGL tutorial**
- [OpenGL Tutorial](http://www.opengl-tutorial.org/)
- [OpenGL Documentation](https://www.khronos.org/opengl/)

**Vulkan:**
- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [Vulkan Guide](https://github.com/KhronosGroup/Vulkan-Guide)
- [Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples)

**Ray Tracing:**
- [Ray Tracing in One Weekend](https://raytracing.github.io/) - Free book series
- [Nvidia RTX Ray Tracing](https://developer.nvidia.com/rtx/ray-tracing)
- [Scratchapixel](https://www.scratchapixel.com/) - Computer graphics from scratch

**Books:**
- "Real-Time Rendering" (4th Edition) by Akenine-Möller et al.
- "Physically Based Rendering" (3rd Edition) by Pharr, Jakob, and Humphreys

### Deep Learning & AI

**Courses:**
- [Deep Learning Specialization - Coursera](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai](https://www.fast.ai/) - Practical deep learning
- [Stanford CS231n](http://cs231n.stanford.edu/) - CNNs for Visual Recognition

**Nvidia Deep Learning:**
- [Nvidia Deep Learning Institute](https://www.nvidia.com/en-us/training/) - Certifications
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [RAPIDS Documentation](https://docs.rapids.ai/) - GPU-accelerated data science

**Frameworks:**
- [PyTorch](https://pytorch.org/tutorials/) - Popular ML framework
- [TensorFlow](https://www.tensorflow.org/tutorials) - Google's ML framework

### Systems Programming

**Linux Programming:**
- [The Linux Programming Interface](http://man7.org/tlpi/) - Book
- [Linux System Programming - O'Reilly](https://www.oreilly.com/library/view/linux-system-programming/9781449341527/)

**Resources:**
- [Linux Kernel Documentation](https://www.kernel.org/doc/html/latest/)
- [Beej's Guide to Network Programming](https://beej.us/guide/bgnet/)

---

## Practice Platforms

### Coding Practice
- [LeetCode](https://leetcode.com/) - Algorithm practice
- [HackerRank](https://www.hackerrank.com/domains/cpp) - C++ challenges
- [Codeforces](https://codeforces.com/) - Competitive programming
- [Project Euler](https://projecteuler.net/) - Mathematical problems
- [Exercism C++ Track](https://exercism.org/tracks/cpp) - Mentored practice

### Coding Challenges
- [Advent of Code](https://adventofcode.com/) - Annual coding event
- [Codewars](https://www.codewars.com/) - Kata challenges
- [CodinGame](https://www.codingame.com/) - Game-based learning

---

## Tools & Software

### Development Tools

**Compilers & Build:**
- [GCC](https://gcc.gnu.org/)
- [Clang/LLVM](https://clang.llvm.org/)
- [CMake](https://cmake.org/)
- [Ninja Build](https://ninja-build.org/)

**Debuggers:**
- [GDB](https://www.sourceware.org/gdb/) - GNU Debugger
- [LLDB](https://lldb.llvm.org/) - LLVM Debugger
- [cuda-gdb](https://docs.nvidia.com/cuda/cuda-gdb/) - CUDA Debugger
- [Visual Studio Debugger](https://visualstudio.microsoft.com/)

**Profilers:**
- [Nsight Compute](https://developer.nvidia.com/nsight-compute) - CUDA kernel profiler
- [Nsight Systems](https://developer.nvidia.com/nsight-systems) - System profiler
- [Valgrind](https://valgrind.org/) - Memory profiler
- [perf](https://perf.wiki.kernel.org/) - Linux profiler
- [Intel VTune](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)

**Testing:**
- [Google Test](https://github.com/google/googletest) - C++ testing framework
- [Catch2](https://github.com/catchorg/Catch2) - Modern C++ test framework
- [Boost.Test](https://www.boost.org/doc/libs/release/libs/test/)

**Static Analysis:**
- [Clang-Tidy](https://clang.llvm.org/extra/clang-tidy/)
- [Cppcheck](http://cppcheck.sourceforge.net/)
- [PVS-Studio](https://pvs-studio.com/)

**Package Managers:**
- [vcpkg](https://vcpkg.io/) - Microsoft's package manager
- [Conan](https://conan.io/) - C++ package manager

### Documentation Tools
- [Doxygen](https://www.doxygen.nl/) - Documentation generator

---

## Community & Forums

### Q&A and Discussion
- [Stack Overflow - C++](https://stackoverflow.com/questions/tagged/c%2b%2b)
- [Stack Overflow - CUDA](https://stackoverflow.com/questions/tagged/cuda)
- [r/cpp](https://www.reddit.com/r/cpp/) - C++ subreddit
- [r/cpp_questions](https://www.reddit.com/r/cpp_questions/)
- [C++ Discord](https://www.includecpp.org/) - #include <C++> community

### Forums
- [Nvidia Developer Forums](https://forums.developer.nvidia.com/)
- [CPPReference Forums](https://en.cppreference.com/w/cpp/links/community)

### Conferences & Talks
- [CppCon](https://cppcon.org/) - Annual C++ conference
  - [CppCon YouTube](https://www.youtube.com/user/CppCon)
- [Nvidia GTC](https://www.nvidia.com/gtc/) - GPU Technology Conference
- [C++Now](https://cppnow.org/)
- [Meeting C++](https://meetingcpp.com/)

### Newsletters & Blogs
- [C++ Weekly - Jason Turner](https://www.youtube.com/c/lefticus1)
- [Fluent C++](https://www.fluentcpp.com/)
- [ModernesCpp](https://www.modernescpp.com/)
- [Nvidia Developer Blog](https://developer.nvidia.com/blog/)

---

## Nvidia-Specific Resources

### Career & Recruitment

**Official:**
- [Nvidia Careers](https://www.nvidia.com/en-us/about-nvidia/careers/)
- [Nvidia University Programs](https://www.nvidia.com/en-us/about-nvidia/academic-partnerships/)
- [Nvidia Internships](https://www.nvidia.com/en-us/about-nvidia/careers/university-recruiting/)

### Certifications
- [Nvidia Deep Learning Institute](https://www.nvidia.com/en-us/training/) - Professional certifications
- [DLI Courses](https://www.nvidia.com/en-us/training/online/)

### Developer Programs
- [Nvidia Developer Program](https://developer.nvidia.com/developer-program) - Free membership
- [Nvidia Inception](https://www.nvidia.com/en-us/deep-learning-ai/startups/) - For startups

### Technical Resources

**Architecture Whitepapers:**
- [Nvidia Architecture Whitepapers](https://www.nvidia.com/en-us/data-center/resources/architecture-whitepapers/)
- [Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)
- [Ampere Architecture](https://www.nvidia.com/en-us/data-center/ampere-architecture/)

**Products & Technologies:**
- [CUDA Zone](https://developer.nvidia.com/cuda-zone)
- [RTX Technology](https://developer.nvidia.com/rtx)
- [Omniverse](https://developer.nvidia.com/omniverse)
- [RAPIDS](https://developer.nvidia.com/rapids) - Data science on GPU
- [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)

### Research Papers
- [Nvidia Research](https://www.nvidia.com/en-us/research/)
- [Nvidia Technical Blog](https://developer.nvidia.com/blog/)

---

## Learning Path Recommendations

### For Complete Beginners
1. Start with [learncpp.com](https://www.learncpp.com/)
2. Practice on [Exercism](https://exercism.org/tracks/cpp)
3. Read "C++ Primer"
4. Build small projects from Phase 1

### For Those with C++ Basics
1. Jump to Phase 4 (Performance) and Phase 5 (CUDA)
2. Take [Udacity's Intro to Parallel Programming](https://www.udacity.com/course/intro-to-parallel-programming--cs344)
3. Read "Programming Massively Parallel Processors"
4. Study Nvidia's CUDA samples
5. Build GPU-accelerated projects

### For GPU Programming Focus
1. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
2. [Udacity CUDA Course](https://www.udacity.com/course/intro-to-parallel-programming--cs344)
3. [Nvidia GTC On-Demand](https://www.nvidia.com/en-us/on-demand/)
4. Practice with [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
5. Profile with Nsight tools
6. Build portfolio projects

---

## Study Tips

### Effective Learning
1. **Learn by Doing**: Code along with tutorials
2. **Build Projects**: Apply knowledge to real problems
3. **Read Documentation**: Primary source of truth
4. **Profile Everything**: Measure, don't guess
5. **Contribute to Open Source**: Real-world experience
6. **Join Communities**: Learn from others

### Time Management
- **Consistency over Intensity**: Code daily
- **Pomodoro Technique**: 25-min focused sessions
- **Active Learning**: Take notes, explain concepts
- **Spaced Repetition**: Review previous topics

### Portfolio Building
- Host all code on GitHub
- Write README files with benchmarks
- Create demo videos
- Blog about your learning
- Contribute to open-source projects

---

## Quick Reference Card

### Essential Daily Resources
- [cppreference.com](https://en.cppreference.com/) - C++ reference
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - CUDA reference
- [Compiler Explorer](https://godbolt.org/) - Assembly viewer
- [Stack Overflow](https://stackoverflow.com/) - Q&A

### Must-Watch Channels
- [CppCon](https://www.youtube.com/user/CppCon)
- [The Cherno](https://www.youtube.com/c/TheChernoProject)
- [Nvidia Developer](https://www.youtube.com/c/NVIDIADeveloper)

### Must-Read Books
1. "C++ Primer" - Lippman et al.
2. "Effective Modern C++" - Scott Meyers
3. "Programming Massively Parallel Processors" - Hwu et al.

---

## Additional Resources

### C++ Standards
- [ISO C++ Website](https://isocpp.org/)
- [C++ Standard Draft](https://eel.is/c++draft/)
- [C++ FAQ](https://isocpp.org/faq)

### Online Compilers
- [Compiler Explorer (Godbolt)](https://godbolt.org/)
- [Wandbox](https://wandbox.org/)
- [Repl.it](https://replit.com/)
- [OnlineGDB](https://www.onlinegdb.com/)

### Visualization Tools
- [C++ Insights](https://cppinsights.io/) - See what compiler does
- [Quick Bench](https://quick-bench.com/) - Quick benchmarks

---

**Remember**: The best resource is hands-on practice. Use these materials as guides, but spend most of your time writing, debugging, and optimizing code. Build projects, measure performance, and iterate.

Good luck on your journey to Nvidia!
