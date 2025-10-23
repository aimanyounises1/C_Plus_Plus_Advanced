# CLion Setup Guide

> Complete guide for setting up CLion IDE for C++ and CUDA development with this repository.

---

## Quick Start (TL;DR)

1. **Open project in CLion:** `File → Open → Select C_Plus_Plus_Advanced folder`
2. **CLion auto-detects CMakeLists.txt** and configures automatically
3. **Select configuration:** Top toolbar → Choose target (e.g., `vector_add`)
4. **Build & Run:** Click green play button ▶️

Done! CLion handles everything.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Opening the Project](#opening-the-project)
3. [Configuration](#configuration)
4. [Building and Running](#building-and-running)
5. [Platform-Specific Setup](#platform-specific-setup)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Configuration](#advanced-configuration)

---

## Prerequisites

### Required
- **CLion 2020.1 or later** (download from [jetbrains.com/clion](https://www.jetbrains.com/clion/))
- **CMake 3.18+** (usually bundled with CLion)
- **C++ Compiler:**
  - Linux: GCC 9+ or Clang 10+
  - macOS: Xcode Command Line Tools
  - Windows: Visual Studio 2019+ or MinGW

### Optional (for CUDA)
- **CUDA Toolkit 11.0+** (for CUDA examples)
  - Download: [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
  - **Not available on macOS** - see [MACOS_SETUP.md](MACOS_SETUP.md)

---

## Opening the Project

### Step 1: Launch CLion

Open CLion application.

### Step 2: Open Project

**Option A: From Welcome Screen**
1. Click `Open`
2. Navigate to `C_Plus_Plus_Advanced` folder
3. Click `OK`

**Option B: From Menu**
1. `File → Open`
2. Select `C_Plus_Plus_Advanced` folder
3. Click `Open`

### Step 3: Trust the Project

CLion will ask if you trust this project:
- Click `Trust Project`

### Step 4: Wait for CMake Configuration

CLion will automatically:
1. ✅ Detect `CMakeLists.txt`
2. ✅ Run CMake configuration
3. ✅ Index the project
4. ✅ Detect available targets

**This takes 30-60 seconds on first open.**

You'll see progress in the bottom status bar:
```
Loading CMake project...
Indexing...
```

---

## Configuration

### CMake Tool Window

After opening, check the CMake tool window (bottom of IDE):

**What You Should See:**
```
-- Build type: Release
-- C++ Standard: 17
-- C++ Compiler: /usr/bin/g++
-- CUDA Available: TRUE
-- CUDA Compiler: /usr/local/cuda/bin/nvcc
-- CUDA Version: 12.0
========================================
Configuration Summary
========================================
✓ CUDA examples configured:
  - vector_add
  - matmul
  - reduction
✓ GPU algorithm examples configured:
  - two_sum_gpu
```

**If CUDA Not Found:**
```
-- CUDA not found - CUDA examples will be skipped
-- To compile CUDA code:
--   - Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
--   - Or use cloud GPU: see CLOUD_GPU_SETUP.md
```

This is normal on macOS or systems without CUDA.

### Build Configurations

CLion creates these build configurations automatically:

| Configuration | Purpose | Flags |
|--------------|---------|-------|
| **Debug** | Development, debugging | `-g -O0` |
| **Release** | Production, benchmarking | `-O3 -march=native` |
| **RelWithDebInfo** | Debugging optimized code | `-O2 -g` |

**Switch between them:**
- Top toolbar: `Debug/Release` dropdown
- Or: `File → Settings → Build, Execution, Deployment → CMake`

---

## Building and Running

### Available Targets

After CMake configuration, you'll see these targets in the toolbar dropdown:

**CUDA Examples:**
- `vector_add` - Vector addition with 3 optimization levels
- `matmul` - Matrix multiplication with 4 optimization levels
- `reduction` - Parallel reduction with 6 implementations

**Algorithm Problems:**
- `two_sum_gpu` - Two Sum CPU vs GPU comparison

### Build & Run a Target

#### Method 1: Using Toolbar (Easiest)

1. **Select target:** Top toolbar dropdown → `vector_add`
2. **Build:** Hammer icon 🔨 or `Ctrl+F9` (Mac: `⌘+F9`)
3. **Run:** Green play button ▶️ or `Shift+F10` (Mac: `⌃+R`)

#### Method 2: Using Run Menu

1. `Run → Run...` or `Alt+Shift+F10`
2. Select target (e.g., `vector_add`)
3. Program runs in integrated terminal

#### Method 3: Using CMake Tool Window

1. Open CMake tool window (bottom)
2. Right-click on target
3. Choose `Build` or `Run`

### Build Output

CLion shows build output in the `Messages` tool window:

```
====================[ Build | vector_add | Release ]=====================
/usr/local/cuda/bin/nvcc ... -o vector_add
Build finished
```

### Run Output

Program output appears in the `Run` tool window:

```
CUDA Vector Addition Performance Comparison
============================================
Array size: 33554432 elements (128 MB)
Block size: 256
Num blocks: 131072

GPU: Tesla T4
Compute Capability: 7.5
Memory Bandwidth: 320 GB/s

Version 1: Naive:
  Average time: 125.3 μs
  Bandwidth: 244.2 GB/s

Version 2: Grid-Stride:
  Average time: 118.7 μs
  Bandwidth: 257.8 GB/s

...
All versions passed verification! ✓
```

---

## Platform-Specific Setup

### Linux (Ubuntu/Debian)

**With NVIDIA GPU:**

```bash
# Install dependencies
sudo apt update
sudo apt install build-essential cmake

# Install CUDA Toolkit
# Download from: https://developer.nvidia.com/cuda-downloads
# Or:
sudo apt install nvidia-cuda-toolkit

# Verify
nvcc --version
nvidia-smi
```

**Open in CLion:**
- Everything should work automatically
- All CUDA targets will be available

### macOS (Intel or Apple Silicon)

**Important:** CUDA does NOT work on macOS!

**C++ Development (Works):**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Open project in CLion
# C++ examples will work
# CUDA examples will be skipped
```

**CUDA Development (Use Cloud):**
- See [MACOS_SETUP.md](MACOS_SETUP.md) for complete guide
- Use Google Colab, Paperspace, or AWS
- Develop remotely via SSH

**CLion Remote Development (Advanced):**
1. Set up cloud GPU instance
2. In CLion: `Tools → Deployment → Configuration`
3. Add SFTP connection to cloud instance
4. Enable automatic upload
5. Configure remote toolchain
6. Build runs on remote GPU!

See [CLion Remote Development Docs](https://www.jetbrains.com/help/clion/remote-projects-support.html)

### Windows

**With NVIDIA GPU:**

```bash
# Install Visual Studio 2019 or later
# Download CUDA Toolkit from Nvidia
# Install CMake (or use CLion's bundled version)
```

**Open in CLion:**
1. CLion should detect Visual Studio compiler
2. CUDA Toolkit should be auto-detected
3. All targets available

**If CUDA not found:**
- `File → Settings → Build, Execution, Deployment → Toolchains`
- Check "Visual Studio" toolchain
- Verify CUDA path in CMakeLists.txt

---

## Troubleshooting

### Issue 1: "CMake Error: Could not find CMAKE_CUDA_COMPILER"

**Cause:** CUDA Toolkit not installed or not in PATH

**Solution (Linux/Windows):**
```bash
# Check if nvcc is installed
which nvcc  # Linux/Mac
where nvcc  # Windows

# If not found, install CUDA Toolkit
# https://developer.nvidia.com/cuda-downloads

# Add to PATH (Linux/Mac)
export PATH=/usr/local/cuda/bin:$PATH

# Reload CMake in CLion
# Tools → CMake → Reload CMake Project
```

**Solution (macOS):**
CUDA is not supported on macOS. This is expected.
- CMake will skip CUDA targets automatically
- Use cloud GPU for CUDA development
- See [MACOS_SETUP.md](MACOS_SETUP.md)

### Issue 2: "No targets found"

**Cause:** CMake configuration failed

**Solution:**
1. Check CMake tool window for errors
2. `Tools → CMake → Reload CMake Project`
3. `File → Invalidate Caches / Restart`

### Issue 3: "Unsupported GPU architecture"

**Cause:** Your GPU is too old/new for the specified `-arch` flags

**Solution:**

1. Check your GPU compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

2. Edit `CMakeLists.txt`:
```cmake
# Change this line:
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# To match your GPU:
# Kepler (K80): 37
# Maxwell (M60): 52
# Pascal (P100): 60
# Volta (V100): 70
# Turing (T4, RTX 2000): 75
# Ampere (A100, RTX 3000): 80/86
# Hopper (H100): 90
```

3. Reload CMake

### Issue 4: "Build failed with nvcc errors"

**Common Causes:**

**A. Wrong CUDA version:**
```bash
# Check CUDA version
nvcc --version

# If < 11.0, upgrade CUDA Toolkit
```

**B. Missing CUDA paths:**
```bash
# Linux/Mac - add to ~/.bashrc:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Restart CLion
```

**C. Incompatible compiler:**
- CUDA requires specific GCC versions
- CUDA 11.x requires GCC 9-11
- CUDA 12.x requires GCC 11-12

Check compatibility: [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)

### Issue 5: "Cannot run CUDA program"

**Symptoms:**
- Build succeeds
- Run fails with "CUDA error" or no output

**Solution:**

1. **Check GPU is available:**
```bash
nvidia-smi
# Should show your GPU

# If error:
sudo nvidia-smi  # Try with sudo
sudo modprobe nvidia  # Reload driver
```

2. **Check CUDA driver:**
```bash
cat /proc/driver/nvidia/version
# Should show driver version

# If missing, reinstall NVIDIA drivers
```

3. **Run with CLion console:**
- Sometimes need to run from terminal
- `Terminal` tool window in CLion
- `cd cmake-build-release`
- `./vector_add`

---

## Advanced Configuration

### Custom CMake Options

**Specify CUDA Architecture:**

`File → Settings → Build, Execution, Deployment → CMake`

Add to CMake options:
```
-DCMAKE_CUDA_ARCHITECTURES=75
```

**Enable Debug Info for CUDA:**

Add to CMake options:
```
-DCMAKE_CUDA_FLAGS="-g -G"
```

Now you can debug CUDA code with breakpoints!

### Multiple CMake Profiles

Create profiles for different scenarios:

1. `File → Settings → Build, Execution, Deployment → CMake`
2. Click `+` to add profile
3. Create profiles:

**Debug-CPU-Only:**
- Build type: Debug
- CMake options: `-DCUDA_AVAILABLE=OFF`

**Release-GPU:**
- Build type: Release
- CMake options: `-DCMAKE_CUDA_ARCHITECTURES=80`

**Profile-GPU:**
- Build type: RelWithDebInfo
- CMake options: `-DCMAKE_CUDA_FLAGS="-lineinfo"`

### External Tools Integration

**Add Nsight Compute:**

`File → Settings → Tools → External Tools → +`

- Name: `Nsight Compute`
- Program: `/usr/local/cuda/bin/ncu`
- Arguments: `$FilePath$`
- Working directory: `$FileDir$`

Now right-click any executable → `External Tools → Nsight Compute`

**Add Nsight Systems:**

- Name: `Nsight Systems`
- Program: `/usr/local/cuda/bin/nsys`
- Arguments: `profile --stats=true -o $FileNameWithoutExtension$ $FilePath$`
- Working directory: `$FileDir$`

---

## CLion Tips & Tricks

### 1. Quick Run

- **Last target:** `Shift+F10` (Mac: `⌃+R`)
- **Debug last:** `Shift+F9` (Mac: `⌃+D`)
- **Build:** `Ctrl+F9` (Mac: `⌘+F9`)

### 2. Code Navigation

- **Go to definition:** `Ctrl+B` (Mac: `⌘+B`)
- **Find usages:** `Alt+F7` (Mac: `⌥+F7`)
- **Search everywhere:** `Shift+Shift`

### 3. Debugging CUDA Code

1. Set breakpoints in `.cu` files (click left margin)
2. Run in Debug mode: `Shift+F9`
3. Use `cuda-gdb` for device code (configure in toolchain)

### 4. Performance Profiling

- **Built-in profiler:** `Run → Profile`
- **Valgrind:** Integrated on Linux
- **External tools:** Configure Nsight (see above)

### 5. CMake Integration

- **Reload CMake:** `Tools → CMake → Reload CMake Project`
- **Clean:** `Tools → CMake → Reset Cache and Reload Project`
- **CMake tool window:** `View → Tool Windows → CMake`

---

## Keyboard Shortcuts Cheat Sheet

### Build & Run
| Action | Windows/Linux | macOS |
|--------|---------------|-------|
| Build | `Ctrl+F9` | `⌘+F9` |
| Run | `Shift+F10` | `⌃+R` |
| Debug | `Shift+F9` | `⌃+D` |
| Stop | `Ctrl+F2` | `⌘+F2` |

### Navigation
| Action | Windows/Linux | macOS |
|--------|---------------|-------|
| Go to declaration | `Ctrl+B` | `⌘+B` |
| Search everywhere | `Shift+Shift` | `Shift+Shift` |
| Find file | `Ctrl+Shift+N` | `⌘+Shift+O` |
| Recent files | `Ctrl+E` | `⌘+E` |

### Editing
| Action | Windows/Linux | macOS |
|--------|---------------|-------|
| Format code | `Ctrl+Alt+L` | `⌘+⌥+L` |
| Comment line | `Ctrl+/` | `⌘+/` |
| Duplicate line | `Ctrl+D` | `⌘+D` |
| Delete line | `Ctrl+Y` | `⌘+⌫` |

---

## Integration with Version Control

CLion has excellent Git integration:

1. **Git tool window:** `View → Tool Windows → Git`
2. **Commit:** `Ctrl+K` (Mac: `⌘+K`)
3. **Push:** `Ctrl+Shift+K` (Mac: `⌘+Shift+K`)
4. **Pull:** `VCS → Update Project`
5. **View history:** Right-click file → `Git → Show History`

---

## Resources

### CLion Documentation
- [CLion Quick Start Guide](https://www.jetbrains.com/help/clion/clion-quick-start-guide.html)
- [CMake in CLion](https://www.jetbrains.com/help/clion/cmake-support.html)
- [CUDA Support](https://www.jetbrains.com/help/clion/cuda-support.html)

### CUDA Documentation
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CMake CUDA Support](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#cuda)

### This Repository
- [README.md](README.md) - Main documentation
- [MACOS_SETUP.md](MACOS_SETUP.md) - Mac users guide
- [CLOUD_GPU_SETUP.md](CLOUD_GPU_SETUP.md) - Cloud GPU setup

---

## Summary

### ✅ What CLion Does Automatically

1. Detects CMakeLists.txt
2. Configures CMake
3. Finds C++ compiler
4. Finds CUDA compiler (if available)
5. Creates build targets
6. Enables code completion
7. Provides debugging support

### 🎯 What You Need to Do

1. Open project: `File → Open`
2. Wait for indexing (30-60 seconds)
3. Select target from dropdown
4. Click play button ▶️

### 🚀 Next Steps

1. Try building `vector_add`
2. Run it and check output
3. Modify code and rebuild
4. Profile with Nsight tools
5. Start working on interview prep!

---

**That's it! CLion makes C++ and CUDA development incredibly smooth. Happy coding! 🎉**

For questions or issues:
- Check [Troubleshooting](#troubleshooting) section
- CLion Help: `Help → Help`
- Open an issue on GitHub
