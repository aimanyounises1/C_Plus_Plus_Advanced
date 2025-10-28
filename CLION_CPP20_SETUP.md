# CLion C++20 Setup for macOS

Your project is now configured for **C++20**! Follow these steps to complete the setup in CLion.

## ✅ What I've Already Done

Updated `CMakeLists.txt` to use C++20:
```cmake
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
```

## 📋 Steps to Complete in CLion

### Step 1: Reload CMake Project

**In CLion menu:**
- **Tools → CMake → Reset Cache and Reload Project**
- Or click the circular "Reload" icon in the CMake tool window

> **Why:** This regenerates `compile_commands.json` with `-std=c++20` flag for clangd

### Step 2: Verify the Standard Flag

Build once (⌘F9) and check the **Build** tool window. Expand a compile command and verify it contains:
```
-std=c++20
```

If you don't see it, repeat Step 1 or restart CLion.

### Step 3: Check Toolchain (Optional)

**CLion → Preferences → Build, Execution, Deployment → Toolchains**

**Option A: Use Apple Clang (Default - Recommended)**
- **Compiler:** Apple Clang (from Xcode Command Line Tools)
- **Debugger:** LLDB
- Click **Apply**

**Option B: Use Homebrew LLVM (If Installed)**
- **C Compiler:** `/opt/homebrew/opt/llvm/bin/clang`
- **C++ Compiler:** `/opt/homebrew/opt/llvm/bin/clang++`
- **Debugger:** LLDB
- Click **Apply**

### Step 4: If Editor Still Shows Warnings

Try these in order:

1. **Tools → CMake → Reset Cache and Reload Project** (again)
2. **File → Invalidate Caches / Restart... → Invalidate and Restart** (nuclear option)

### Step 5: (Optional) CMake Profile Method

Instead of editing CMakeLists.txt, you can also set the standard via CMake profile:

**Preferences → Build, Execution, Deployment → CMake**
- Select your profile (Debug/Release)
- In **CMake options** add:
```
-DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_EXTENSIONS=OFF
```
- Click **Apply**, then **Reload**

## 🎯 C++20 Features Now Available

With C++20 enabled, all these features work:

✅ **Ranges** (`std::ranges`, `std::views`)
```cpp
auto result = data | views::filter(...) | views::transform(...);
```

✅ **Concepts**
```cpp
template<typename T> requires std::integral<T>
T add(T a, T b) { return a + b; }
```

✅ **Coroutines** (`co_await`, `co_return`, `co_yield`)

✅ **Modules** (experimental)

✅ **Three-way comparison** (`<=>`)

✅ **Designated initializers**
```cpp
Point p = {.x = 10, .y = 20};
```

## 🔧 Troubleshooting

### "constexpr lambda" warnings?

Use a regular `constexpr` function instead:

**Before:**
```cpp
constexpr auto is_leap = [](int y) { /* ... */ };
```

**After (more portable):**
```cpp
constexpr bool is_leap(int y) {
    return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}
```

### Compilation errors about C++20 features?

Check the build log contains `-std=c++20`. If not:
1. Reload CMake project
2. Clear build directory: `rm -rf build/`
3. Rebuild

## 📚 Related Files

- Main config: `CMakeLists.txt`
- macOS setup: `MACOS_SETUP.md`
- Cloud GPU: `CLOUD_GPU_SETUP.md`

## ✨ Current Configuration

```
C++ Standard: C++20
CUDA Standard: C++20 (when available)
Compiler Flags: -Wall -Wextra
Release Flags: -O3 -march=native
```

---

**You're all set for C++20!** 🚀

All the modern C++ exercises in Phase 3 will now compile without warnings.
