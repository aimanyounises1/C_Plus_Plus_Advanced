# ML Inference Engine

## Overview
Build a neural network inference engine optimized for CUDA.

## Goals
- Support common layers (Conv2D, Dense, ReLU, Softmax, etc.)
- Implement efficient CUDA kernels for each operation
- Model loading from common formats (ONNX)
- Batch processing support
- Integration with TensorRT (optional)

## Key Files to Create
- `main.cpp` - Entry point
- `layers/*.cu` - Individual layer implementations
- `engine.h/cpp` - Inference engine
- `model_loader.cpp` - Model loading
- `benchmark.cpp` - Performance testing

## Learning Outcomes
- Deep learning operations
- Custom CUDA kernels for ML
- Memory optimization
- Inference optimization techniques
