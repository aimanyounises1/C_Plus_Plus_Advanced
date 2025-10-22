# GPU-Accelerated Physics Simulation

## Overview
Create a particle-based physics simulation using CUDA.

## Goals
- N-body simulation with gravitational forces
- Collision detection and response
- Spatial partitioning for optimization
- Real-time visualization
- Support for thousands/millions of particles

## Key Files to Create
- `main.cpp` - Application entry
- `simulation.cu` - Physics kernels
- `particle.h` - Particle definition
- `spatial_hash.cu` - Spatial partitioning
- `visualization.cpp` - OpenGL/Vulkan rendering

## Learning Outcomes
- Parallel physics algorithms
- Spatial data structures on GPU
- CUDA-OpenGL interop
- Performance optimization
