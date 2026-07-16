# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A hybrid 3D active nematic hydrodynamics solver using D3Q15 Lattice Boltzmann for the flow field and explicit finite-differences for Q-tensor dynamics (modified Beris-Edwards model). The primary audience is computational and theoretical physicists — keep code physics-readable and avoid unnecessary abstractions.

## Build commands

```bash
# Standard build (auto-detects CUDA)
cmake -B build
cmake --build build -j$(nproc)

# Force CPU-only
cmake -B build -DLBM_FORCE_CPU=ON
cmake --build build -j$(nproc)

# Release build (enables -Ofast, disables bounds-check asserts in idx())
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Using the gcc15 preset (sets compiler paths + enables compile_commands.json)
cmake --preset gcc15
cmake --build build -j$(nproc)

# Run the simulation (output goes to data/)
mkdir -p data
./build/main

# Run the benchmark binary
./build/benchmark
```

C++ standard: C++23 for CPU-only builds; C++20 when CUDA is enabled (nvcc does not support C++23). The CMakeLists.txt handles this automatically.

There is no test suite yet — one needs to be added.

## User-facing configuration files

Only two files need to be touched for a typical simulation:

- **`src/sim_config.h`** — boundary conditions (`SimBC`) and time loop parameters (`kNumSteps`, `kSaveInterval`). Each of the six faces gets a `WallSpec<QBC, UBC>` where `QBC` controls the Q-tensor FD stencil and `UBC` controls LBM streaming/bounce-back. Built-in presets: `FullyPeriodicConfig`, `ChannelConfig`.
- **`src/params.h`** — grid dimensions (`nx`, `ny`, `nz`), LBM relaxation (`TAUF`), Landau coefficients (`A`, `B`, `C`), elastic constant (`L`), activity (`ALPHA`), flow-alignment (`LAMBDA`), rotational viscosity (`GAMMA`), friction (`MU`), and logging flags.

Everything else under `src/` is solver code. The BC interface in `sim_config.h` is a key design goal — any refactor must preserve or improve its readability, never complicate it.

## Architecture

### Solver hierarchy

```
ActiveNematicSim<BC>          (active_nematic.h)
├── LbmSolver<BC>             (lbm_solver.h / .tpp)
├── QTensorSolver<BC>         (qtensor_solver.h / .tpp)   ← subclassable
├── DeviceSolver<BC>          (device_solver.h / cuda/)   ← no-op struct on CPU
├── DeviceFields              (device_fields.h / cuda/)   ← no-op struct on CPU
├── FluidFields               (fluid_fields.h)
├── QTensorFields             (qtensor_fields.h)
└── SimIO                     (sim_io.h)
```

`ActiveNematicSim` is the top-level orchestrator. `LbmSolver` is **intentionally decoupled from Q-tensor physics** — the only coupling point is `FluidFields::fx/fy/fz`. `QTensorSolver::Step()` writes those force fields before `LbmSolver::LatticeBoltzmannStep()` reads them. You can build an entirely different simulation (e.g., a Poiseuille flow) by using `LbmSolver` directly.

To inject a custom activity model, subclass `QTensorSolver<BC>` and override `SetActiveStressAndComputeBodyForce`.

### Timestep structure (CPU and GPU)

Each timestep runs three sequential phases. On CPU (`qtensor_solver.tpp`) these are OpenMP parallel loops; on GPU (`src/cuda/kernels.cu`) they are consecutive kernel launches on the default stream (which provides the required barriers between phases):

1. **Q update + passive stress** (`StepAndSetupBodyForce` / `GpuQTensorStep`) — Beris-Edwards FD update for Q; computes passive stress tensor `P`; initialises backflow contribution to `ff.fx/fy/fz` (the `H:∇Q` term).
2. **Active stress + body force** (`SetActiveStressAndComputeBodyForce` / `GpuComputeBodyForce`) — adds active stress (`∇·(αQ)`) and `∇·P` (passive stresses) and linear friction to `ff.fx/fy/fz`. **Phases 1 and 2 cannot be merged**: phase 2 reads `∇·P` at neighbours written by phase 1.
3. **LBM** (`LatticeBoltzmannStep` / `GpuCollideAndStream`) — BGK collision + streaming + boundary reconstruction.

The pointwise physics (`PointwiseStepAndSetupBodyForce`, `PointwiseSetActiveStressAndComputeBodyForce`) and all boundary/ghost helpers (`QGradientAndLaplacian`, `VelocityGradientTensor`, `PassiveStressDivergence`, `HandleBoundaryPoint`) are `CUDA_HOST_DEVICE` functions shared verbatim between both paths — there is no fork of the physics.

### GPU path

The GPU path (`src/cuda/`) is a full-featured implementation supporting all boundary condition types and passive stresses. It is physics-equivalent to the CPU path.

### Boundary condition dispatch

BCs are resolved entirely at **compile time** via template parameters. The BC type flows from `SimBC` in `sim_config.h` through `ActiveNematicSim<BC>` down into every solver and helper.

Two complementary mechanisms live in `offsets.h` and `boundary_handler.h`:

- **`offsets.h`** (`QXoff/QYoff/QZoff`) — Neumann-only (zero-gradient) index clamp. Used only where a Dirichlet ghost is not needed: the passive-stress-tensor `P` gradient in `SetActiveStressAndComputeBodyForce`.
- **`boundary_handler.h`** — wall-type-aware ghost values (not indices) for: LBM streaming (`StreamXoff`), velocity gradients (`VelocityGradientTensor`, used in the Beris-Edwards co-rotation and strain rate), and Q gradients/Laplacians (`QGradientAndLaplacian`, used everywhere Q derivatives appear). Mirrors `offsets.h`'s design but with proper Dirichlet (anchoring) and SpecularReflection handling.

Ghost methods (`SafeFetchAxisOffset`, `QAxisGhostPair`, `VelocityAxisGhostPair`) operate on **values, not raw memory indices**, which is deliberate: it avoids out-of-domain raw pointer arithmetic and keeps the design compatible with future MPI ghost-cell exchange.

### Memory layout

`idx(x, y, z)` returns a flat row-major offset: `z * ny * nx + y * nx + x`.

`idx(x, y, z, i)` has **different layouts on host vs device**:
- Host: direction `i` is **fastest** (all 15 directions for one point are contiguous) — matches per-point inner loops.
- Device: direction `i` is **slowest** (all grid points for one direction are contiguous) — matches per-direction CUDA kernels.

This split is invisible to callers since `__CUDA_ARCH__` branches inside the same function body; do not change it.

### `CUDA_HOST_DEVICE` functions

`model.h`, `physics_helpers.h`, `lattice_stencil.h`, `boundary_handler.h`, and `offsets.h` are shared between host and device. Any function marked `CUDA_HOST_DEVICE` must compile cleanly under both `g++` and `nvcc`. Avoid `std::` calls that lack device support (e.g., `std::clamp` without `--expt-relaxed-constexpr`).

## Future scalability

The code is designed with multi-GPU/multi-node MPI in mind. When adding features that touch neighbour lookups:
- Use the ghost-value functions in `boundary_handler.h` (e.g., `SafeFetchAxisOffset`, `QAxisGhostPair`) rather than raw index arithmetic — these are the correct seam for future MPI halo-exchange insertion.
- Do not hardcode assumptions that the entire domain fits in one contiguous block of memory.
