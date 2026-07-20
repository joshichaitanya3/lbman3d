# tests/CLAUDE.md

Guidance for Claude Code when working in this directory.

## Overview

This directory holds the unit and integration test suite for lbman3d. Tests are built with GoogleTest (fetched via `FetchContent`) and registered with CTest via `gtest_discover_tests()`. The suite is opt-in: the root CMake must be configured with `-DLBM_BUILD_TESTS=ON`.

## Directory structure

```
tests/
├── CLAUDE.md
├── CMakeLists.txt               # defines lbm_add_test(); fetches GoogleTest; add_subdirectory(unit/integration)
├── params/                      # per-scenario params.h files that shadow src/params.h at compile time
│   ├── unit/params.h            # 8×8×8, ALPHA=0, numprocs=1  — used by all unit tests
│   ├── poiseuille/params.h      # 32×16×4, ALPHA=0, tuned TAUF
│   ├── qtrelax/params.h         # 16×16×16, ALPHA=0, A=0 B=-0.3 C=0.3
│   ├── bc_check/params.h        # 16×8×8, ALPHA=0
│   └── active/params.h          # 24×12×12, ALPHA=0.04
├── unit/
│   ├── CMakeLists.txt
│   ├── test_lattice_stencil.cc
│   ├── test_feq_moments.cc
│   ├── test_bgk_collide.cc
│   ├── test_offsets.cc
│   ├── test_boundary_handler.cc
│   ├── test_anchored_q.cc
│   └── test_idx.cc
└── integration/
    ├── CMakeLists.txt
    ├── test_poiseuille.cc        # A1: Poiseuille flow
    ├── test_qt_relaxation.cc     # A2: Q-tensor Beris-Edwards relaxation
    ├── test_bc_combinations.cc   # A3: BC wall/anchoring verification
    └── test_active_mass.cc       # A4: Active nematic mass conservation
```

## The params-shadowing mechanism

All physics parameters live in `src/params.h` as `inline constexpr` — there is no runtime configuration. Tests that need a different grid size, ALPHA, or physical parameters get their own `params.h` file under `tests/params/<scenario>/`. The `lbm_add_test()` helper in `tests/CMakeLists.txt` puts that directory **first** in the include path, before `src/`. All `src/` headers use `#include <params.h>` (angle brackets), so the compiler consults the `-I` list rather than the file-relative directory — the test-specific version wins. Test source files should `#include "params.h"` (quoted is fine here since test files are not in `src/`), not `#include "params_unit.h"` or any other non-standard name. Each test params file may use its own unique include guard.

```cmake
# tests/CMakeLists.txt
function(lbm_add_test target params_dir)
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/params/${params_dir}  # shadows src/params.h
        ${CMAKE_SOURCE_DIR}/src
        ${HDF5_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PRIVATE GTest::gtest_main ${HDF5_C_LIBRARIES})
    gtest_discover_tests(${target})
endfunction()
```

Example: `lbm_add_test(unit_tests unit)` → compiler finds `tests/params/unit/params.h` before `src/params.h`.

## CMake targets

### Unit tests

All unit test `.cc` files are compiled into **one executable** (`unit_tests`) sharing `params/unit/params.h`. They also link `src/fluid_fields.cc` and `src/qtensor_fields.cc` (needed for `FluidFields`/`QTensorFields` constructors that some tests invoke).

Do **not** link `sim_io.cc`, `vtkhdf_writer.*`, or the defect-finder — they pull in HDF5 write paths that are not needed for unit tests. HDF5 headers are still needed at configure time because `find_package(HDF5 REQUIRED)` runs unconditionally in the root.

### Integration tests

Each integration test is a **separate executable** because each needs a different `params.h`:

| Executable            | params dir    | Solver classes used                          |
|-----------------------|---------------|----------------------------------------------|
| `test_poiseuille`     | `poiseuille`  | `LbmSolver<PoiseuilleBC>` only               |
| `test_qt_relaxation`  | `qtrelax`     | `QTensorSolver<FullyPeriodicConfig>` only    |
| `test_bc_combinations`| `bc_check`    | Both solvers, multiple BC types              |
| `test_active_mass`    | `active`      | Both solvers, `FullyPeriodicConfig`          |

Integration tests construct `LbmSolver` and `QTensorSolver` directly — **not** `ActiveNematicSim`. This avoids linking `SimIO`, HDF5 writers, and the defect finder.

## Unit test catalogue

### `test_lattice_stencil.cc` — D3Q15 compile-time invariants

No params needed (only touches `lattice_stencil.h` arrays). Tests:

- **WeightsSum**: `∑ w[i] == 1.0`
- **OppInvolution**: `opp[opp[i]] == i` for all i
- **OppNegatesVelocity**: `ex[opp[i]] == -ex[i]` etc. for i=1..14
- **SpecZCorrectness**: `specZ[i]` reverses `ez` only; `ex[specZ[i]] == ex[i]`, `ey[specZ[i]] == ey[i]`, `ez[specZ[i]] == -ez[i]`. Repeated for `specY`, `specX`.
- **MissingDirectionSets**: cross-check `missingYLo` etc. against directions with `ey[i] > 0` etc.

### `test_feq_moments.cc` — `Feq` and `ComputeMoments`

- **ZeroVelocityFeq**: at `u=(0,0,0)`, `Feq = w[i]*rho` for all i
- **FeqZerothMoment**: `∑ Feq = rho` for several (rho, u) pairs
- **FeqFirstMoment**: `∑ ex[i]*Feq = rho*ux` etc.
- **MomentsRoundtrip**: construct `f = Feq(m,...)`, verify `ComputeMoments` recovers rho, u to `1e-12`
- **GuoHalfStepCorrection**: `f = Feq(rho, u=0)`, force `(F_x,0,0)` → recovered `ux = 0.5*F_x*DT/rho`

### `test_bgk_collide.cc` — `PointwiseBGKCollide`

- **FixedPointAtEquilibrium**: `BGK(feq, feq, 0) == feq` (since `omega + omega_prime = 1`)
- **RelaxationDirection**: `|f_star - feq| < |f - feq|` for `f ≠ feq`
- **ForcingLinearity**: `BGK(f,feq,2F) - BGK(f,feq,F) == BGK(f,feq,F) - BGK(f,feq,0)`
- **ForcingAntisymmetry**: `forcing_i + forcing_{opp[i]} ≈ 0` for face-normal directions (Guo operator antisymmetry)

### `test_offsets.cc` — `QWallOffset`, `QAxisOffset`, `QXoff/QYoff/QZoff`

- **InteriorPassthrough**: `QAxisOffset(x, ±1, n) == x±1` for non-boundary x
- **NeumannClamp**: `QWallOffset<Neumann>(0, -1, n) == 0`; `QWallOffset<Neumann>(n-1, +1, n) == n-1`
- **PeriodicWrap**: `QWallOffset<Periodic>(0, -1, 8) == 7`; `QWallOffset<Periodic>(7, +1, 8) == 0`
- **AxisDispatch**: For `FullyPeriodicConfig`, `QXoff(0, -1) == nx-1`. For `ChannelConfig` (Periodic X, Neumann Y/Z), `QXoff(0,-1) == nx-1` but `QYoff(0,-1) == 0`.

### `test_boundary_handler.cc` — streaming offsets, velocity ghosts, gradients

- **StreamWallOffsetPeriodic**: wraps in-domain
- **StreamWallOffsetNoSlip**: returns out-of-domain index (triggers `!InDomain`)
- **VelocityGhostNoSlip**: `ghost = -v_boundary` (wall velocity = 0)
- **VelocityGhostMovingWall**: `ghost = 2*Ux - v_boundary`
- **VelocityGhostSpecularNormal**: normal component reflected: `ghost = -v_boundary`
- **VelocityGhostSpecularTangential**: tangential component preserved: `ghost = +v_boundary`
- **InteriorGhostPairPassthrough**: at non-wall points, `VelocityAxisGhostPair` returns raw neighbors unchanged
- **CentralDifferenceInterior**: linear velocity profile → gradient exactly `0.01` (central difference exact for linear)
- **GradientAtNoSlipWall**: verify analytical formula at x=0 with NoSlip Lo wall

### `test_anchored_q.cc` — `AnchoredQ`, `QGhost`, `QGradientAndLaplacian`

- **AnchoredQTraceless**: `xx + yy + zz == 0`
- **AnchoredQMagnitude**: `HalfTrQ2 == S²/3`
- **AnchoredQAlignedAlongZ**: at `theta=0, phi=0` → `xx=yy=-S/3`, all off-diag=0
- **QGhostNeumann**: returns `q_boundary` unchanged
- **QGhostAnchoring**: returns `2*anchored_comp - q_boundary`
- **GradientUniformField**: all derivatives zero for constant Q, any BC, including at boundary nodes
- **GradientLinearField**: exact gradient `0.01` for `qxx(x) = 0.01*x` with periodic BC
- **LaplacianQuadratic**: `∑ 7-point stencil = 2*scale` for `qxx(x) = scale*x²` (stencil exact for degree-2 polynomials)

### `test_idx.cc` — index function

- **Uniqueness**: all `nx*ny*nz` values of `idx(x,y,z)` are distinct (fill a `std::set`, check size)
- **RowMajorFormula**: `idx(x,y,z) == z*ny*nx + y*nx + x`
- **HostDirectionLayout**: `idx(x,y,z,i) == idx(x,y,z)*15 + i` on host
- **InDomainEdgeCases**: `(0,0,0)` true; `(-1,0,0)` false; `(nx,0,0)` false; `(nx-1,ny-1,nz-1)` true

## Integration test designs

### A1: `test_poiseuille.cc` — Poiseuille flow (LBM-only)

**Purpose**: Verify `LbmSolver` alone converges to the analytical parabolic velocity profile and conserves mass.

**BC**: Periodic X/Z, NoSlip Y — defined as a local `PoiseuilleBC` type in the test file.

**Setup**: Construct `FluidFields ff` and `LbmSolver<PoiseuilleBC> lbm`. Call `lbm.Initialize(ff)`. Set `ff.fx = F_x = 1e-5` everywhere, `ff.fy = ff.fz = 0`. Do **not** run `QTensorSolver` — it would overwrite `ff.fx`.

**Run**: 5000 steps, maintaining constant `ff.fx` every step.

**Verification**:
1. **Parabolic profile** (bounce-back midpoint convention): `u_x_analytical(y) = F_x/(2η) * (y+0.5) * (ny-0.5-y)` where `η = (1/3)*(TAUF - DT/2)`. Compare y-slice averages to `EXPECT_NEAR(..., 1e-4)`.
   - Note: the simple formula `y*(H-y)` without the half-cell offset is wrong here — it produces a systematic bias at the walls.
2. **Profile symmetry**: `u_x_mean[y] == u_x_mean[ny-1-y]` to `1e-8`.
3. **Mass conservation**: `∑ρ_final / ∑ρ_initial ≈ 1.0` to `1e-10`.

**Expected runtime**: < 1 second.

### A2: `test_qt_relaxation.cc` — Q-tensor Beris-Edwards relaxation

**Purpose**: Verify the Beris-Edwards dynamics drives the nematic free energy monotonically downward and converges to a spatially uniform state.

**Key design choice**: Call only `QTensorSolver::StepAndSetupBodyForce` with zero velocity in `ff`. Do **not** run `LbmSolver`. Use a `ZeroActivitySolver` subclass (overrides `SetActiveStressAndComputeBodyForce` to be a no-op) — but since we skip the LBM step, the body force field is irrelevant anyway. The point is to isolate pure Q-tensor dynamics.

**Starting state**: Moderately ordered uniform Q (`qxx = 0.3`, `qyy = -0.15`, off-diagonals = 0 everywhere) with a small fixed-seed perturbation (`std::mt19937(42)`). Do **not** start from Q≈0: with A=0 the molecular field at Q=0 vanishes, so there is no driving force and the system will not leave the isotropic state.

**BC**: `FullyPeriodicConfig`.

**Run**: 2000 steps. Compute `TotalNematicFreeEnergy` every 100 steps.

**Verification**:
1. **Monotone free energy decrease**: `F[t+100] <= F[t] + 1e-10` for all checkpoints.
2. **Spatial uniformity**: `std(S over domain) / mean(S) < 0.01` at end.
3. **Convergence**: change in free energy between last two checkpoints < `1e-6`.

**Expected runtime**: ~3 seconds.

### A3: `test_bc_combinations.cc` — wall and anchoring BC verification

**Purpose**: Verify that each BC type enforces the correct physical condition.

**ZeroActivitySolver**: same subclass as A2 — override `SetActiveStressAndComputeBodyForce` to no-op, so `ff.fx/fy/fz` stays zero and the LBM sees no forcing.

**Scenarios** (separate `TEST` cases, each is a distinct compile-time BC type):

| BC Config | Verification |
|-----------|-------------|
| `FullyPeriodicConfig` | `∑ρ` conserved to `1e-10` after 500 steps |
| `ChannelConfig` (NoSlip Y/Z) | `∑ρ` conserved; after small initial `ux` perturbation damps, `ux` at y=0 and y=ny-1 walls ≈ 0 |
| Custom with `SpecularReflection` Z | `uz = 0` at wall; tangential velocity preserved in mirror |
| Custom with `Anchoring<0.5, π/4, 0>` Z faces | Q at z=0 nodes converges within 0.01 of `AnchoredQ<>()` components after 500 steps |

**Important**: Initialize with a small nonzero `ux` perturbation for the NoSlip wall test — `ux=0` trivially satisfies the wall condition at step 0.

**Expected runtime**: ~5 seconds total.

### A4: `test_active_mass.cc` — active nematic mass conservation

**Purpose**: Verify `∑ρ` is conserved to machine precision throughout a full active + LBM run.

**BC**: `FullyPeriodicConfig`. Run both solvers (full `ActiveNematicSim`-equivalent loop, but constructed manually without `SimIO`). `numprocs = 1` for determinism.

**Run**: 1000 steps. Check after every step.

**Verification**:
1. `|∑ρ_t - ∑ρ_0| / ∑ρ_0 < 1e-10` at every step.
2. `std::isfinite(∑ρ_t)` throughout (catches NaN divergence).

**Expected runtime**: ~8 seconds.

### A5 (additional): equation of state sanity

After `lbm.Initialize(ff)`, before any steps: `ff.rho[i] ≈ Params::RHO` everywhere and `ff.ux[i] ≈ 0` everywhere. Zero steps needed — tests initialization consistency.

### A6 (additional): rotational symmetry of Q-tensor dynamics

Run Q-tensor-only relaxation (no LBM, same setup as A2) twice: once with the original uniform IC, once with x↔y coordinates swapped. The free-energy trajectories must match to `1e-8`. Catches anisotropic index-order bugs in the FD stencil.

## GitHub CI

### Branch strategy

- **`feature/*` → `dev`**: unit tests required to pass
- **`dev` → `main`**: integration tests (+ unit tests) required to pass
- **`v*.*.*` tag**: full test suite runs before GitHub release is created

### `.github/workflows/unit-tests.yml`

Triggers on PRs into `dev`. Uses `Debug` build to keep `idx()` bounds-check asserts live.

```yaml
on:
  pull_request:
    branches: [ dev ]
```

Key steps: install `cmake ninja-build g++ libhdf5-dev libomp-dev`, configure with `-DCMAKE_BUILD_TYPE=Debug -DLBM_FORCE_CPU=ON -DLBM_BUILD_TESTS=ON`, build `--target unit_tests`, run `ctest -R "^unit_"`.

### `.github/workflows/integration-tests.yml`

Triggers on PRs into `main`. Uses `Release` build (`-Ofast`) so multi-thousand-step tests run fast.

```yaml
on:
  pull_request:
    branches: [ main ]
```

Key steps: same dependencies + `actions/cache@v4` for `build/_deps` (keyed on `CMakeLists.txt` hash to cache GoogleTest and fmt), configure with `-DCMAKE_BUILD_TYPE=Release`, build the four integration test targets explicitly, run `ctest --timeout 300`.

**Note**: `find_package(HDF5 REQUIRED)` in the root CMakeLists runs unconditionally at configure time, so `libhdf5-dev` is needed in both workflows even though test targets do not link `sim_io.cc`. This is a known limitation to clean up eventually by guarding the HDF5 find behind a condition.

### `numprocs` in test params

Set `numprocs = 1` in all `tests/params/*/params.h` files. GitHub-hosted runners have 2 vCPUs and OpenMP threading introduces ordering non-determinism that complicates exact-value assertions in integration tests.
