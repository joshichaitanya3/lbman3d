# MPI implementation guide

This directory contains the GPU-aware MPI parallelisation layer for lbman3d.
The target model is **hybrid MPI+OpenMP**: each MPI rank owns one subdomain
(`LocalGrid`) and one GPU; OpenMP threads parallelise the inner loops within
each rank on CPU builds; CUDA kernels replace those loops on GPU builds.

## Key design decisions

### `LocalGrid` convention
`local_nx/ny/nz` are **owned-cell counts, without halo layers**. Halo cells are
extra and added to the allocation separately when introduced. This keeps
`LocalGrid::volume()` equal to the owned volume (useful for load-balancing and
the GPU DRAM check) and keeps `offset_x/y/z` directly usable as global
coordinate offsets. Core loops will run `1 <= x <= local_nx` once halos exist
(indices 0 and `local_nx+1` being ghost layers), not `1 <= x < local_nx-1`.

### Domain decomposition
`MPI_Dims_create` factorises the total rank count into a 3D Cartesian grid that
minimises surface-area-to-volume ratio. `MPI_Cart_create` is then called with
`reorder=1` (topology-aware rank assignment) to build `MPIContext::cart_comm`.
For uneven splits the standard block-distribution formula is used: the first
`global % n_ranks` ranks along each axis get one extra cell.

### GPU DRAM validation
The maximum local volume belongs to rank (0,0,0) since it always gets `ceil` on
every axis. `CheckGpuMemory` checks that volume against the result of
`cudaMemGetInfo` and aborts with an informative message (including the minimum
rank count needed) if it does not fit.

### `MPIContext` ownership flag
`owns_mpi_` tracks whether this instance called `MPI_Init`. The destructor only
calls `MPI_Finalize` when `owns_mpi_` is true, so `MPIContext` can be safely
constructed inside GoogleTest fixtures where the test binary's `main` already
called `MPI_Init`.

### `periods` argument
`MPIContext` takes `std::array<int,3> periods` (default `{1,1,1}`). For
non-periodic axes `MPI_Cart_shift` returns `MPI_PROC_NULL` at domain edges,
making `MPI_Sendrecv` a no-op there. `ActiveNematicSim` derives the right value
from its `BC` template parameter. `FullyPeriodicConfig` → `{1,1,1}`;
`ChannelConfig` → `{1,1,0}` (periodic in x/y, walled in z).

### Halo seam
The ghost-value functions in `boundary_handler.h`
(`SafeFetchAxisOffset`, `QAxisGhostPair`, `VelocityAxisGhostPair`) are the
correct insertion point for MPI halo exchange — they already operate on values
rather than raw indices, which makes them compatible with halo buffers without
touching the physics kernels.

## PR roadmap

Mark each PR done as it merges to `dev`.

| PR | Status | Summary |
|----|--------|---------|
| I | done | `LocalGrid` struct; field constructors use local volume; rename `numprocs` → `kNumOMPThreads` |
| II | done | `LBM_ENABLE_MPI` CMake flag; `MPIContext` (RAII, Cartesian topology); wired as first member of `ActiveNematicSim`; MPI test suite under `tests/mpi/` |
| III | — | `HaloExchange` class: posts MPI sends/receives for Q and LBM ghost layers; single-rank no-op |
| IV | — | Wire `HaloExchange` into the three-phase timestep: before Q update and before LBM step |
| V | — | GPU-aware path: CUDA-aware MPI or pinned-buffer staging; isolated to `HaloExchange` |
| VI | — | Global reductions in `SimIO`: `MPI_Allreduce` for diagnostics (mass, energy, order parameter) |
| VII | — | Actual decomposition: `LocalGrid::FromMpiContext`; uneven-split arithmetic; `CheckGpuMemory` + `CheckMinSubdomainSize`; `idx()` and loop bounds switch to local dims |

PRs III–VI can proceed without VII (single-rank no-ops throughout). VII is the last
PR and is the one that changes observable memory layout and loop bounds.

## Testing MPI code

MPI test executables need a custom `main` that calls `MPI_Init` before
`RUN_ALL_TESTS()` and uses `MPI_Allreduce(MPI_MAX)` to propagate per-rank
failures to a single exit code. See `tests/mpi/mpi_test_main.cc`. CTest runs
these with `${MPIEXEC_EXECUTABLE}` (set by `find_package(MPI)`) rather than
hardcoding `mpirun`. Use `ctest -V` to see individual GoogleTest names inside
the mpirun invocation.
