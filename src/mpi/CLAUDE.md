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

### Indexing lives on `LocalGrid` (reverses the constexpr-dims decision)

`idx(x,y,z)`, `idx(x,y,z,i)` and `InDomain` become **members of `LocalGrid`**, not
free functions in `physics_helpers.h`. The free `idx()` is retired; the halo-aware
member is renamed to `idx` and used everywhere (a single-rank run is just a
`LocalGrid` with a small/zero halo, so there is no non-MPI special case).

This deliberately **undoes** the earlier optimisation recorded in
[`offsets.h`](../offsets.h) (a stateful `Grid<BC>` class was deleted when
`Params::nx/ny/nz` became `constexpr`). Uneven splits mean each rank's
`local_nx/ny/nz` are decided at `mpirun` time, so grid dimensions are now
**runtime** values again. Do not re-`constexpr` them or reintroduce a global
`idx()` — that would silently assume a single domain.

Consequences to preserve:
- **Member, not free-function-with-a-grid-param.** `grid.idx(x,y,z,i)` reads
  cleanly; a free `idx(x,y,z,grid)` would collide with the existing
  `idx(x,y,z,i)` (direction) overload.
- **No periodic modulo in `idx`.** The old free `idx` wrapped with `%`; the
  member does not. Periodicity is handled by halo exchange and by the ghost
  offsets in `offsets.h` / `boundary_handler.h` (`QWallOffset`,
  `SafeFetchAxisOffset`, …), never by `idx`. This is safe because every caller
  already guarantees in-domain coordinates (the old `%` was documented as
  "insurance"); streaming destinations are guarded by `InDomain` before they
  reach `idx`.
- **One source of truth, copied cheaply.** `ActiveNematicSim::grid_` is
  authoritative. `FluidFields`/`QTensorFields` hold a **by-value** `LocalGrid`
  (a ~7-int POD) constructed from it; solvers index through the fields' grid
  (`ff.grid.idx(...)`) so the layout always matches the allocation it came
  from. Pass `LocalGrid` **by value** into CUDA kernels; pass it **by
  `const&`** into the shared `CUDA_HOST_DEVICE` helpers to avoid per-call
  copies. Because `LocalGrid` is immutable after construction, the copies
  cannot diverge.

### GPU: `LocalGrid` must be a by-value kernel argument, never a device pointer

Before decomposition, kernels baked `Params::nx/ny/nz` into the SASS as compile-time
immediates. With runtime dims (see above), the strides now live in `LocalGrid`, and
**how it reaches the kernel decides whether `idx` is free or costs a memory load per
call**:

- **Do:** take `LocalGrid` by value as a kernel parameter (`__global__ void
  k(..., LocalGrid g)`). It's a ~7-int POD, so it lands in constant/parameter space
  and every thread reads the strides from registers — same cost as the old
  constexpr path.
- **Don't:** pass a `LocalGrid*` (or reference into device global memory) and
  dereference it inside the kernel. That turns every `g.idx(...)` into a global-memory
  read, adding a load to the hottest arithmetic in the solver.

This is the only GPU-side cost of moving dims to runtime; get the argument-passing
right and there is no measurable regression. See the root `CLAUDE.md` "Future
scalability" note for an optional single-process `constexpr` fast path (not worth
building unless a CPU interior-loop benchmark shows a real regression).

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

## PR VI implementation plan — distributed streaming & LBM halo exchange

Landing PR VI is three ordered steps (each unblocks the next):

1. **Fix pack/unpack coordinate convention.** `PackField*`/`UnpackField*` loop
   `1..local_n` and pass those straight into `grid_.halo_idx`, but `halo_idx`
   is **0-based** and adds `kHaloMPI` itself — so they currently pack the
   `[2 .. local_n+1]` band (2nd-owned through hi-ghost) instead of owned
   `[0 .. local_n-1]`, and the 4-arg LBM pack trips the `InDomain` assert at
   `x==local_nx` in Debug. Loops must become `0..local_n-1`; owned boundary =
   logical `0` / `local_n-1`; ghost = logical `-1` / `local_n`.

2. **Streaming destination dispatch (thread local `n`, keep `InDomain` local).**
   `Stream*Off` takes local `n`; drop any global-`nx` use. `InDomain` stays a
   pure local check. The *dispatch* (not `InDomain`) decides the streamed
   population's fate, three-way:
   - destination in owned cells → owned write;
   - crosses a **physical wall this rank owns** (face BC is a wall type **and**
     `offset==0` (lo) / `offset+local_n==Params::n` (hi)) → bounce-back;
   - otherwise (rank seam, or a periodic axis) → **halo-write** into the ghost.

   A periodic axis is never a physical wall, so it always takes the halo-write
   path. **Decision for a periodic axis that isn't split (`dims[d]==1`):**
   - **(A)** wrap it during streaming (mod `local_n` == global n). Then only
     split axes touch the halo, so a 1-D decomposition (e.g. n=2) produces
     **no corner ghosts and needs only the face exchange below — no sweep.**
     Costs a runtime `dims[d]>1?` branch in the dispatch (cheap, hoistable).
   - **(B)** treat every periodic axis as halo-write uniformly; the unsplit
     axis's exchange is a self-copy (`MPI_Cart_shift` returns self when
     `dims[d]==1`). Cleaner dispatch, but ≥2 axes use the halo even at n=2, so
     the corner sweep (step 3b) is required immediately.

   **Do (A) first** to get a green n=2 run that validates steps 1–2 + the face
   exchange in isolation; generalise afterwards.

3. **`ExchangeLBM` for the push scheme (post-stream).** Move `ExchangeLBM` to
   run **after** `LatticeBoltzmannStep`, every step. Push streaming writes each
   boundary cell's outgoing (crossing) populations into the ghost layer; the
   post-stream exchange ships them to the neighbour's owned boundary. Only the
   **crossing subset** of directions is exchanged per face (D3Q15, verified
   against `lattice_stencil.h`):

   | face | crossing dirs |
   |------|---------------|
   | +x   | 1, 7, 10, 11, 14 |
   | −x   | 3, 8, 9, 12, 13 |
   | +y   | 2, 7, 8, 11, 12 |
   | −y   | 4, 9, 10, 13, 14 |
   | +z   | 5, 7, 8, 9, 10 |
   | −z   | 6, 11, 12, 13, 14 |

   - **3a (face exchange, 1-D split):** send rank *r*'s +x **ghost** plane, dirs
     {1,7,10,11,14}, over owned (y,z); neighbour **assigns** them into its owned
     lo-x boundary (`x=0`) at the **same dir indices** (a pop leaving as dir 1
     arrives as dir 1); the other 10 dirs there, filled by local streaming, are
     left untouched. Symmetric for every face. This is the whole exchange for a
     single-axis split.
   - **3b (sweep, multi-axis split):** D3Q15's only diagonal populations are the
     8 full body-corners (dirs 7–14) — there are **no edge (2-component)
     populations**, so multi-axis routing is only ever a 3-hop corner case. A
     corner pop belongs to the *diagonal* rank; route it with the sequential
     axis sweep: pick one consistent axis order (x → y → z below) and at each
     axis send the ghost plane *including the transverse ghosts filled by
     earlier hops*, assigning that axis's crossing subset. dir 7 = (1,1,1) is in
     the crossing set of all three faces, so it rides three hops
     (0,0,0)→(1,0,0)→(1,1,0)→(1,1,1) to the diagonal rank's owned origin; a pop
     like dir 1 = (1,0,0) is in the +x set only, so it's consumed as owned at
     the first hop and never forwarded. (Either sweep direction works — each
      order routes the packet along a different chain of face-neighbours to the
     same destination — as long as every rank uses the same one.) **Unit-test this hard:** seed a
     single pop in one rank's corner-owned cell on a 2×2×2 decomposition and
     assert it arrives at the diagonal rank's mirror cell, same dir index, with
     every other slot untouched.

   The current `Pack/UnpackField*` pack **owned** faces (`local_n × local_n`),
   which is all the QTensor star-stencil exchange and the 1-D LBM face exchange
   need. The multi-axis LBM sweep needs widened faces, and — because push
   streaming has **already** deposited pops into face, edge, *and* corner ghosts
   before any exchange runs — the widening **decreases** along the sweep (the
   mirror of a pull fill-sweep, whose ghosts start empty and widen toward the
   last axis). Invariant: **each ghost cell is packed by the first sweep-axis on
   which it lies outside owned**, so axis `k`'s face is halo-inclusive in the
   axes swept *after* k and owned-sized in those swept *before* — which packs
   every ghost cell exactly once. With order x→y→z (`h = kHaloMPI`):
   - x-face (YZ): `(local_ny + 2h) × (local_nz + 2h)` — widened in both
   - y-face (XZ): `local_nx × (local_nz + 2h)` — widened in z only
   - z-face (XY): `local_nx × local_ny` — owned

   So VI-b adds a widened-extent LBM pack/unpack variant (or a transverse-range
   parameter) and grows `max_yz/max_xz/max_xy` accordingly. This is *not* a
   bigger-buffer shortcut for a simultaneous 6-face exchange — the sequential
   ordering is essential, since the y-hop reads x-ghost cells the x-hop just
   delivered.

   `Initialize` fills **owned cells only** — no halo faces. Push *reads* owned
   `f` and *writes* the halo, never the reverse; a boundary cell's incoming-
   crossing populations live in its own owned slots, kept valid by `Initialize`
   at step 0 and by the post-stream exchange every step after. Ghost cells of
   `f` are write-only from the kernel's view, so garbage there is harmless.

### Push vs. pull: why we stream with push

Both push and pull are **two-lattice** schemes (`f`, `f_new`; equal DRAM
footprint — collision is per-cell, so pull's "f_star" is just `f` collided
**in place**, not a third array). The difference is memory *traffic*, not
storage:

- **push** fuses collide+stream into one pass: read `f(x)` → collide in
  registers → scatter-write `f_new(x+eᵢ)`. One full-lattice R+W per step.
- **pull** must read *post-collision* `f_star` at each source `x−eᵢ`; in a
  fused kernel there is no stored `f_star` (getting it would mean re-colliding
  every neighbour), so pull is forced into **two passes** — collide (`f→f`
  in place), exchange, then stream (`f→f_new`). That's **~2× the DRAM traffic**
  of the fused push step.

For a bandwidth-bound LBM kernel that 2× is the whole cost, so **we keep push.**
The price is that the halo exchange is fiddlier (post-stream ordering, the
per-face crossing subset, and the corner sweep) — but that complexity is
one-time, isolated in `HaloExchange`, and **off the hot path**, whereas pull's
cost is recurring and on it. Pull's only advantage is exchange simplicity
(fill ghosts wholesale, cleaner forward sweep, no direction subset). Revisit
pull only if the post-stream corner sweep proves intractable in practice.

## PR roadmap

Mark each PR done as it merges to `dev`.

| PR | Status | Summary |
|----|--------|---------|
| I | done | `LocalGrid` struct; field constructors use local volume; rename `numprocs` → `kNumOMPThreads` |
| II | done | `LBM_ENABLE_MPI` CMake flag; `MPIContext` (RAII, Cartesian topology); wired as first member of `ActiveNematicSim`; MPI test suite under `tests/mpi/` |
| III | done | `HaloExchange` class: pack/unpack buffers, `ExchangeQTensor` / `ExchangePassiveStresses` / `ExchangeLBM`; single-rank no-op |
| IV | done | Wire `HaloExchange` into the three-phase timestep: before Q update, between phases 1–2, before LBM step |
| V | done | Global reductions in `SimIO`: `MPI_Allreduce` for diagnostics; parallel HDF5 export with collective MPI-IO |
| VI | — | Actual decomposition: `LocalGrid::FromMpiContext`; uneven-split arithmetic; `CheckMinSubdomainSize`; `idx()`/`InDomain` move onto `LocalGrid` (runtime local dims, no modulo wrap); loop bounds switch to local dims; pack/unpack tests now runnable |
| VII | — | GPU-aware path: CUDA-aware MPI or pinned-buffer staging; isolated to `HaloExchange`; `CheckGpuMemory`; decomposition already validated on CPU |

PRs III–V are single-rank no-ops that wire infrastructure before decomposition exists.
VI is where MPI does real distributed work on CPU — validate correctness here before
touching GPU. VII is then an isolated communication-layer swap with known-correct logic.

## Testing MPI code

MPI test executables need a custom `main` that calls `MPI_Init` before
`RUN_ALL_TESTS()` and uses `MPI_Allreduce(MPI_MAX)` to propagate per-rank
failures to a single exit code. See `tests/mpi/mpi_test_main.cc`. CTest runs
these with `${MPIEXEC_EXECUTABLE}` (set by `find_package(MPI)`) rather than
hardcoding `mpirun`. Use `ctest -V` to see individual GoogleTest names inside
the mpirun invocation.
