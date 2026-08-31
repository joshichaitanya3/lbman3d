# CUDA path — single-GPU status and NVSHMEM multi-GPU roadmap

`src/cuda/` owns the GPU-side timestep: kernel launches, D2H/H2D copies, and
device state. All physics is shared verbatim with the CPU path via
`CUDA_HOST_DEVICE` functions (`model.h`, `physics_helpers.h`,
`lattice_stencil.h`, `boundary_handler.h`, `offsets.h`) — see the top-level
`CLAUDE.md`. This directory does *not* fork the physics; any change to a kernel
must round-trip through the shared header.

This file has two purposes:

1. Record where the **single-GPU optimisation pass landed** (baseline, what was
   done, what is deliberately skipped) so we do not re-litigate those choices.
2. Lay out the **NVSHMEM-based multi-GPU roadmap** for extending the GPU path
   across ranks. That is the next active workstream and the bulk of the file.

## Single-GPU status (this workstream is closed)

### Baseline (10 000 steps, ChannelConfig 64×64×128, single-node GPU)

| Configuration                          | ms/step |
| -------------------------------------- | ------- |
| No logging, no export                  | 1.395   |
| Logging every 100 steps                | 2.192   |
| Logging + export every 100 steps       | 3.108   |
| Post-§1 (one CopyToHost per event)     | 2.875   |
| Post-§2 (async overlap of Step + Log)  | ~1.5    |

### What landed

- **§1 — one `CopyToHost` per event.** `Log()` and `Export()` no longer each
  drag nine full fields across PCIe on the same tick; a single
  `SnapshotToHost()` call in `main.cc` populates `host_snapshot_step_` and both
  read from that. Saving: 3.108 → 2.875 ms/step.
- **§2 — async overlap of GPU stepping with host logging/export.** After
  `SnapshotToHost` returns, `Step()×kSaveInterval` enqueues asynchronously on
  the default stream while the host runs `Log`/Q→director/`FindDefects`/VTKHDF
  write in parallel; the next tick's `SnapshotToHost` is the implicit barrier.
  Saving: → ~1.5 ms/step at 100-step cadence, i.e. close to the 1.395 no-log
  floor. Landed with `5e371d1 Fix clock with cuda` /
  `61d498d Run stepping kernels asynchronously while host exports/logs`.
- **§3 — pinned host output buffers (unshipped, low priority).** Registering
  the `ff.rho/ux/uy/uz` and `qf.q**` backing stores via `cudaHostRegister`
  would shave `CopyToHost` from ~23 ms to ~10–12 ms. That is ~0.11 ms/step
  averaged over the run — real but small, and since §2 already puts CPU work
  in the shadow of GPU stepping, it only matters if a future workflow makes
  the CPU side the bottleneck again. Left as a drop-in improvement, not
  scheduled.

### Deliberately not doing: defect analysis on GPU

An earlier version of this file laid out §4–§9: instrumenting `FindDefects`,
porting Q→director + `ComputeWindingNumbers` + a parallel connected-components
count to CUDA, and rewriting `UndirectedGraph`. That work is **descoped.**
Rationale:

- After §1+§2 the analysis pipeline runs concurrently with 100 GPU steps, so
  its ~150 ms cost is fully hidden at `kSaveInterval = 100`.
- The only workflow that would justify porting it is per-step logging
  (`kLogInterval = 1`), which is a diagnostic mode, not the shipped
  configuration.
- The compact device fields those steps would have added (`d_director_*`,
  `d_order`, `d_def_*`) are still worth budgeting for — see the DRAM table
  below — but nothing else in that chain is prioritised.

If per-step defect telemetry becomes a real requirement, the plan in git
history (commit prior to this rewrite) still stands as a starting point.

## Device DRAM budget

Per-cell footprint on device today (see
[device_fields.cu:47-77](device_fields.cu#L47-L77)):

| Group                                          | Fields                    | Doubles/cell |
| ---------------------------------------------- | ------------------------- | -----------: |
| LBM populations (double-buffered, D3Q15)       | `d_f`, `d_f_new`          | 2 × 15 = 30  |
| Hydrodynamic moments                            | `d_rho, d_ux, d_uy, d_uz` | 4            |
| Body force                                     | `d_force_{x,y,z}`         | 3            |
| Q-tensor (double-buffered, 5 components)       | `d_q**` + `d_q**_new`     | 2 × 5 = 10   |
| Passive symmetric stress                       | `d_Sigma_{xx,xy,xz,yy,yz}`| 5            |
| Passive antisymmetric stress                   | `d_Tau_{xy,xz,yz}`        | 3            |
| **Total**                                      |                           | **55**       |

55 doubles/cell = **440 B/cell**.

On an 8 GB card, budget ~7.2 GB usable after CUDA context + thrust scratch +
driver overhead. Max domain: 7.2 × 2³⁰ / 440 ≈ **17.6 M cells ≈ 260³**.
Representative shapes at the ceiling: 128×256×512, 256×256×256, 200×200×400.

The current 64×64×128 benchmark is 524 288 cells ≈ 230 MB — about **3 % of
budget**. NVSHMEM adds a symmetric-heap accounting layer on top of this (see
"Symmetric-heap sizing" below), but does not change the per-cell size of any
individual field.

## NVSHMEM multi-GPU roadmap

`src/mpi/CLAUDE.md` PR VII is currently listed as *"GPU-aware MPI or
pinned-buffer staging"*. We are **replacing that with NVSHMEM**. This section
is that replacement.

### Why NVSHMEM, not CUDA-aware MPI

CUDA-aware MPI would work — its two-sided `Isend`/`Irecv` semantics can be
kicked from CUDA streams — but PR VI's post-stream LBM exchange design is
awkward on top of it in one specific place: **the corner sweep** (PR VI step
3b, `src/mpi/CLAUDE.md`).

The relevant physics: D3Q15's eight body-corner populations (dirs 7–14) belong
to the *diagonal* rank on a multi-axis decomposition. Push streaming deposits
them into a corner ghost of the local rank, and they must reach the diagonal
rank's owned origin. With two-sided MPI we have to route them in a **sequential
axis sweep** — x → y → z, three hops, each hop's pack widening its face buffer
to include ghost cells filled by the previous hop:

- x-face pack: `(local_ny + 2h) × (local_nz + 2h)` — widened in y and z
- y-face pack: `local_nx × (local_nz + 2h)` — widened in z
- z-face pack: `local_nx × local_ny` — owned

This works, but the widening + strict axis ordering is real complexity, is
easy to break silently (the three invariants at the end of `src/mpi/CLAUDE.md`),
and forces the halo exchange to serialise across axes (x cannot begin until y
finishes, y cannot begin until z finishes) even though the three axes have no
data dependency at a single-hop level.

**NVSHMEM's one-sided `put` semantics collapse the sweep into a single hop.**
Each rank knows the topological ID of its diagonal neighbour (`MPI_Cart_rank`
gives it), and `nvshmemx_double_put_on_stream(dst, src, nelems, diag_pe,
stream)` writes directly into the diagonal PE's owned corner cell. No widening,
no intermediate rank, no axis ordering, no invariants about "each ghost cell
packed by the first sweep-axis on which it lies outside owned". The 8 corner
puts, 6 face puts, and (if we generalise Plan A → all-halo-write) 12 edge puts
all issue in parallel on the same stream, followed by one
`nvshmemx_barrier_all_on_stream`.

The face exchange for Q-tensor / passive stresses gains less relative to
CUDA-aware MPI — those are pure face-to-face — but still benefits from
one-sided semantics: no `Isend`/`Irecv` pairing, no `Waitall`, no receive-side
buffer allocation. Pack, put, barrier.

Additional wins that fall out of the choice:

- **Stream-triggered ops preserve §2's async overlap exactly.**
  `nvshmemx_*_on_stream` calls enqueue on the CUDA stream and return to the
  host in microseconds. That matches the "Stream-triggered MPI / NVSHMEM"
  row of the overlap table below: full overlap, per-step host block ~μs.
  CUDA-aware `Isend`/`Irecv` + `Waitall` sits at ~100s of μs per Step of host
  orchestration and degrades the overlap somewhat.
- **No host-side staging buffer.** `send_buf_/recv_buf_` in
  `halo_exchange_lbm.h`/`halo_exchange_qtensor.h` become device-side pack
  scratch on the symmetric heap. Removes an entire class of "did we forget to
  register these as pinned?" questions.
- **Single memory model on-node.** IPC over NVLink between GPUs on the same
  node; IB verbs (GPUDirect RDMA) between nodes. Same call site either way.

### Bootstrap: MPI still runs the show for topology

Keep `MPIContext` exactly as it is. MPI launches the job, builds the Cartesian
communicator, and — post-PR-VII — still handles the global reductions in
`SimIO` (PR V) plus the parallel HDF5 collective I/O.

NVSHMEM initialises **from** the existing MPI communicator:

```cpp
nvshmemx_init_attr_t attr;
attr.mpi_comm = &mpi.cart_comm;   // reuse the cart topology
nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
```

This gives:
- One PE per MPI rank, PE IDs equal to `cart_comm` ranks — so
  `mpi.neighbor_lo[d]` and the topology built by `MPI_Cart_shift` /
  `MPI_Cart_rank` remain the addresses NVSHMEM uses. No parallel rank-space
  bookkeeping.
- Bootstrap happens **after** `MPI_Init` and **after** `cudaSetDevice(0)` —
  see the note at the end of this section about GPU affinity.

Bringing NVSHMEM in this way means the CPU-MPI path (existing
`LBM_ENABLE_MPI`) and the GPU-NVSHMEM path (new `LBM_ENABLE_NVSHMEM`) are not
alternatives to each other — NVSHMEM sits *on top of* MPI on the GPU build.
The MPI-only CPU build stays a first-class citizen (that is the CI target that
validates decomposition correctness).

### Symmetric-heap sizing

NVSHMEM allocates from a symmetric heap sized identically on every PE. Only
fields that participate in halo exchange need to live there; local-only fields
can stay in ordinary `cudaMalloc` (thrust::device_vector today).

| Field group                                  | On symmetric heap? | Reason                                                       |
| -------------------------------------------- | ------------------ | ------------------------------------------------------------ |
| `d_f`, `d_f_new` (LBM populations)           | **Yes**            | Post-stream LBM exchange (ghost → neighbour's owned).        |
| `d_qxx…d_qyz`                                | **Yes**            | Q-tensor halo exchange (owned → neighbour's ghost).          |
| `d_qxx_new…d_qyz_new`                        | No*                | Written by phase 1, immediately swapped; never read across seam. \*But see LBM double-buffer note below. |
| `d_Sigma_*`, `d_Tau_*` (passive stresses)    | **Yes**            | Passive-stress halo between phases 1 and 2.                  |
| `d_rho`, `d_ux/uy/uz`                        | No                 | Diagnostic-only host copy; never crossed seam.               |
| `d_force_{x,y,z}`                            | No                 | Written and consumed locally within one step.                |

\*LBM double buffer: we exchange the *post-stream* populations, which is
`d_f_new` before the swap (or `d_f` after). Both buffers alternate the
"exchanged" role each step, so **both** `d_f` and `d_f_new` live on the
symmetric heap. `d_qxx_new` etc. by contrast are read only by the pointwise
swap; nothing crosses a seam in the `_new` copy of Q.

Per-cell symmetric heap footprint:

- 2 × 15 (populations) + 5 (Q) + 5 (Σ) + 3 (τ) = **43 doubles/cell = 344 B/cell**

That is 79 % of the per-cell budget. The other 21 % (moments, body force,
Q_new) is on the ordinary device allocator.

Symmetric heap size per PE = `HaloVolume() × 344 B + slack`. Pass this to
NVSHMEM via `NVSHMEM_SYMMETRIC_SIZE`. `CheckGpuMemory` in the MPI path
(`src/mpi/CLAUDE.md`, "GPU DRAM validation") already computes max local volume
on rank (0,0,0); extend it to add up "symmetric" and "cuda-malloc" separately
and check against `cudaMemGetInfo(free_mem)` after `nvshmem_init`.

### Halo exchange: how the three existing exchanges become NVSHMEM puts

The MPI path has three exchange calls (`ExchangeQTensor`, `ExchangePassiveStresses`,
`ExchangeLBM`). Each becomes a device-side pack + one-sided put + barrier
under NVSHMEM. In all three, packing lives in a kernel launched on the CUDA
stream so the whole exchange enqueues without touching the host.

**Q-tensor halo (owned → ghost), 5 fields, star stencil.**

```
for each of 6 face directions d:
    pack_kernel<<<grid, block, 0, stream>>>(d_qxx…d_qyz, pack_buf_d, face_geom(d))
    nvshmemx_double_put_nbi_on_stream(
        neighbour(d)_ghost_ptr, pack_buf_d, face_nelems, neighbour_pe(d), stream)
nvshmemx_barrier_all_on_stream(stream)
```

For a star stencil (Q update, `QGradientAndLaplacian`) the six face puts are
sufficient; no edges, no corners. Same semantics as the current MPI path,
one-sided.

**Passive-stress halo (owned → ghost).** Identical structure to Q-tensor,
different field list (5 Σ + 3 τ = 8 fields). Runs between phases 1 and 2 (see
top-level `CLAUDE.md` "Timestep structure").

**LBM post-stream halo (ghost → owned).** The one that gets dramatically
simpler. First a note on why there are three tiers of targets in D3Q15, since
this is easy to get wrong: D3Q15 has no edge-native populations (all
non-axis-aligned dirs are body corners, dirs 7–14). What determines whether a
given push destination is a face, edge, or corner *ghost* is where the
**source cell** sits in the owned subdomain:

| Source cell in owned subdomain          | Dir-7 destination ghost | Target PE       | CPU-MPI hops |
| --------------------------------------- | ----------------------- | --------------- | ------------ |
| Interior of a face (one coord at bdy)   | Face ghost (one axis)   | +x (etc.)       | 1            |
| Subdomain edge (two coords at bdy)      | Edge ghost (two axes)   | +x+y edge PE    | 2            |
| Subdomain corner (three coords at bdy)  | Corner ghost (three ax) | +x+y+z diag PE  | 3            |

The CPU-MPI sequential sweep handles all three tiers implicitly via the
widening rule (each ghost cell packed by the first sweep-axis on which it lies
outside owned). Under NVSHMEM's one-sided puts we address the three tiers
directly:

```
for each of 6 faces d:
    pack_kernel<<<...>>>(d_f_or_f_new, pack_buf_d,
                         crossing_dirs_for_face(d), face_geom(d))
    nvshmemx_double_put_nbi_on_stream(
        neighbour_owned_boundary_ptr, pack_buf_d,
        crossing_dirs × face_nelems, neighbour_pe(d), stream)

for each of 12 edges e:                     # body-corner dirs from
                                            # edge-owned source cells
    pack_kernel<<<...>>>(d_f_or_f_new, pack_buf_e,
                         crossing_dirs_for_edge(e), edge_geom(e))
    nvshmemx_double_put_nbi_on_stream(
        edge_neighbour_owned_ptr, pack_buf_e, …, edge_neighbour_pe(e), stream)

for each of 8 corners c:                    # body-corner dirs from
                                            # corner-owned source cells
    pack_kernel<<<...>>>(d_f_or_f_new, pack_buf_c,
                         crossing_dirs_for_corner(c), corner_geom(c))
    nvshmemx_double_put_nbi_on_stream(
        corner_neighbour_owned_ptr, pack_buf_c,
        crossing_dirs × corner_nelems, corner_pe(c), stream)

nvshmemx_barrier_all_on_stream(stream)
```

Key differences from the MPI version in `src/mpi/CLAUDE.md`:

- **No corner sweep.** Corners go directly to the diagonal PE.
- **No pack widening.** Every face packs `local_ny × local_nz` (etc.); every
  edge packs `local_n × 1`; every corner packs `1 × 1`. The transverse-range
  logic in `HaloExchangeLBM` disappears.
- **Edges become an explicit exchange step.** On the CPU-MPI path the
  edge-tier routing is implicit in the sweep (the y-hop picks up dir-7 pops
  that the x-hop deposited into an edge ghost); under NVSHMEM there is no
  sweep, so edges must be enumerated and put directly. This is more code than
  a face-only exchange, but each edge put is trivial (one column, ~2 dirs)
  and they issue in parallel with the face and corner puts.
- **Plan A no longer needed.** With one-sided puts there is no penalty to
  putting to yourself — `MPI_Cart_rank` on a `dims[d]==1` axis returns the
  same PE both directions, and NVSHMEM's put becomes a local device-side
  memcpy (no network involvement). Plan A's local-wrap-at-streaming
  optimisation can stay if profiling shows it helps; it is no longer required
  for correctness on 1-D decompositions.
- **Skip-unpack-at-physical-wall (invariant 3) is still required.** Under
  NVSHMEM this becomes: the pack kernel checks `is_wall_by_face<BC>` and
  writes zeros (or skips the slot entirely with a compaction mask) so that the
  put does not overwrite the neighbour's locally-bounced value. Same physics,
  different mechanism — the check moves from the unpack side to the pack side.

Q-tensor and passive-stress halos are new to per-step GPU work (currently
only exercised on the CPU-MPI path). LBM halo is the one that was blocked on
the corner-sweep design.

### Stream / synchronisation model

Every NVSHMEM call in the timestep is stream-enqueued on the same default
CUDA stream the kernels use:

```
default stream, one Step():
  ┌──────────────────────────────────────────────────────────────┐
  │ 0. ExchangeQTensor (put + barrier on stream)                 │
  │ 1. GpuQTensorStep kernel                                     │
  │ 2. ExchangePassiveStresses (put + barrier on stream)         │
  │ 3. GpuComputeBodyForce kernel                                │
  │ 4. GpuCollideAndStream kernel (push scheme, writes ghosts)   │
  │ 5. ExchangeLBM (put + barrier on stream)                     │
  └──────────────────────────────────────────────────────────────┘
```

Exchange-then-step for Q matches how the Q halo is thought about everywhere
else in the codebase (fill ghosts, *then* read stencils). Bootstrapping the
loop needs one `ExchangeQTensor` before the first `Step()` — either during
`ActiveNematicSim::Initialize` or as an unconditional first call in `main`'s
step loop.

The host does not touch the device between steps. `SnapshotToHost` remains
the barrier from §2: `thrust::copy` synchronises with the default stream,
which now also serialises all outstanding NVSHMEM puts.

### Overlap-preservation reminder (from the earlier version of this file)

For completeness — the section below was written pre-PR-VII and its verdict is
what motivated choosing NVSHMEM.

| Halo-exchange flavour                                       | Per-step host block  | Overlap window preserved?                        |
| ----------------------------------------------------------- | -------------------- | ------------------------------------------------ |
| Stream-triggered MPI (MPI-4) / **NVSHMEM**                   | ~μs (no sync)        | Full — same as single-rank                       |
| CUDA-aware `MPI_Isend/Irecv` + batched `Waitall`             | ~100s of μs per Step | Partial                                          |
| Blocking `MPI_Sendrecv` per exchange                         | ~ms per Step         | Degraded                                         |

NVSHMEM sits on the top row.

### Testing plan

1. **Single-node, 2 GPUs, `dims = {2,1,1}`.** IPC-only path. Poiseuille
   integration test (`tests/mpi/test_poiseuille_mpi.cc`-shaped, adapted to
   GPU). Correctness: assert bit-identical (up to reduction order) to the
   single-GPU run at low step counts, then bulk match to analytic parabola
   at 10 000 steps.
2. **Single-node, 4 GPUs, `dims = {2,2,1}`.** Exercises edge puts (in the
   (x,y) plane) and the four in-plane "corners" which are actually edge PEs
   for D3Q15. Same Poiseuille assertion.
3. **Single-node, 8 GPUs, `dims = {2,2,2}` (corner-put focused unit test).**
   Analogous to the "seed a single pop in one rank's corner-owned cell,
   assert it arrives at diagonal rank's mirror cell, same dir index, every
   other slot untouched" test called out in `src/mpi/CLAUDE.md` VI step 3b —
   but on GPU + NVSHMEM. This is the test that validates the one-hop-corner
   design; skip it and we lose the whole point of the NVSHMEM choice. If the
   cluster only offers 4 GPUs per node in the queue we can hit, the fallback
   is a two-node × 4-GPU `{2,2,2}` job — which is fine, and doubles as a
   partial multi-node smoke.
4. **Multi-node over IB.** Two nodes × 4 GPUs, `dims = {2,2,2}` or
   `{4,2,1}`. Verifies GPUDirect RDMA is picked up (i.e. that puts don't
   fall back to host-staged). `NVSHMEM_INFO=1` on the launch line prints the
   transport chosen per pair; check that IB verbs is used across the node
   seam and IPC within a node.

None of these run on the local single-GPU box — every one of them is a
cluster job. See "Development environment" below for what does run locally.

Scaling: the perf table at the top of this file is single-node; add a
multi-node counterpart once (4) is running.

### Development environment: local single-GPU box vs SLURM cluster

The local dev machine has **one** NVIDIA GPU. The multi-GPU target is a
SLURM-managed cluster with NVSHMEM support. Neither is optional — most of the
work happens locally, and the cluster is the only place that exercises the
actual transport. The split determines what can be validated where and drives
the PR structure below.

**What the local single-GPU box can cover.** NVSHMEM runs correctly with one
PE: it is a valid degenerate configuration. Every symmetric-heap allocation
still happens, every `nvshmemx_double_put_on_stream` still fires (into and out
of the same PE, becoming a device-side memcpy), every barrier still
serialises. So local dev covers:

- **Build + link.** `find_package(NVSHMEM)`, `-rdc=true`, `libnvshmem_host` /
  `libnvshmem_device` linking. This alone catches a large fraction of the
  integration bugs.
- **Correctness at `nranks = 1`.** All existing single-GPU tests must still
  pass after each of VII-a through VII-h. Any regression is a bug in how the
  NVSHMEM code path handles the self-PE case, not in cross-PE physics.
- **Pack/unpack kernel logic.** The pack kernels write into a device buffer;
  their shape and per-face indexing can be unit-tested by inspecting the
  packed buffer directly (before the put). This is the highest-leverage local
  check: a wrong pack kernel with a single PE looks fine (put target == self,
  wraparound trivially "works") but breaks catastrophically on ≥2 PEs. Add a
  unit test that packs a known field, reads the buffer back, and asserts the
  expected values face by face.
- **Same-PE put semantics.** A `put(self_ghost, self_owned, …)` is a legal
  intra-PE copy under NVSHMEM. Exercising it locally is the *only* way to
  catch pointer-arithmetic bugs (wrong offset, wrong stride, wrong direction
  subset) before hitting the cluster.

**What the local box cannot cover.**

- Real cross-PE data movement (IPC via NVLink or IB verbs).
- The GPU affinity path (VII-b): with one GPU, `cudaSetDevice(local_rank)`
  reduces to `cudaSetDevice(0)`, so the derivation from
  `MPI_COMM_TYPE_SHARED` is untested until at least 2 ranks per node.
- The corner-single-hop test (VII-g's headline correctness check): needs a
  2×2×2 decomposition = 8 PEs, or a scaled-down 2×2×1 = 4 PEs. Cluster only.
- GPUDirect RDMA transport selection (`NVSHMEM_INFO=1` verifies this): needs
  a multi-node job.

**Practical workflow.**

1. **Local dev loop.** Iterate on kernel logic, pack/unpack unit tests, and
   the `nranks = 1` regression suite on the local box. This is where the bulk
   of debugging happens — turnaround is seconds, no queue wait, `cuda-gdb`
   works normally. `mpirun -n 1 ./bin` is the launch command; NVSHMEM
   initialises with one PE and no complaints.
2. **Cluster dev loop for cross-PE code.** Push to git, `sbatch` a short
   two-GPU job on a single node for VII-e/f/g. Target queue turnaround under a
   few minutes for interactive iteration; longer batch jobs for the actual
   Poiseuille integration test. Keep a `scripts/slurm/` (or similar) directory
   with reusable submission templates so a "run this on 2 GPUs" doesn't
   require rewriting an sbatch header each time.
3. **Cluster validation for VII-h.** Multi-node jobs with 2×2 or 2×2×2
   layouts, plus the scaling perf run. These are less frequent (once per PR
   candidate) so queue latency is tolerable.

**Local NVSHMEM install.** NVSHMEM ships with the CUDA Toolkit ≥ 12.2 as a
separate package (`nvshmem` in the CUDA archive, or via `apt` on Debian/Ubuntu
where NVIDIA provides `libnvshmem-dev`). Install locally so the build compiles
and `nranks = 1` runs execute. If a local install is not viable for some
reason, guard the NVSHMEM code path with `#ifdef LBM_ENABLE_NVSHMEM` so the
non-NVSHMEM GPU build (current shipped path) still compiles — but avoid
letting the guarded and unguarded paths drift by keeping the guard tight and
adding a CI job that builds *without* `LBM_ENABLE_NVSHMEM` even after the
feature lands.

**Cluster access constraint on the PR structure.** The PRs below are ordered
so that each one can be *developed* on the local box, and only the
correctness-on-≥2-PEs assertions require cluster time. VII-a, VII-c, VII-d
land purely on local single-PE regression. VII-b needs a two-rank cluster run
to actually validate the affinity logic (local single-GPU cannot distinguish
"correct" from "silently wrong"). VII-e/f/g each need a two-PE Poiseuille
assertion from the cluster in addition to the local `nranks = 1` regression.
VII-h is cluster-only. Batch cluster work when possible: land VII-a → VII-d
locally, then submit VII-b's affinity test alongside VII-e's exchange test in
one cluster session.

### Build integration

- **CMake option:** `LBM_ENABLE_NVSHMEM` (default `OFF`), implies
  `LBM_ENABLE_MPI` (bootstrap dep) and `SIM_WITH_CUDA`. Cannot combine with a
  CPU-only build.
- **Find NVSHMEM:** `find_package(NVSHMEM REQUIRED)` (available in the CUDA
  Toolkit ≥ 12.2 as `nvshmem-config.cmake`). Link `nvshmem::nvshmem_host` for
  the host translation units and `nvshmem::nvshmem_device` for kernels
  compiled with `nvcc`.
- **Compile flag:** kernels that call NVSHMEM device APIs need `-rdc=true`
  (relocatable device code). Isolate them to a small file (probably one new
  translation unit `nvshmem_halo.cu`) to avoid slowing down the rest of the
  compile.
- **Launcher:** must use `mpirun` (or `srun`) with GPU-per-rank binding. The
  existing MPI launch guidance at the end of `src/mpi/CLAUDE.md` extends: add
  `--map-by ppr:1:node:PE=T,gpu` or similar so each rank gets one GPU and
  `T = kNumOMPThreads` cores. On single-node dev machines,
  `CUDA_VISIBLE_DEVICES=0,1 mpirun -n 2 …` with `cudaSetDevice(rank %
  local_ranks)` inside `InitializeComputeBackend` picks the right device per
  rank.

### GPU affinity (small but load-bearing)

`device_fields.cu:10` currently hardcodes `cudaSetDevice(0)`. This must
become `cudaSetDevice(local_rank_on_node)` **before** `nvshmemx_init_attr` —
NVSHMEM binds to whatever device is current at init time, and getting this
wrong silently pins every PE to GPU 0, causing catastrophic contention with
no error message. Compute local rank from the MPI world with
`MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`.

### Interaction with CPU-MPI CI

The MPI-only CPU build stays the primary CI target for decomposition
correctness. NVSHMEM builds are opt-in and run in a separate CI job on the
GPU runner (no CI system without GPUs can exercise this path). Any bug that
reproduces on both paths gets debugged on CPU-MPI first — no GPU debugger
tax, and the failure mode is the physics/BC/decomposition logic, not the
transport.

### PR sequencing (extends the roadmap in `src/mpi/CLAUDE.md`)

Replaces "PR VII — CUDA-aware MPI or pinned-buffer staging" with a
multi-step VII:

| PR    | Summary |
|-------|---------|
| PR    | Summary                                                                                                                                                                                                                       | Dev environment |
|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| VII-a | `LBM_ENABLE_NVSHMEM` CMake option, `find_package(NVSHMEM)`, `-rdc=true` on one isolated TU, empty stub kernel that just links against `libnvshmem`.                                                                            | Local (single-PE build + link only). |
| VII-b | GPU affinity: `cudaSetDevice(local_rank_on_node)` in `InitializeComputeBackend`; local-rank derivation from `MPI_COMM_TYPE_SHARED`.                                                                                            | Local build; **cluster** to validate (needs ≥2 ranks/node). |
| VII-c | NVSHMEM init from `MPIContext::cart_comm`; symmetric heap sizing via extended `CheckGpuMemory`; PE-ID sanity check (`nvshmem_my_pe() == mpi.rank`).                                                                            | Local (single-PE run exercises init path). |
| VII-d | Move `d_f`, `d_f_new`, `d_qxx…d_qyz`, `d_Sigma_*`, `d_Tau_*` from `thrust::device_vector` to `nvshmem_malloc` allocations. All existing kernels keep working — these are still device pointers with the same layout. Correctness: existing 1-GPU tests must pass unchanged. | Local (regression on `nranks = 1`). |
| VII-e | `ExchangeQTensor` on NVSHMEM (star-stencil face-only). Two-GPU Poiseuille integration test.                                                                                                                                   | Local pack unit tests + `nranks = 1` regression; **cluster** for two-PE assertion. |
| VII-f | `ExchangePassiveStresses` on NVSHMEM (same shape as VII-e).                                                                                                                                                                   | Same split as VII-e. |
| VII-g | `ExchangeLBM` on NVSHMEM: face puts first, then edge, then corner. Single-hop corner test lands here.                                                                                                                         | Local pack unit tests; **cluster** for 2×2×1 (edge) and 2×2×2 (corner) assertions. |
| VII-h | Multi-node IB smoke test; scaling perf run; retire the CPU-MPI corner-sweep code path (dead once NVSHMEM is the shipping GPU-MPI implementation, but keep the CPU sweep — that is the CI target).                              | **Cluster only** (multi-node IB). |

VII-d is the pivot: after it lands, the halo exchanges in VII-e/f/g can be
developed and tested one at a time, each landing as a working two-GPU build
with progressively more of the exchange on NVSHMEM. If VII-g proves harder
than expected, the fallback is to run VII-e/f on NVSHMEM and keep LBM on
CUDA-aware MPI temporarily — the transports coexist trivially.

## Design invariants (do not regress)

- **`Log()` and `Export()` operate on a self-consistent host snapshot.** The
  snapshot is taken while the GPU has no in-flight work; kernels launched
  afterwards write only device buffers, never the host copies.
- **`host_snapshot_step_ <= time_step_` always.** Anything reading a step
  number must be explicit about which one it wants. `time_step_` is the
  logical clock (may be ahead of GPU-committed state); `host_snapshot_step_`
  is what the host buffers correspond to.
- **The async overlap depends on host buffers being read-only after the
  snapshot.** If a future feature needs a kernel to read from host state
  after the snapshot (e.g., a host callback that writes back to device),
  §2's ordering breaks and must be redesigned.
- **`CopyToHost` synchronises with the default stream.** If a future change
  introduces additional CUDA streams (e.g., for I/O overlap beyond §2, or a
  dedicated NVSHMEM communication stream), ensure `SnapshotToHost`
  synchronises with all streams that mutate device fields, not just the
  default. Under NVSHMEM this means either keeping puts on the default
  stream, or `nvshmemx_barrier_all_on_stream(default) + comm_stream sync`
  before the `thrust::copy`.
- **Any new persistent device field (`d_director_*`, `d_order`,
  `d_def_*`, adjacency scratch, NVSHMEM pack buffers) must update
  `CheckGpuMemory`'s per-cell size** — see `src/mpi/CLAUDE.md` §"GPU DRAM
  validation". Under NVSHMEM the check splits into two: symmetric-heap
  footprint (from the "Symmetric-heap sizing" table above) and
  regular-heap footprint. Silently under-reporting either makes the
  min-rank guidance wrong.
- **NVSHMEM PE IDs equal MPI cart_comm ranks.** Do not maintain a parallel
  PE↔rank map. The bootstrap in VII-c enforces this at init time; anything
  that breaks the equality (e.g., a rebased NVSHMEM init that does not use
  `NVSHMEMX_INIT_WITH_MPI_COMM`) is a correctness bug, not a style
  preference — `MPI_Cart_shift` addresses would silently target the wrong
  PE.
- **Only halo-exchanged fields belong on the symmetric heap.** Local-only
  fields (`d_rho`, `d_ux/uy/uz`, `d_force_*`, `d_qxx_new` etc.) stay on
  ordinary `cudaMalloc`. Moving them to the symmetric heap wastes memory
  and, more subtly, invites future code to `nvshmem_put` them, which would
  be a physics bug (their values are step-local and rank-local).
