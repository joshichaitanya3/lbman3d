# CUDA path — performance plan

`src/cuda/` owns the GPU-side timestep: kernel launches, D2H/H2D copies, and
device state. All physics is shared verbatim with the CPU path via
`CUDA_HOST_DEVICE` functions (`model.h`, `physics_helpers.h`,
`lattice_stencil.h`, `boundary_handler.h`, `offsets.h`) — see the top-level
`CLAUDE.md`. This directory does *not* fork the physics; any change to a kernel
must round-trip through the shared header.

This document is the plan for optimising the logging + export path, which
currently dominates wall time at any non-trivial save cadence.

## Baseline

ChannelConfig 64×64×128, 10000 steps, single-node GPU:

| Configuration                          | ms/step |
| -------------------------------------- | ------- |
| No logging, no export                  | 1.395   |
| Logging every 100 steps                | 2.192   |
| Logging + export every 100 steps       | 3.108   |
| Post-§1 (one CopyToHost per event)     | 2.875   |

Per-event overhead:
- Log tick (`CopyToHost` + reduction loop): ~80 ms
- Export tick (Q→director + FindDefects + VTKHDF write): ~92 ms on top

The gap between "no log" (1.395) and "log+export" (3.108) is ~1.7 ms/step
averaged over the run — that is what this plan targets.

**Measured `CopyToHost` cost.** ~23 ms per call at this grid, not the ~2 ms
originally estimated. Nine `thrust::copy` invocations move ~38 MB of D2H
traffic; `thrust::copy` on unpinned pageable memory achieves ~1.5 GB/s
effective (not the ~6 GB/s of ideal PCIe), so each field's copy is
throughput-bound *and* eats per-call launch/sync overhead. This underestimate
propagates into §1 and §3 savings below.

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
budget**, so any of the sections below that add per-cell fields are trivially
affordable at benchmark scale. The additions the defect-analysis plan
contemplates:

- `d_director_{x,y,z}` + `d_order`: +4 doubles/cell (+7 %). Max domain drops
  from ~260³ to ~253³ (a ~3 % linear shrinkage). +16 MB at 64×64×128.
- `d_def_{x,y,z}` as `uint8_t`: +3 B/cell. Max domain shift ~sub-1 %.
  +1.5 MB at 64×64×128.

Both are cheap; the linear shrinkage at the 8 GB ceiling is the only cost, and
`CheckGpuMemory` (planned in `src/mpi/CLAUDE.md`) will need to know the new
per-cell size once these land.

## Optimization plan (priority order)

### 1. Skip redundant `CopyToHost` — small, easy, prerequisite for (2)

`ActiveNematicSim::Log()` and `Export()` both call
`d_fields_.CopyToHost(fluid_, qtensor_)`, which unconditionally copies all
nine fields (rho, ux, uy, uz, qxx…qyz — see
[device_fields.cu:117-127](device_fields.cu#L117-L127)). At the default
`kLogInterval == kSaveInterval` this is a straight double copy per tick.

Change:
- Hoist `CopyToHost` out of `Log()` and `Export()` into a new
  `ActiveNematicSim::SnapshotToHost()` method that also stamps
  `host_snapshot_step_ = time_step_`.
- `main.cc` calls `SnapshotToHost()` once before Log/Export on a tick.
- Add a `HostFieldsAreUpToDate()` guard so tests calling `Log()`/`Export()` in
  isolation still work (either they call `SnapshotToHost()` themselves, or the
  guard triggers an on-demand snapshot).

Saving: **~23 ms per redundant copy × 100 events ≈ 0.23 ms/step** (measured:
3.108 → 2.875). The original ~2 ms/copy estimate was too optimistic about
`thrust::copy` throughput on unpinned pageable memory — see the note in
Baseline. Small on its own but still an order of magnitude larger than
predicted, and the decoupling is what makes (2) possible.

### 2. Overlap GPU stepping with CPU logging/export — big, medium effort

Currently a save/log tick is fully serial:
```
Step … Step   CopyToHost   Log   Export   Step … Step
                     ^                          ^
              GPU idle here              GPU idle here again
```

`SnapshotToHost` is a synchronous D2H copy, so at the moment it returns the
GPU has no in-flight work AND the host has a valid snapshot. From that moment
the device state is safe to advance — the CPU work in `Log`/`Export`/`FindDefects`
reads only the host copy, never the device. Reorder:

```
SnapshotToHost                                      ← sync, GPU idle for ~23 ms
Step() × kSaveInterval                              ← async kernel launches, CPU flies through in μs
Log, Q→director, FindDefects, Export                ← CPU work, runs concurrently with the queued kernels
[next iteration] SnapshotToHost                     ← implicitly waits for the queued kernels
```

Kernel launches on the default stream are asynchronous, so the "Step × N"
loop enqueues ~3–4 kernels per Step and returns immediately on the CPU. The
next `SnapshotToHost` is the barrier: `thrust::copy` used inside `CopyToHost`
synchronizes with the default stream, so it cannot start copying until all
queued kernels have finished.

State bookkeeping:
- Add `int host_snapshot_step_` alongside `time_step_`.
- `Step()` increments `time_step_` immediately, as today. This runs ahead of
  what the GPU has actually completed — that is fine, `time_step_` is a
  logical clock, not a "GPU state up to here is done" flag.
- `SnapshotToHost()` sets `host_snapshot_step_ = time_step_`.
- `Log()` / `Export()` report `host_snapshot_step_`, not `time_step_`.

Interaction with §1's guard: `HostFieldsAreUpToDate() == (host_snapshot_step_
== time_step_)` was fine for §1 (Log and Export fire on the same tick, second
call is a no-op), but in §2 `time_step_` intentionally races ahead of
`host_snapshot_step_` during the async Step-loop window. If `Log`/`Export`
keep calling `SnapshotToHost()` internally with the current guard they will
force a redundant sync and kill the overlap. Two safe options for §2:

- Replace the internal call with `assert(host_snapshot_step_ >= 0)` (or a
  similar sentinel) so a caller that forgot to snapshot crashes loudly
  instead of silently reading stale data. Preferred — matches the "never
  silently work with wrong data" property.
- Weaken the guard to "any snapshot ever taken" (`host_snapshot_step_ >= 0`,
  initialised to −1). Keeps the internal call as a belt-and-braces trigger,
  but a stale snapshot from an earlier tick passes silently.

Saving: with post-§1 CPU work at ~148 ms/event (measured 2.875 ms/step × 100
= 287 ms, minus ~140 ms GPU), the async model gives wall ≈ max(148, 140) =
148 ms/event ≈ **1.48 ms/step averaged**, close to the 1.395 no-logging
baseline. (Original plan predicted ~1.7 ms/step assuming a smaller §1 gain;
the larger measured §1 saving pulls the §2 estimate closer to the floor.)

### 3. Pin the host-side output buffers — small, easy (bigger win than originally scoped)

`CopyToHost` uses `thrust::copy(d_rho.begin(), …)` = unpinned D2H. Measured
throughput is ~1.5 GB/s effective (not ~6 GB/s ideal PCIe), so one call is
~23 ms rather than ~2 ms. Registering `ff.rho.data()`, `ff.ux/uy/uz`,
`qf.qxx…qyz` backing stores as pinned via `cudaHostRegister` should roughly
double the throughput and eliminate the driver's pinned-staging memcpy,
dropping the sync-copy portion to ~10–12 ms. That is ~11 ms/event × 100
events / 10000 steps = **~0.11 ms/step** — an order of magnitude bigger than
the original ~0.01 ms/step estimate.

This also improves §2's picture: post-§1+§3 CPU work drops to ~137 ms/event,
matching the ~140 ms GPU work. At that point the overlap in §2 is on the
knife edge — further CPU cuts stop shortening wall clock until GPU stepping
itself gets faster (see the note at the end of §"Interaction with §2").

Footgun: pinned memory cannot be moved, so the underlying `std::vector` must
not reallocate after registration. Register once in the `ActiveNematicSim`
constructor after `LocalGrid` is fixed, unregister in the destructor.

### 4. Instrument `FindDefects` — small, easy, prerequisite for (5)–(9)

Wrap `ComputeWindingNumbers`, `BuildConnectivityGraph`, and
`IsolateDisclinationsFromGraph` with `omp_get_wtime` (or a scoped
`cudaEvent_t` for later GPU variants). Without per-phase numbers, the
GPU-porting priorities below are guesses.

### 5. Move Q→director to GPU — medium, easy

Add `d_director_{x,y,z}` and `d_order` to `DeviceFields` (see the DRAM budget
section above for the +7 % per-cell cost). Port `QtensorToOrderDirector`
([analysis_fields.cc:59-93](../analysis_fields.cc#L59-L93)) as a CUDA kernel
— it is strictly pointwise and small (~30 flops, one `acos`, one `sqrt`), so
it will be dominated by the read of the five Q components and the write of
four output values. Register on the same default stream so §2's ordering
holds: it runs after the last `Step()` and before `SnapshotToHost` on export
ticks.

This is a prerequisite for (6) and (7).

### 6. Move `ComputeWindingNumbers` to GPU — medium, medium

[analysis/defect_finder.tpp:13-165](../analysis/defect_finder.tpp#L13-L165)
is pointwise per face: each iteration reads four director triples (16
doubles, read-only), runs a few sign flips + a final winding check, and
writes a single `uint8_t` to a unique index in `def_x` / `def_y` / `def_z`.
Zero cross-iteration dependencies; memory-bandwidth bound.

Add `d_def_{x,y,z}` (uint8, +1.5 MB at benchmark size) and a fused CUDA
kernel that launches over the union of the three face grids. On the CPU
side, the OMP version of this pass (~20 lines) is a viable stopgap if (5)/(6)
slip, but the async overlap in §2 already hides most of the current serial
cost, so **prefer going straight to GPU** rather than paying the OMP-then-CUDA
tax.

### 7. GPU-side disclination count — medium, medium

Only the *number* of disclinations enters `SimIO::Log` — the ordered
polylines are an export-only concern (see the feasibility discussion below).
"Number of disclinations" is exactly the number of connected components in
the defect graph (Hierholzer produces one path per component). So a parallel
connected-components (CC) pass over the compacted defect faces from (6)
produces the one integer the log actually needs, without ever touching the
host graph structure.

Two implementation options:
- **Approximate:** total defect-face count (`thrust::count_if` on the three
  `d_def_*` arrays). Cheap; drift from true component count is small in
  practice because typical disclination lines have many more faces than there
  are lines.
- **Exact:** parallel union-find (Shiloach-Vishkin, or the standard
  atomic-CAS variant) over the neighbourhood graph from (6). One kernel,
  final component count via a device-side reduction.

Either way the result is one D2H integer per log tick.

**What makes `kLogInterval = 1` cheap after this lands.** Per-tick cost today
is dominated by `CopyToHost` of nine full fields (~tens of MB across PCIe)
plus the domain-scale reduction loop over 7 host arrays
([sim_io.cc:83-109](../sim_io.cc#L83-L109)) — together the ~80 ms/event
overhead measured in the baseline. The formatted append to `lbm.log` is
never the bottleneck: it is a single text line on the order of microseconds.
Moving both the reductions (mass, momentum, KE, `umax`, `nematic_energy`)
and the CC count onto the device shrinks the per-tick D2H traffic from ~38 MB
to ~72 bytes (nine doubles + one int); host formats and appends. Estimated
per-tick cost after (7): **~1 ms** vs today's ~80 ms — the difference
between "per-step logging is a research tool" and "per-step logging melts
the run."

**Follow-up: revisit `std::flush(log_file_)`** at
[sim_io.cc:158](../sim_io.cc#L158). It is there for historical reasons —
worth reconsidering whether it is still required once (7) lands, because at
`kLogInterval = 1` a per-tick flush is a real syscall on the hot path even
if the byte count is tiny.

### 8. Replace `std::set<FaceId>` in `UndirectedGraph` — medium, medium

[analysis/defect_connectivity_graph.h:14](../analysis/defect_connectivity_graph.h#L14):
```cpp
std::unordered_map<FaceId, std::set<FaceId>> edge_map;
```

Even once (6) and (7) move to GPU, the CPU still builds this graph during
**export ticks** in order to run Hierholzer for the VTKHDF polylines. Every
`AddEdge` allocates a red-black tree node inside the inner `std::set`. By the
`compute_voxel` geometry each node has at most ~6 neighbours, so a
`std::vector<FaceId>` with dedup-on-insert (or a `small_vector<FaceId, 8>`)
beats the tree comfortably.

This is also the prerequisite for (9): `unordered_map` + `set` cannot be
mutated safely from multiple threads or GPU threads.

### 9. `BuildConnectivityGraph` on GPU — large, hard

Only worth doing if (4) shows `BuildConnectivityGraph` is a real chunk of the
export budget after (5)–(8) have landed. Blocked on (8). Redesign path:

1. **Compact defect faces:** parallel exclusive scan over `d_def_{x,y,z}`
   → `defect_faces[num_defects]` dense array of `FaceId`. Standard thrust
   primitive.
2. **Build adjacency:** one thread per defect face. For each candidate
   neighbour position (≤6 by geometry — see `compute_voxel`), check
   `d_def_*` and, if a defect, atomically append to a
   `neighbours[num_defects][6]` array. Two design choices:
   - **(a)** Preserve the CPU code's "each edge added once" discipline via a
     thread-per-voxel kernel that mirrors `compute_voxel`. Direct port,
     tricky to get race-free.
   - **(b)** Thread-per-defect-face, add each edge twice (F→N and N→F), then
     dedupe. Simpler kernel; 2× writes to adjacency plus a dedup pass. On
     modern GPUs, likely a net win.

`IsolateDisclinationsFromGraph` (Hierholzer) stays on the host — it is
inherently serial per connected component and only runs on export ticks
anyway.

## GPU-side defect analysis — feasibility reference

Systematic breakdown of the three current stages (referenced by (5)–(9)):

| Stage                              | Location today | GPU-portable? | Notes                                                                                                                                       |
| ---------------------------------- | -------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `ComputeWindingNumbers`            | CPU (serial)   | **Yes, trivially** | Pointwise per face; single write to a unique index; memory-bandwidth bound. One CUDA kernel per face type (or a fused kernel).                          |
| `BuildConnectivityGraph`           | CPU (serial)   | **Yes, moderate** | Requires redesign of `UndirectedGraph` storage (§8) + atomic append with dedup; ~200–300 lines of CUDA. Blocked on `std::set` removal.  |
| `IsolateDisclinationsFromGraph`    | CPU (serial)   | **No**        | Hierholzer is inherently serial per connected component. Stays on host, runs only on export ticks. **But** the *count* of components is trivial in parallel (union-find), which is all `Log` needs — see (7). |

The strategic split this enables:

| Consumer     | Needs                                                            | GPU pipeline required |
| ------------ | ---------------------------------------------------------------- | --------------------- |
| `Log()`      | Scalar `num_disclinations`                                       | (5) + (6) + (7). No graph build, no host copies of Q/director. |
| `Export()`   | Full polyline mesh (points, connectivity, cell types) for VTKHDF | (5) + (6), then a host copy of `d_def_*` + `d_director_*` + `d_order`. CPU builds graph (§8 makes this fast) + runs Hierholzer. Full pipeline stays sensible on CPU because it only fires on save ticks. |

This is why (5)+(6)+(7) is the bigger structural change than (5)+(6) alone:
it decouples log cadence from export cost, so `kLogInterval = 1` becomes
affordable at real problem sizes.

## Interaction with §2 (async overlap)

Once §2 lands, the wall-clock model is:
```
wall_per_event = max(CPU_work_per_event, GPU_work_per_event)
```
so a saving on either side only converts to wall-clock savings when it moves
`max(⋅, ⋅)`. Post-§1+§2 (measured `CopyToHost` cost, unpinned):
- CPU_work ≈ 148 ms/event (one 23 ms `CopyToHost` + Log + Q→director + FindDefects + VTKHDF)
- GPU_work ≈ 140 ms/event (100 async Step launches)
- Wall ≈ 148 ms.

Post-§1+§2+§3 (pinned `CopyToHost`):
- CPU_work ≈ 137 ms/event
- GPU_work ≈ 140 ms/event
- Wall ≈ 140 ms — GPU-bound.

Estimated CPU/GPU deltas from each subsequent step (**note:** Wall Δ column
below assumes CPU still exceeds GPU, but with the measured `CopyToHost` cost
we are already only ~8 ms of CPU slack away from being GPU-bound after §1+§2,
and past that after §3. Treat CPU-side savings past that point as
diagnostics, not wall-clock wins):

| Step                         | ΔCPU work (ms/event) | ΔGPU work (ms/event) | Wall Δ (ms/event, post-max) | Averaged over 10k steps |
| ---------------------------- | -------------------: | -------------------: | --------------------------: | -----------------------: |
| (5) Q→director on GPU        | −10                  | +2                   | −8 (clips at GPU floor)     | −0.08 ms/step           |
| (6) WindingNumbers on GPU    | −25 (est.)           | +3                   | 0 (already GPU-bound)       | 0                        |
| (7) CC count on GPU          | −20 (log-only ticks) | +2                   | depends on cadence          | biggest on `kLogInterval=1` |
| (8) Small-vector graph       | −5 to −15            | 0                    | −5 to −15                   | −0.05 to −0.15 ms/step  |
| (9) Graph on GPU             | −10 to −30 (est.)    | +3                   | −10 to −30                  | −0.10 to −0.30 ms/step  |

Numbers are estimates until (4) lands. The transformational change is (7) —
it changes what workflows are *feasible*, not just how fast the current one
is.

Note also: once CPU work drops below GPU work (~140 ms/event), further CPU
reductions do **nothing** for wall clock. At that point the bottleneck is the
GPU stepping itself, and the only remaining lever is stepping-throughput
work (kernel occupancy, memory layout, launch overhead) — a different
category of optimisation not covered here.

## GPU + MPI: does the async overlap pattern survive?

**Yes, but the amount of overlap depends on how the halo exchange is wired.**
The pattern in §2 relies on two things:

1. `SnapshotToHost` remains synchronous. This is unchanged by MPI.
2. `sim.Step()` on the GPU path can enqueue work without blocking the host
   for long. Today's single-rank GPU meets this trivially — kernel launches
   are async and there is no MPI. GPU+MPI meets it depending on the
   implementation:

| Halo-exchange flavour                                       | Per-step host block  | Overlap window preserved?                        |
| ----------------------------------------------------------- | -------------------- | ------------------------------------------------ |
| Stream-triggered MPI (MPI-4) / NVSHMEM                       | ~μs (no sync)        | Full — same as single-rank                       |
| CUDA-aware `MPI_Isend/Irecv` + batched `Waitall`             | ~100s of μs per Step | Partial — ~30 ms of CPU orchestration per block, wall clock `max(30 + 172, 140) ≈ 202 ms` vs serial `~312 ms` |
| Blocking `MPI_Sendrecv` per exchange                         | ~ms per Step         | Degraded — CPU stalls in the step-launch loop, less window for Log/Export |

**Design guidance for the imminent GPU-aware MPI work**: structure the halo
exchange to be as async-friendly as the MPI implementation allows. In order
of preference: stream-triggered ops → non-blocking Isend/Irecv with batched
Waitall → blocking Sendrecv only as a fallback. As long as `sim.Step()` on
GPU+MPI does not do per-step `cudaStreamSynchronize`s or per-exchange
blocking MPI, the pattern from §2 continues to work per rank.

Additional constraint that only shows up under MPI: if halo exchange requires
staging through host buffers (non-CUDA-aware MPI), those staging buffers must
either be their own device→host copies (not touching the `ff.rho`/`ff.ux`/…
snapshot buffers §2 uses), or the overlap breaks because the halo-exchange
copies would race with the CPU's Log/Export reads. Keeping halo exchange on
device pointers or on dedicated staging buffers avoids this entirely.

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
- **CopyToHost synchronizes with the default stream.** If a future change
  introduces additional CUDA streams (e.g., for I/O overlap beyond §2),
  ensure `SnapshotToHost` synchronizes with all streams that mutate device
  fields, not just the default.
- **Any new persistent device field (`d_director_*`, `d_order`,
  `d_def_*`, adjacency scratch) must update `CheckGpuMemory`'s per-cell
  size** — see `src/mpi/CLAUDE.md` §"GPU DRAM validation". Silently
  under-reporting the footprint makes the min-rank guidance wrong.
