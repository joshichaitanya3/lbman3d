# resource_estimator

Prints host-RAM and device-DRAM budgets for a given grid size, and can
generate a ready-to-submit SLURM batch script for a target HPC cluster.
Built automatically alongside `main` and `benchmark`.

## Usage

```
resource_estimator NX NY NZ             CPU single-rank memory estimate
resource_estimator N                    CPU cube  (NX = NY = NZ = N)
resource_estimator gpu NX NY NZ         GPU device-DRAM + host snapshot
resource_estimator gpu N                GPU cube
resource_estimator slurm [NX NY NZ]     Generate SLURM job script
resource_estimator slurm [N]            Same with cube domain
resource_estimator --list-clusters      List known cluster configurations
```

The first positional argument is the mode token (`gpu`, `slurm`, or absent
for CPU).  Dimensions match the axis convention in `params.h` — X is the
plate-separation axis, Z is slowest-varying.

### Slurm options

| Option | Default | Description |
|--------|---------|-------------|
| `--cluster <id>` | `perlmutter` | Target cluster (`perlmutter`, `generic`, …) |
| `--account <acct>` | `ACCOUNT` | HPC project account number |
| `--ranks <N>` | `1` | Total MPI ranks |
| `--omp-threads <T>` | `kNumOMPThreads` | OMP threads/rank (CPU builds only) |
| `--time <HH:MM:SS>` | cluster maximum | Walltime override |
| `--gpu-variant <40g\|80g>` | `40g` | GPU memory tier (GPU builds only) |
| `--output <file>` | `job.slurm` | Output file path |

When `NX NY NZ` are omitted in `slurm` mode the binary uses the grid dims
compiled in from `params.h` (see *CMake integration* below).

## CMake integration

### Compile-time baking

`CMakeLists.txt` parses `src/params.h` (or the file in `LBM_PARAMS_DIR`) at
**configure time** and bakes these values into
`build/generated/resource_estimator_config.h`:

| CMake variable | What it captures |
|---|---|
| `LBM_COMPILED_NX/NY/NZ` | `Params::nx/ny/nz` |
| `LBM_COMPILED_OMP_THREADS` | `Params::kNumOMPThreads` |
| `LBM_BUILD_GPU/MPI/NVSHMEM` | Build-mode flags |
| `LBM_DEFAULT_CLUSTER` | `LBM_CLUSTER` cache variable |
| `LBM_DEFAULT_ACCOUNT` | `LBM_NERSC_ACCOUNT` cache variable |

This means `resource_estimator slurm` (no explicit dims) uses the same grid
that `main` was compiled for, and the generated SLURM script reflects the
actual build flags without any extra arguments.

### Auto-generating job.slurm at build time

Pass `-DLBM_CLUSTER=<id>` to cmake to have the build system write
`build/job.slurm` automatically each time `resource_estimator` is rebuilt:

```bash
# CPU+MPI build targeting Perlmutter with 256 ranks
cmake -B build \
  -DLBM_FORCE_CPU=ON \
  -DLBM_ENABLE_MPI=ON \
  -DLBM_CLUSTER=perlmutter \
  -DLBM_NERSC_ACCOUNT=m1234 \
  -DLBM_MPI_RANKS=256
cmake --build build -j$(nproc)
# → build/job.slurm is ready to scp and sbatch

# GPU+NVSHMEM build, 8 ranks, 80 GB A100s
cmake -B build \
  -DLBM_ENABLE_MPI=ON \
  -DLBM_ENABLE_NVSHMEM=ON \
  -DLBM_CLUSTER=perlmutter \
  -DLBM_NERSC_ACCOUNT=m1234 \
  -DLBM_MPI_RANKS=8 \
  --gpu-variant 80g
cmake --build build -j$(nproc)
# → build/job.slurm uses #SBATCH -C gpu&hbm80g
```

You can also call `resource_estimator slurm` manually at any time to
regenerate the script with different parameters without rebuilding.

### OMP thread efficiency warning

If `kNumOMPThreads` does not divide the cluster's `cores_per_node` evenly,
the tool emits a warning and embeds a suggestion in both stderr and the
generated script comment block:

```
warning: kNumOMPThreads=10 → 12 tasks/node × 10 threads = 120/128 cores used (8 idle)
         For full core utilisation set kNumOMPThreads=8 or 16 in params.h (both divide 128 evenly)
```

This applies to CPU builds only.  For GPU builds the OMP thread count is
irrelevant to node packing.

## Known clusters

| `--cluster` id | System | CPU partition | GPU partition |
|---|---|---|---|
| `perlmutter` | NERSC Perlmutter | 128 cores / 512 GiB, queue `regular` | 4× A100 (40 or 80 GiB), queue `regular` |
| `generic` | Generic HPC | 128 cores / 256 GiB placeholder | — |

### Adding a new cluster

Edit `kClusters[]` in `src/resource_estimator.cc`.  Each entry needs:

```cpp
{
    "mycluster",             // id used with --cluster
    "My Cluster (HPC)",      // human-readable name
    // Module setup — newline-separated; nullptr = omit
    "module load PrgEnv-gnu",         // setup_base_cpu
    "module load PrgEnv-gnu cuda",    // setup_base_gpu
    "module load openmpi",            // setup_mpi   (appended if MPI build)
    nullptr,                          // setup_nvshmem
    // CPU partition
    { "regular", "cpu", { 128, 256.0, 0, 0.0 }, "48:00:00" },
    // GPU partition  (set gpus_per_node=0 if CPU-only cluster)
    { "gpu",     "gpu", {  64, 128.0, 4, 40.0 }, "24:00:00" },
    // 80 GB GPU constraint (nullptr = no variant)
    nullptr,
},
```

No CMake changes needed — `kClusters` is a plain static array.

## Sample output

### CPU

```
$ ./build/resource_estimator 20 100 100

Resource estimate — 20 x 100 x 100  [CPU, single rank]
  200000 cells  |  55 doubles/cell  |  440 B/cell

  Field group                         doubles    Size
  --------------------------------------------------------------------
  LBM: f, f_new                       30 d/cell    45.78 MiB
  LBM: rho, ux, uy, uz                 4 d/cell    6.104 MiB
  LBM: fx, fy, fz                      3 d/cell    4.578 MiB
  Q: q (current)                       5 d/cell    7.629 MiB
  Q: q_new (scratch)                   5 d/cell    7.629 MiB
  Q: Sigma (passive symmetric)         5 d/cell    7.629 MiB
  Q: Tau (passive antisymmetric)       3 d/cell    4.578 MiB
  --------------------------------------------------------------------
  Total host RAM                      55 d/cell    83.92 MiB
```

### SLURM (CPU+MPI, Perlmutter)

```
$ ./build/resource_estimator slurm 20 100 100 --ranks 256 --account m1234
SLURM script written → job.slurm
  Cluster  : NERSC Perlmutter (HPE Cray EX)
  Grid     : 20 x 100 x 100  |  Build: CPU+MPI
  Memory   : 83.92 MiB/rank — fits
  Layout   : 256 ranks  →  2 nodes × 128 tasks/node
```

```bash
# job.slurm contents:
#!/bin/bash
# lbman3d SLURM job script — auto-generated by resource_estimator
# Cluster  : NERSC Perlmutter (HPE Cray EX)
# Grid     : 20 x 100 x 100  (200000 cells)
# Build    : CPU+MPI
# Memory   : 83.92 MiB/rank
# Layout   : 256 ranks  →  2 nodes × 128 tasks/node
# OMP      : 1 thread/rank  (128/128 cores/node used)
#
#SBATCH --job-name=lbman3d_20x100x100
#SBATCH -A m1234
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH -t 24:00:00
#SBATCH --output=lbman3d-%j.out

module load PrgEnv-gnu
module load cray-mpich

srun ./build/main
```

### SLURM (GPU+NVSHMEM, Perlmutter)

```
$ ./build/resource_estimator slurm 64 64 128 --ranks 8 --account m1234
SLURM script written → job.slurm
  Layout   : 8 ranks  →  2 nodes × 4 tasks/node
```

```bash
#!/bin/bash
# Build    : CUDA+NVSHMEM+MPI
# Layout   : 8 ranks  →  2 nodes × 4 tasks/node
#
#SBATCH -A m1234
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH --output=lbman3d-%j.out

module load PrgEnv-gnu
module load cudatoolkit
module load cray-mpich
module load nvshmem

srun --ntasks-per-node=4 --gpus-per-task=1 ./build/main
```

## What the numbers mean

### Field breakdown (55 doubles/cell = 440 B/cell)

| Field | Source struct | Doubles/cell |
|---|---|---|
| `f`, `f_new` | `FluidFields` | 2 × 15 = 30 |
| `rho`, `ux`, `uy`, `uz` | `FluidFields` | 4 |
| `fx`, `fy`, `fz` | `FluidFields` | 3 |
| `q` (current, 5 components) | `QTensorFields` | 5 |
| `q_new` (scratch) | `QTensorFields` | 5 |
| `Sigma` (passive symmetric stress) | `QTensorFields` | 5 |
| `Tau` (passive antisymmetric stress) | `QTensorFields` | 3 |

The GPU path allocates the same field set twice: once in `DeviceFields`
(device DRAM) and once in the host `FluidFields`/`QTensorFields` that
`CopyToHost` writes into.

### `[sym-heap]` annotation (GPU output only)

Fields marked `[sym-heap]` must live on the NVSHMEM symmetric heap when
`LBM_ENABLE_NVSHMEM` is enabled — they are exchanged across ranks during
halo operations.

Symmetric-heap total: 43 doubles/cell (`d_f`/`f_new`, `d_q`, `d_Sigma`, `d_Tau`).
Regular device total: 12 doubles/cell (`d_q_new`, moments, body force).
See `src/cuda/CLAUDE.md` → "Symmetric-heap sizing" for the full rationale.

### GPU card fit table

The per-card rows show the maximum domain the card can hold (device DRAM only,
not counting the host snapshot) and the largest cube that fits.  The
`~0.8 GiB overhead` deduction accounts for the CUDA context, driver reserved
memory, and thrust scratch.
