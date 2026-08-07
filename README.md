# lbman3d

[![Build](https://github.com/joshichaitanya3/lbman3d/actions/workflows/build.yml/badge.svg)](https://github.com/joshichaitanya3/lbman3d/actions/workflows/build.yml)
[![Unit Tests](https://github.com/joshichaitanya3/lbman3d/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/joshichaitanya3/lbman3d/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/joshichaitanya3/lbman3d/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/joshichaitanya3/lbman3d/actions/workflows/integration-tests.yml)
[![MPI Tests](https://github.com/joshichaitanya3/lbman3d/actions/workflows/mpi-tests.yml/badge.svg)](https://github.com/joshichaitanya3/lbman3d/actions/workflows/mpi-tests.yml)

Lattice Boltzmann Method-based solver for 3D Active Nematics

The flow equation is solved using a D3Q15 scheme. The Q-tensor equation is solved using a simple finite-difference scheme with an optional upwinding scheme for the advective derivative.

> **v0.2.0**: CPU MPI parallelisation with domain decomposition is now available. See [MPI (CPU only, v0.2.0)](#mpi-cpu-only-v020). GPU-aware MPI is planned for v0.3.0.

## Dependencies

| Dependency | Version | Notes |
|---|---|---|
| CMake | ≥ 3.23 | Build system |
| C++ compiler | C++23 (CPU) / C++20 (CUDA) | GCC 13+ or Clang 17+ recommended; nvcc does not support C++23, so the standard drops to C++20 automatically when building with CUDA |
| OpenMP | — | Usually bundled with the compiler |
| HDF5 | any recent | C library only; **parallel HDF5** required for MPI builds (see below) |
| CUDA Toolkit | optional | Enables the GPU-accelerated path; auto-detected by CMake if present (see `-DLBM_FORCE_CPU` under [Building](#building)) |
| MPI | optional | Required for the CPU MPI-parallel path; see [MPI (CPU only, v0.2.0)](#mpi-cpu-only-v020) |

### Installing HDF5

**Debian / Ubuntu**
```bash
sudo apt install libhdf5-dev
```

**Fedora / RHEL**
```bash
sudo dnf install hdf5-devel
```

**macOS (Homebrew)**
```bash
brew install hdf5
```

For MPI builds, install the **parallel HDF5** variant instead. On Debian/Ubuntu:
```bash
sudo apt install libhdf5-openmpi-dev libopenmpi-dev
```
(there's also `libhdf5-mpich-dev` if you prefer MPICH.) On other platforms, install an MPI implementation (OpenMPI or MPICH) and the corresponding parallel HDF5 package.

CMake prefers the parallel HDF5 when `-DLBM_ENABLE_MPI=ON` is set, but on distros that keep both serial and parallel HDF5 installed side by side (Debian/Ubuntu, Fedora), CMake may pick the serial one first and abort with a hint. Point it at the parallel install with `-DHDF5_ROOT=<path>` — see the [MPI section](#mpi-cpu-only-v020) for concrete paths.

## Configuration

There are two files you need to edit before running a simulation. Everything else under `src/` is library code that does not need to be touched for typical use.

### `src/sim_config.h` — boundary conditions, discretisation, time loop

This is the main entry point. Define your boundary condition, pick the Q advection scheme, and set how long to run:

```cpp
// Use a built-in preset:
using SimBC = ChannelConfig;

// Or define a custom configuration (here, a slit configuration):
struct MyConfig {
    using XLo = WallSpec<Neumann, SpecularReflection>;
    using XHi = WallSpec<Neumann, SpecularReflection>;
    using YLo = WallSpec<Periodic, Periodic>;
    using YHi = WallSpec<Periodic, Periodic>;
    using ZLo = WallSpec<Periodic, Periodic>;
    using ZHi = WallSpec<Periodic, Periodic>;
    static constexpr std::string_view name = "SlitFreeSlip";
};
using SimBC = MyConfig;

inline constexpr Advection kQAdvection = Advection::Centred;

inline constexpr int kNumSteps     = 1000001;
inline constexpr int kSaveInterval = 1000;    // write output every N steps
```

Note the confined axis above is **x**, not z. `idx` makes `z` the slowest-varying index and the OpenMP loop in every solver phase runs over `z`, so putting the short axis on `z` couples the usable thread count to the plate separation. Both shipped slit presets (`SlitConfig`, free-slip; `SlitNoSlipConfig`, no-slip) confine along x for that reason, as does `ChannelConfig`.

**Q advection scheme** (`kQAdvection`):

| Value | Behaviour |
|---|---|
| `Advection::Centred` | Second-order central differences. Imposes a cell-Péclet constraint, `|u| ≤ 2·GAMMA·L` |
| `Advection::Upwind` | First-order upwinding. Lifts that constraint, at the cost of a numerical Q diffusivity `|u|·DX/2` |

Centred is the default and is what all benchmark numbers refer to. Upwinding buys stability at high activity or high `LAMBDA` — measured on a channel at 20×100×100, it raises the reliable `LAMBDA` ceiling from about 0.5 to 0.7 — but be aware of the price: the correction has the same form as the elastic term `GAMMA·L·∇²Q`, so it enlarges the *effective* elastic constant to `L_eff = L + |u|/(2·GAMMA)`. At `|u| = 0.05` that is 2.2·L, which silently moves a nominal activity number of 17.5 to roughly 11.7. See the comment block next to `kQAdvection` for the full measured envelope.

**Available Q-tensor wall types** (first argument of `WallSpec`):

| Type | Behaviour |
|---|---|
| `Periodic` | Periodic stencil |
| `Neumann` | Zero-flux (`∂Q/∂n = 0`) |
| `Anchoring<S, theta, phi>` | Strong anchoring: order parameter `S`, director angle `theta` and `phi` (radians) |

**Available velocity wall types** (second argument of `WallSpec`):

| Type | Behaviour |
|---|---|
| `Periodic` | Periodic streaming |
| `NoSlip` | Mid-point bounce-back, zero wall velocity |
| `MovingWall<Ux, Uy, Uz>` | Mid-point bounce-back with imposed wall velocity |
| `SpecularReflection` | Free-slip (mirror reflection of normal component) |

**Built-in presets** defined in `src/boundary.h`:

| Preset | Description |
|---|---|
| `FullyPeriodicConfig` | All walls periodic (alias: `PeriodicBC`) |
| `ChannelConfig` | Periodic in Z, no-slip Neumann walls in X, Y |

Two more slit presets live in `src/sim_config.h` itself, since they are simulation setups rather than reusable BC building blocks:

| Preset | Description |
|---|---|
| `SlitConfig` | Periodic in Y, Z; free-slip (`SpecularReflection`) Neumann plates in X |
| `SlitNoSlipConfig` | Periodic in Y, Z; no-slip Neumann plates in X |

Prefer `SlitNoSlipConfig` unless you specifically want free-slip. Free-slip plates exert no tangential stress, so with `MU = 0` the system has no momentum sink at all and energy accumulates at box scale until the run diverges — measured at step 16000 for ν = 2/3, and *earlier* (step 11000) at ν = 1.833, so raising viscosity does not rescue it.

### `src/params.h` — physical and numerical parameters

| Parameter | Description |
|---|---|
| `nx`, `ny`, `nz` | Grid dimensions. For the slit/channel presets the confined axis is `x`, so `nx` is the plate separation |
| `kNumOMPThreads` | Number of OpenMP threads |
| `DT` | Lattice time step. **Must be 1.0**, enforced by `static_assert` — see below |
| `TAUF` | LBM relaxation time, and the only handle on viscosity: `ν = c_s²(TAUF/DT − 1/2)` |
| `L` | Elastic constant in the gradient free energy (one-constant approximation; `L = 2K` for Frank constant `K` at `S_eq = 1/3`) |
| `A`, `B`, `C` | Landau free-energy coefficients. With `A = 0` the equilibrium order parameter is `S_eq = −B/(2C)` |
| `GAMMA` | Inverse rotational viscosity |
| `LAMBDA` | Flow-aligning parameter |
| `ALPHA` | Activity coefficient |
| `MU` | Linear friction coefficient |
| `NOISE` | Amplitude of initial Q-field noise, absolute (not relative to `S_eq`) |
| `kDebugLogging` | `true` → log every step, save LBM fields; `false` → log every `kSaveInterval` steps |
| `kTrackNematicEnergy` | Track the total nematic free energy in the log. Off by default; it is an extra O(N) pass, useful mainly as a sanity check (with `ALPHA = 0` it must decrease monotonically) |

**On `DT`:** streaming advances exactly one cell per `LatticeBoltzmannStep`, so the LBM timestep is one lattice unit by construction, while `DT` only scales the Q-tensor right-hand side and the Guo forcing. Any `DT ≠ 1` silently decouples the two — at `DT = 0.05`, Q was advected at `u/20`. To take smaller *effective* Q steps, scale `GAMMA`, `L` and `ALPHA` instead of `DT`.

**Initial condition:** a uniform director along `z` (a periodic direction for the slit presets) at the bulk equilibrium order parameter, derived from `A`, `B`, `C` rather than hardcoded, plus per-cell uniform noise of amplitude `NOISE` on every component.

## Building

```bash
cmake -B build
cmake --build build -j$(nproc)
```

If CMake finds a CUDA compiler, it automatically builds the GPU-accelerated path. Pass `-DLBM_FORCE_CPU=ON` to force a CPU-only build even when CUDA is available:

```bash
cmake -B build -DLBM_FORCE_CPU=ON
cmake --build build -j$(nproc)
```

The GPU path is fully featured: it supports all boundary condition types and computes the full passive and active stress contributions. It is physics-equivalent to the CPU path.

### MPI (CPU only, v0.2.0)

CPU-side MPI parallelisation is fully wired: domain decomposition via
`MPI_Dims_create` (with a post-permute to align the largest split with the
largest axis), cross-rank halo exchange for the Q-tensor / passive-stress
stencils and the LBM populations (including a sequential axis sweep to route
D3Q15 body-diagonal directions across multi-axis decompositions), and
parallel HDF5 output. Physics-equivalent to the serial CPU path — mass
conservation is verified to machine precision over thousands of steps under
`mpi_poiseuille_np2`, and the 2×2×2 corner-sweep test (`mpi_corner_sweep_np8`)
delivers body-diagonal pops to the correct diagonal rank.

**GPU-aware MPI is not in v0.2.0** — multi-GPU / multi-node CUDA support is
planned for v0.3.0. GPU builds continue to work in the single-machine mode
from v0.1.0.

**Build**:

```bash
cmake -B build-mpi -DLBM_ENABLE_MPI=ON -DLBM_FORCE_CPU=ON
cmake --build build-mpi -j$(nproc)
```

On distros that keep both serial and parallel HDF5 installed, point CMake at
the parallel install explicitly:

```bash
# Ubuntu with libhdf5-openmpi-dev:
cmake -B build-mpi -DLBM_ENABLE_MPI=ON \
      -DHDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/openmpi

# Fedora / RHEL: check `pkg-config --variable=libdir hdf5-openmpi` for the
# right path.
```

**Run**:

```bash
mpirun -n 4 ./build-mpi/main
```

For hybrid MPI+OpenMP on a single node, the default OpenMPI binding pins
each rank to a single core, which defeats OpenMP within each rank. Give each
rank a chunk of cores explicitly:

```bash
# 2 ranks, 8 OpenMP threads/rank, each rank pinned to 8 physical cores:
mpirun -n 2 --map-by socket:PE=8 --bind-to core \
       -x OMP_PROC_BIND=close -x OMP_PLACES=cores \
       ./build-mpi/main
```

See [`src/mpi/CLAUDE.md`](src/mpi/CLAUDE.md) for the full MPI implementation
notes: three-invariant post-stream LBM exchange contract, launcher/runtime
pitfalls, and guidance for hybrid P/E-core desktop CPUs.

## Testing

The test suite (GoogleTest, fetched automatically by CMake) builds by default alongside `main`/`benchmark`:

```bash
cmake -B build -DLBM_FORCE_CPU=ON
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

The integration tests run thousands of solver steps, so they're slow without optimizations. For a quicker run, add `-DCMAKE_BUILD_TYPE=Release`:

```bash
cmake -B build -DLBM_FORCE_CPU=ON -DCMAKE_BUILD_TYPE=Release
```

Note this also strips the `idx()` bounds-check `assert()`, which one unit test (`TestIdx.InDomainEdgeCases`) deliberately triggers — expect that single test to fail under a Release build; it's not a real regression, just that test needing a Debug build to exercise the assert it's checking. The CI `unit-tests` workflow always uses Debug for this reason, and `integration-tests` skips the unit test binary entirely (via `-DLBM_BUILD_UNIT_TESTS=OFF`) so it can use Release freely.

See [`tests/CLAUDE.md`](tests/CLAUDE.md) for the full test catalogue, the per-test `params.h`-shadowing mechanism, and the CI branch strategy (`feature/*` → `dev` runs build + unit tests; `dev` → `main` additionally runs the integration suite).

## Running

Create an output directory, then run the binary from the repo root:

```bash
mkdir -p data
./build/main
```

Output is written to `data/` every `kSaveInterval` steps. Progress and divergence checks are logged to `lbm.log` in the working directory.

## Output

Output is written as VTKHDF (HDF5), one file per frame.

| File | Format | Contents |
|---|---|---|
| `data/lbm_<step>.vtkhdf` | VTKHDF (HDF5) | `rho`, `ux/uy/uz`, `order`, `director` — all fields in one file, readable by ParaView 5.10+. With `kDebugLogging` it also carries the raw D3Q15 populations `f0`..`f14` |
| `data/disclinations_<step>.vtkhdf` | VTKHDF (HDF5) | Disclination lines as a polyline mesh (non-MPI builds only) |
| `lbm.log` | text | Simulation parameters, and per-step mass / momentum / kinetic-energy / nematic-energy / disclination-count diagnostics |

### Reading the output in Python

The point-data arrays are stored `[nz, ny, nx]` — the **reverse** of `params.h`'s `nx, ny, nz`. Reshaping from `params.h` by eye gives a silently transposed array, and if two of the dimensions are equal it will not fail loudly:

```python
# VTKHDF PointData is [nz, ny, nx] — the reverse of params.h's nx, ny, nz.
import h5py
with h5py.File('data/lbm_00050.vtkhdf') as f:
    ux = f['/VTKHDF/PointData/ux'][:]        # shape (nz, ny, nx)
    n  = f['/VTKHDF/PointData/director'][:]  # shape (nz, ny, nx, 3)
```

Without `h5py`, the `h5dump` binary that ships with HDF5 is enough — `h5ls -r file.vtkhdf` lists the datasets and their shapes, and `h5dump -d /VTKHDF/PointData/ux -b LE -o ux.bin file.vtkhdf` writes raw little-endian doubles for `numpy.fromfile`.

## Monitoring a live run

`monitor.py` tails `lbm.log` and plots the per-step diagnostics as the run progresses. Launch it from the same working directory as the simulation (i.e. the one that contains `lbm.log`):

```bash
python3 monitor.py
```

Requires `numpy` and `matplotlib`. The window refreshes every 2 s and shows:

- **Kinetic Energy** (log scale) with **Max |u|** overlaid on a twin y-axis — useful for eyeballing the cell Péclet number `|u|·DX/(2·GAMMA·L)` against the centred-scheme stability bound, and for catching divergence early.
- **Net momentum** components `Px`, `Py`, `Pz` — should stay near zero for periodic setups; a linear drift signals a body-force imbalance.
- **Disclination count** over time.
- **Info panel** with the run's parameters (as logged at startup) and a live mass-conservation indicator (`|ΔM/M₀|`).

Pass `--total_energy` to swap the top panel from KE to the total energy (KE + nematic free energy). This requires the run to have been built with `Params::kTrackNematicEnergy = true` in [src/params.h](src/params.h), and is the right view for passive benchmarks (`ALPHA = 0`), where total energy must decrease monotonically.

The parser is backwards-compatible with older log formats — fields introduced later (Total Energy, disclination count, Max |u|) are optional and simply omitted from the plot if absent.

## Visualization with Paraview

The full sequence of output VTKHDF files can be visualized conveniently by running the included script from this directory:

```
/path/to/paraview/binary --script=visualize_in_paraview.py
```

You will need to set the following parameters within that script according to your simulation:
```
DATA_DIR   = './data'
OUTPUT_DIR = os.path.join(DATA_DIR, 'frames')

NX, NY, NZ = 20, 100, 100
```
