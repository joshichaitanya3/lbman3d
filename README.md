# lbman3d
Lattice Boltzmann Method-based solver for 3D Active Nematics

The flow equation is solved using a D3Q15 scheme. The Q-tensor equation is solved using a simple finite-difference scheme.

## Dependencies

| Dependency | Version | Notes |
|---|---|---|
| CMake | ≥ 3.23 | Build system |
| C++ compiler | C++23 | GCC 13+ or Clang 17+ recommended |
| OpenMP | — | Usually bundled with the compiler |
| HDF5 | any recent | C library only |
| [kokkos/mdspan](https://github.com/kokkos/mdspan) | `stable` | Fetched automatically by CMake |

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

## Configuration

There are two files you need to edit before running a simulation. Everything else under `src/` is library code that does not need to be touched for typical use.

### `src/sim_config.h` — boundary conditions and time loop

This is the main entry point. Define your boundary condition and set how long to run:

```cpp
// Use a built-in preset:
using SimBC = ChannelConfig;

// Or define a custom configuration (here, a slit configuration):
struct MyConfig {
    using XLo = WallSpec<Periodic, Periodic>;
    using XHi = WallSpec<Periodic, Periodic>;
    using YLo = WallSpec<Periodic, Periodic>;
    using YHi = WallSpec<Periodic, Periodic>;
    using ZLo = WallSpec<Neumann, SpecularReflection>;
    using ZHi = WallSpec<Neumann, SpecularReflection>;
    static constexpr std::string_view name = "SlitFreeSlip";
};
using SimBC = MyConfig;

inline constexpr int kNumSteps     = 1000001;
inline constexpr int kSaveInterval = 1000;    // write CSV output every N steps
```

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
| `ChannelConfig` | Periodic in X, no-slip Neumann walls in Y, Z |

### `src/params.h` — physical and numerical parameters

| Parameter | Description |
|---|---|
| `nx`, `ny`, `nz` | Grid dimensions |
| `numprocs` | Number of OpenMP threads |
| `DT` | Lattice time step |
| `TAUF` | LBM relaxation times (shear and forcing) |
| `RHO` | Initial lattice density |
| `L` | Frank elastic constant |
| `A`, `B`, `C` | Landau free-energy coefficients |
| `GAMMA` | Inverse rotational viscosity |
| `LAMBDA` | Flow-aligning parameter |
| `ALPHA` | Activity coefficient |
| `MU` | Linear friction coefficient |
| `NOISE` | Amplitude of initial Q-field noise |
| `kDebugLogging` | `true` → log every step, save LBM fields; `false` → log every `kSaveInterval` steps |

## Building

```bash
cmake -B build
cmake --build build -j$(nproc)
```

The `mdspan` library is fetched from GitHub automatically on the first configure step — no manual download needed.

## Running

Create an output directory, then run the binary from the repo root:

```bash
mkdir -p data
./build/main
```

Output is written to `data/` every `kSaveInterval` steps. Progress and divergence checks are logged to `lbm.log` in the working directory.

## Output

There are two output formats provided: CSV and VTKHDF. It can be selected via a simple Enum in the main simulation loop.

| File | Format | Contents |
|---|---|---|
| `data/rho_<step>.csv` | CSV | Density field |
| `data/ux_<step>.csv` | CSV | x-velocity |
| `data/uy_<step>.csv` | CSV | y-velocity |
| `data/uz_<step>.csv` | CSV | z-velocity |
| `data/qxx_<step>.csv` | CSV | Q-tensor component Qxx |
| `data/qxy_<step>.csv` | CSV | Q-tensor component Qxy |
| `data/qxz_<step>.csv` | CSV | Q-tensor component Qxz |
| `data/qyy_<step>.csv` | CSV | Q-tensor component Qyy |
| `data/qyz_<step>.csv` | CSV | Q-tensor component Qyz |
| `data/delta_m_<step>.csv` | CSV | Local density change since last export |
| `data/lbm_<step>.vtkhdf` | VTKHDF (HDF5) | All fields in a single file, readable by ParaView 5.10+ (together with the LBM fields when `kDebugLogging` is `true`)|
| `lbm.log` | text | Simulation parameters and per-step mass/momentum diagnostics |

## Visualization with Paraview

The full sequence of output VTKHDF files can be visualized conveniently by running the included script from this directory:

```
/path/to/paraview/binary --script=visualize_in_paraview.py
```

You will need to set the following parameters within that script according to your simulation:
```
DATA_DIR   = './data'
OUTPUT_DIR = os.path.join(DATA_DIR, 'frames')

NX, NY, NZ = 100, 100, 15
```
