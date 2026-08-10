# Python frontend — plan

This directory will host the Python-side of lbman3d: a build-and-run driver, a
reactive monitor, and (later) pybind11 bindings that let a user orchestrate
the solvers directly. Nothing is implemented yet — this file is the plan.

The audience for this frontend is the same as for the C++ code: physicists
running simulations, not application developers. So the frontend must not
require them to learn scikit-build, ninja, mpi4py, or trame. It has to look
like `run_channel.py` (Tier 1) or `for step in sim: ...` (Tier 2).

## Why "build almost every run" is fine

Everything downstream of the solver is templated on either the grid dims
([../src/params.h](../src/params.h)) or the boundary-condition config
([../src/sim_config.h](../src/sim_config.h)); both are compile-time. The C++
side has no runtime constructor for `ActiveNematicSim(nx=…, ny=…, BC=…)` and
adding one is a large refactor that the
[../src/mpi/CLAUDE.md](../src/mpi/CLAUDE.md) explicitly rules out until a
benchmark justifies it. So the Python frontend is **a code-generation and
build orchestrator**, not a traditional bindings library.

This is not as expensive as it sounds. Release GPU builds finish in seconds
after a warm ccache, and the whole pipeline is cacheable on a hash of the
generated headers plus backend flags.

## Modernization stack

The frontend is greenfield, so it commits to modern-Python defaults from
day one. Concretely:

| Concern            | Tool                                | Why                                                                                                       |
|--------------------|-------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Env + deps + lock  | **uv**                              | Fast, one binary, deterministic `uv.lock` committed to the repo. Replaces pip / pip-tools / virtualenv.   |
| Build backend      | **scikit-build-core** (+ pybind11)  | The wheel packager for CMake-backed projects. `uv build` composes on top of it.                           |
| Lint + format      | **ruff** (optional gate)            | One tool for what black + isort + flake8 + pyupgrade used to do. Fast enough to be a pre-commit hook.     |
| Type checking      | **ty** (Astral) — pyright fallback  | Astral's checker, philosophically consistent with uv/ruff. Pyright if ty misbehaves on pybind11 stubs.    |
| Config schema      | **pydantic v2**                     | Validation at construction + free JSON serialization → a run config can be persisted next to the HDF5.    |
| Log dataframes     | **polars**                          | Clean, typed API. Volume is small so perf is a bonus, not the reason.                                     |
| CLI                | **typer** + **rich**                | Type-hint-driven subcommands; nice terminal output at zero extra effort.                                  |
| File watching      | **watchfiles**                      | Rust-backed, from the Pydantic ecosystem. Smaller install than watchdog and faster.                       |
| Interactive plots  | **plotly**                          | Marimo embeds it inline; zoom / pan / hover for free.                                                     |
| Notebook           | **marimo**                          | No hidden state, git-diffable, reactive cells natively.                                                   |
| Tests              | **pytest** + **hypothesis**         | Hypothesis is used specifically for codegen round-trip properties (see "Testing").                        |
| Cache locations    | **platformdirs**                    | Build cache lands in the OS's XDG cache dir, not a hand-rolled `~/.lbman3d`.                              |

Baseline Python is **3.12+**. That gives us PEP 695 `type` statements, the
new generic syntax (`class Foo[T]:`), and `Self` without a `typing` import —
which matches the "modern C++" style discipline the project already keeps.

`ruff` and `ty` are **optional in the sense that they don't gate a run** —
you can `uv run python examples/channel_serial_cpu.py` on unlinted code. They
gate CI. That split is deliberate: physicists iterating on parameters should
not be interrupted by a lint failure; a PR going into `main` should.

## Existing seams we exploit

- **Angle-bracket includes for the two config headers** — from the project
  CLAUDE.md: `params.h` and `sim_config.h` are included with `<…>` throughout
  `src/` so a physicist can shadow them by prepending a directory to the
  include path. This is exactly what `lbm_add_test` already does per-test in
  [../tests/](../tests/); the Python frontend does the same thing per-run.
- **Existing outputs** — the solver already writes VTKHDF into `data/` and a
  parseable text log to `lbm.log`. The Python frontend consumes those, it
  does not need to be plumbed into the solver's inner loop for monitoring.
- **`monitor.py` parser** — [`../monitor.py`](../monitor.py) already has a
  regex parser for `lbm.log` (`_DATA_RE` etc.). Lift that into a library and
  reuse it in every consumer (CLI monitor, marimo cells, tests) rather than
  re-writing.
- **ParaView scripts** — [`../visualize_in_paraview.py`](../visualize_in_paraview.py)
  and [`../visualize_frame_in_paraview.py`](../visualize_frame_in_paraview.py)
  already know how to render our VTKHDF. They currently hardcode `NX/NY/NZ`;
  make them read the same `RunConfig` object the driver used, so the numbers
  never desync.

## Directory layout (target)

```
python/
├── CLAUDE.md                        # this file
├── pyproject.toml                   # scikit-build-core + uv metadata
├── uv.lock                          # committed
├── ruff.toml                        # or [tool.ruff] in pyproject
├── README.md                        # user-facing quickstart
├── src/lbman3d/
│   ├── __init__.py
│   ├── py.typed                     # PEP 561 marker — types ship with the wheel
│   ├── config.py                    # pydantic ParamsSpec, SimConfigSpec, RunConfig
│   ├── codegen.py                   # RunConfig → params.h, sim_config.h
│   ├── build.py                     # cmake configure/build, per-config cache
│   ├── runner.py                    # launch binary; MPI via mpirun
│   ├── logparse.py                  # lifted from monitor.py; returns polars.DataFrame
│   ├── monitor/
│   │   ├── __init__.py              # LiveMonitor: watchfiles + polars
│   │   ├── cli.py                   # typer app, replaces top-level monitor.py
│   │   └── marimo.py                # reactive-cell helpers
│   ├── io.py                        # read VTKHDF frames as numpy arrays
│   ├── analysis/                    # reusable post-hoc diagnostics on VTKHDF outputs
│   │   ├── __init__.py
│   │   ├── correlation.py           # velocity / director correlation functions
│   │   ├── defects.py               # defect counting / tracking on host output
│   │   └── order.py                 # director-field histograms, S(r), etc.
│   ├── visualize/
│   │   ├── paraview.py              # invoke pvpython, static frames, batch pipeline
│   │   ├── trame.py                 # optional live 3-D (see "3D in a notebook")
│   │   └── pyvista.py               # pyvista fallback (no ParaView needed)
│   ├── plots.py                     # plotly figure builders — consume analysis output
│   └── bindings.py                  # Tier-2 loader: builds + imports the per-config .so
├── cpp/                             # Tier-2 pybind11 bindings
│   ├── CMakeLists.txt
│   └── bindings.cc
├── tests/                           # python-side pytest suite (separate from ../tests/)
│   ├── test_codegen.py
│   ├── test_codegen_roundtrip.py    # hypothesis property tests
│   ├── test_preset_docstrings.py    # BC preset docstrings ↔ sim_config.h structs
│   ├── test_build_cache.py
│   ├── test_output_cache.py         # re-launching a completed run is a no-op
│   ├── test_launch_many.py          # enumerated sweep semantics
│   ├── test_smoke_serial_cpu.py
│   └── test_smoke_bindings.py
├── examples/
│   ├── channel_serial_cpu.py
│   ├── channel_mpi.py
│   ├── marimo_monitor.py
│   ├── marimo_shendruk_benchmark.py # end-to-end paper-shaped notebook (see below)
│   ├── marimo_3d_paraview.py
│   └── custom_orchestration.py
└── .github/workflows/python.yml     # uv + ruff + ty + pytest
```

Keeping the pybind11 module under `python/cpp/` (not merged into `../src/`)
means the C++ tree stays free of pybind11 headers and the existing CMake
build does not gain a Python dependency.

## Tier 1 — Run standard simulations from Python

### 1. Config schema (pydantic v2)

Two `BaseModel` subclasses in `config.py`, one per header, plus a
`RunConfig` that composes them with backend metadata:

```python
from lbman3d import ParamsSpec, SimConfigSpec, RunConfig, WallSpec, Backend

sim = SimConfigSpec.slit_no_slip()          # preset; equivalent to sim_config.h
params = ParamsSpec(                        # every field validated on construction
    nx=20, ny=100, nz=100,
    ALPHA=0.023, LAMBDA=0.3, TAUF=2.5,
    seed=0,                                 # RNG seed for the noise IC — required
    ...
)
run = RunConfig(params=params, sim=sim, backend=Backend.gpu, mpi_ranks=1,
                omp_threads=10)
run.model_dump_json(indent=2)               # serializable, ships next to the HDF5
```

**Everything that affects a run's output bit-for-bit lives in `RunConfig`.**
That includes the RNG seed, OpenMP thread count, and MPI rank count — all
of them can shift results (either byte-identically or statistically) and
all of them therefore enter the hash. If a run's output can change without
a change in its serialized `RunConfig`, the reproducibility contract in
section 8 is broken and paper-shaped notebooks silently produce different
figures on different machines.

Presets `slit_free_slip()`, `slit_no_slip()`, `channel()`, `fully_periodic()`
are ordinary classmethods returning a `SimConfigSpec` matching the C++ presets
in `boundary.h` and `sim_config.h` — nothing magic. `BoundaryConfig` is itself
a plain pydantic model with six `WallSpec` fields (`x_lo`, `x_hi`, `y_lo`,
`y_hi`, `z_lo`, `z_hi`), so users can:

- Start from a preset and modify one wall:
  `bc = BoundaryConfig.slit_no_slip().model_copy(update={"y_lo": ...})`
- Or build a novel geometry from scratch:
  `bc = BoundaryConfig(x_lo=WallSpec(q=BC.Neumann, u=BC.NoSlip), ...)`

**Transparency is a first-class concern here.** A Python user should never
have to open `sim_config.h` to know what a preset actually does. Two
mechanisms enforce that:

1. **`BoundaryConfig._repr_html_`** renders the six walls as a table (rows
   = faces, columns = Q BC, U BC). In a marimo cell, `sim.bc` on its own
   line renders inline, so the notebook itself documents what
   `slit_no_slip` means, right next to the prose describing it. The
   fallback `__repr__` for the REPL prints the same six walls in text
   form. Displaying `sim.bc` in a cell **is** the documentation — no need
   to point the reader at source code.
2. **Each preset's docstring carries its C++ equivalent verbatim.**
   `help(BoundaryConfig.slit_no_slip)` in a REPL, or the auto-rendered
   docstring in a marimo tooltip, shows:

   ```
   Slit with no-slip plates on X, periodic Y/Z. Equivalent to:

       struct SlitNoSlipConfig {
           using XLo = WallSpec<Neumann, NoSlip>;
           using XHi = WallSpec<Neumann, NoSlip>;
           using YLo = WallSpec<Periodic, Periodic>;
           using YHi = WallSpec<Periodic, Periodic>;
           using ZLo = WallSpec<Periodic, Periodic>;
           using ZHi = WallSpec<Periodic, Periodic>;
       };
   ```

   A snapshot test (`test_preset_docstrings.py`, see Testing) parses each
   preset struct out of `../src/sim_config.h` and checks that the
   corresponding Python classmethod's docstring contains it verbatim.
   Drift between Python presets and C++ presets fails CI, so the docstring
   cannot silently become stale.

Pydantic's validators enforce the physics constraints already documented in
`params.h`:

- `DT == 1.0` — hard `Literal[1.0]` (mirrors the `static_assert` in
  [../src/params.h](../src/params.h):30).
- `TAUF >= 0.5 * DT` — otherwise `omega` diverges.
- `A_eff = params.ALPHA * params.nx / sqrt(params.L / 2)` computed as a
  derived field, with a warning above `A_act = 45` (from the stability
  budget in `params.h`).

These are physics guardrails, not paranoia. They belong here and not in the
C++ side because `static_assert` cannot check runtime-supplied numbers, and
the Python config **is** the runtime source.

Every `RunConfig` also carries the git SHA of the C++ tree it was built
against. The JSON dump next to the HDF5 output is the reproducibility
artifact.

### 2. Header codegen

`codegen.py` renders each model to a header string byte-identical to the
in-tree originals, then writes them to a per-run shadow dir:

```
<cache-root>/<hash>/include/params.h
<cache-root>/<hash>/include/sim_config.h
```

`<cache-root>` is `platformdirs.user_cache_dir("lbman3d")`. Overridable with
`$LBMAN3D_CACHE`.

Hash inputs: rendered params.h + rendered sim_config.h + backend flags +
compiler version + git HEAD sha of the C++ tree. Same inputs → same hash →
same cached build.

Rendering must be a straight `f"inline constexpr double ALPHA = {p.ALPHA};"`
job. Do not try to be clever about which values are `constexpr` vs `static
constexpr` vs `inline constexpr` — copy the existing declarations verbatim.
A property test in [`tests/test_codegen_roundtrip.py`](tests/test_codegen_roundtrip.py)
ensures that regenerating from a parsed `params.h` gives the original file
back byte-for-byte for a hypothesis-generated `ParamsSpec`.

### 3. Build cache

`build.py` invokes `cmake -B <cache-root>/<hash>/build -S <repo>` with
`-DCMAKE_C_FLAGS=-I<shadow-include>` — actually, since the target uses
`target_include_directories(... src/)`, we need to prepend the shadow dir.
The cleanest way is to add one CMake option to the C++ side:

```cmake
option(LBM_EXTRA_INCLUDE "Extra include dir prepended before src/" "")
if(LBM_EXTRA_INCLUDE)
    foreach(target main benchmark)
        target_include_directories(${target} BEFORE PUBLIC ${LBM_EXTRA_INCLUDE})
    endforeach()
endif()
```

This is a **one-line change to [../CMakeLists.txt](../CMakeLists.txt)** and
it is the only C++-side change Tier 1 needs. The rest is pure Python.

Backend selection maps to existing CMake flags — no new options needed:

| Backend        | CMake flags                                      | Runner              |
|----------------|--------------------------------------------------|---------------------|
| serial CPU     | `-DLBM_FORCE_CPU=ON`                             | `./main`            |
| serial GPU     | (default; CUDA auto-detected)                    | `./main`            |
| MPI CPU        | `-DLBM_FORCE_CPU=ON -DLBM_ENABLE_MPI=ON`         | `mpirun -n N ./main`|
| MPI GPU (future) | `-DLBM_ENABLE_MPI=ON`                          | `mpirun -n N ./main`|

Always `-DCMAKE_BUILD_TYPE=Release` unless the user sets `debug=True` on the
run config. Debug is the exception, not the default — the GPU run cost noted
in the auto-memory (~12 s per 10 k steps) assumes Release.

### 4. Runner

`runner.py` exposes two entry points:

```python
run: Run           = lb.launch(config)                 # single run, non-blocking
runs: list[Run]    = lb.launch_many(configs, max_concurrent=1)  # enumerated sweep
lb.wait_all(runs, show_progress=True)
```

`launch()` is **content-addressed**: it computes the run hash, looks under
`<cache-root>/runs/<hash>/`, and:

- If `.completed` exists → return an already-populated `Run` handle. No
  build, no launch. This is what makes reopening a marimo notebook cheap.
- If the dir exists without `.completed` (crashed / killed run) → delete
  and restart. Resumption is out of scope (see non-goals).
- Otherwise → build (if the build hash isn't cached), then launch the
  binary with cwd set to the run dir so `data/` and `lbm.log` land inside
  it. `OMP_NUM_THREADS` comes from `config.omp_threads`; MPI backends wrap
  the invocation in `mpirun -n N`. On clean exit, write `.completed`.

`Run` is a small handle:

```python
@dataclass(frozen=True)
class Run:
    config: RunConfig
    hash: str
    data_dir: Path        # <cache-root>/runs/<hash>/data
    log_path: Path        # <cache-root>/runs/<hash>/lbm.log
    manifest: Path        # <cache-root>/runs/<hash>/config.json
    # ... plus poll() / wait() / is_running()
```

`launch_many()` is a thin wrapper: hash-check each config, launch the
uncached ones respecting `max_concurrent` (default 1 — one GPU, one MPI
job), return handles for all. Enumerated sweeps for a notebook — four
plate separations, five seeds — belong here. Real sweep frameworks (SLURM,
hyperparameter search) do not; see non-goals.

Live output during a launch goes through `rich.progress` so
`uv run lbman3d run config.json` gives a live-updating progress bar with
elapsed / ETA out of the box. Failure detection uses the same signal the
C++ side prints ("Simulation diverged at step …" — see
[../src/main.cc](../src/main.cc):18); on failure, `.completed` is **not**
written, so re-launching after fixing the config just reruns cleanly.

Do **not** invoke via `subprocess.check_call(shell=True)` — build the arg
list explicitly. Log paths are user-controlled and shell-metachar-hostile.

### 5. Live monitor (marimo)

Three-way split of the current `monitor.py`:

- `logparse.py` — pure parser lifted from `monitor.py`'s `parse_log` +
  `_DATA_RE`. Returns `(params: dict, rows: polars.DataFrame)`. Zero
  matplotlib. Polars gives us typed columns (`Int64` for time, `Float64` for
  the rest) and cheap `.filter(pl.col("kinetic_energy").is_finite())` in
  place of the numpy mask dance the current script does.
- `monitor/cli.py` — a `typer` app that keeps the current animated
  matplotlib UI as a drop-in replacement for `../monitor.py`. Same
  command-line interface (`--total-energy` etc., in kebab-case per typer
  convention).
- `monitor/marimo.py` — reactive-cell helpers. The pattern:

```python
# marimo notebook cell
import lbman3d as lb
import marimo as mo

run = lb.launch(run_config)                    # returns immediately

tick = mo.ui.refresh(default_interval="2s")    # marimo built-in
tick

# reactive cell — re-runs when tick fires
_ = tick.value
df = lb.logparse.read(run.log_path)
mo.hstack([
    lb.plots.ke_and_umax(df),                  # plotly.Figure
    lb.plots.momentum(df),
])
```

Prefer `mo.ui.refresh` (marimo-native periodic refresh) over polling
threads — it keeps the reactivity story clean and lets the notebook be the
scheduler. If you need file-driven reactivity outside marimo (e.g. a
long-running Python driver script that wants to trigger on log growth),
use `watchfiles.awatch(log_path)` in an async coroutine.

Plot builders in `lb.plots` should return **plotly** figures, not
matplotlib: plotly's zoom / pan / hover is what makes the reactive story
worth the effort, and marimo renders it inline without extra setup.

If the C++ side's log-line format ever changes, the parser is the single
place that has to catch up — the CLI, the marimo helpers, and the tests
all go through it.

### 6. 3-D visualization in a notebook

ParaView has three sensible integration paths, in order of ambition:

**a) Static frames via `pvpython`** (safest, works everywhere)
   Wrap [../visualize_frame_in_paraview.py](../visualize_frame_in_paraview.py)
   as `lbman3d.visualize.paraview.render_frame(run, step, camera=..., out=...)`.
   Under the hood: shell out to `pvpython`, pass a JSON blob of the camera
   / colormap on stdin, get a PNG back. Notebook displays the PNG. No 3-D
   interactivity, but robust and offline-friendly.

**b) trame + iframe** (adjustable 3-D in the notebook)
   [trame](https://kitware.github.io/trame/) is Kitware's own way to embed
   a VTK/ParaView view in a browser. A trame server can be launched from
   the notebook and its URL fed into `mo.iframe(url)`. This gives a real
   adjustable 3-D view. Cost: an extra background process per notebook,
   and trame's install footprint is nontrivial (~500 MB with ParaView).
   Ship it behind `uv sync --extra trame`.

**c) pyvista fallback** (no ParaView needed)
   For users without a working ParaView install, `visualize/pyvista.py`
   renders a subset of the diagnostic views (director glyphs, defect
   isosurfaces) using pyvista, which speaks numpy arrays directly and
   embeds in Jupyter/marimo via `trame` too. Feature set is deliberately
   smaller than (a) — mainly single-time snapshots.

Start with (a) + (c). Add (b) only after (a) has proved itself; trame is
not worth the packaging complexity if static frames suffice for the
day-to-day.

### 7. Analysis on outputs

Reusable diagnostics that read VTKHDF frames and produce polars/numpy live
in `lbman3d.analysis`. This mirrors [../src/analysis/](../src/analysis/) on
the C++ side (defect fields, disclination detection) — the Python side
does the post-hoc statistics the C++ side deliberately skips because they
are not on the solver hot path.

Entries expected on day one:

- `analysis.velocity_corr(run, axis="z")` — 1-D correlation function of
  the velocity field along one axis, averaged over the perpendicular
  directions and over `run`'s saved frames after a warm-up window.
- `analysis.velocity_corr_ratio(run)` — the Shendruk 2018 diagnostic:
  ratio of in-plane to out-of-plane correlation lengths. Takes a `Run`,
  returns a `polars.DataFrame` with columns `(H, A_act, ratio,
  ratio_err)`.
- `analysis.defect_count(run)` — pass-through wrapper around whatever the
  C++ side already writes into the HDF5 (defect count is already logged).
- `analysis.director_histogram(run)` — angular distribution of the
  director field over the domain.

Every entry takes a `Run` handle (from `launch()`) rather than a raw path
— this keeps analysis coupled to the config that produced the data, so
`analysis.velocity_corr(run)` can label its axes with the run's `H`,
`A_act`, etc. without the caller re-supplying them.

Entries return either a `polars.DataFrame` (for tabular diagnostics that
compose across a sweep with `pl.concat`) or a `numpy.ndarray` (for
whole-field results feeding into a plot). No matplotlib in here — plotting
belongs in `lb.plots`, which reads what `analysis` produces.

### 8. Reproducibility contract

The end-goal workflow — a research paper published as a marimo notebook —
only works if reopening the notebook does not recompute the sims. That
imposes an honest contract: every ingredient that affects a run's output
must be captured in the hash, and every artifact must land at a stable
hash-keyed path.

**What's in the hash:**

- `RunConfig.model_dump_json` — grid, physics constants, BC, advection
  scheme, num_steps, save_interval, `params.seed`, `omp_threads`,
  `mpi_ranks`, backend.
- C++ git SHA.
- Compiler + CUDA + MPI versions, captured via `cmake --system-information`.

**What's not:** clock time, hostname, cwd, user, environment variables
that don't feed into the build. These may be logged into the manifest for
provenance but do not enter the cache key.

**Cache layout:**

```
<cache-root>/
├── builds/<build-hash>/           # per (headers + backend + toolchain)
│   ├── include/params.h
│   ├── include/sim_config.h
│   └── build/main
└── runs/<run-hash>/               # per RunConfig (build-hash ⊂ run-hash)
    ├── config.json                # RunConfig.model_dump_json — the manifest
    ├── data/*.vtkhdf              # solver outputs, layout unchanged from today
    ├── lbm.log
    └── .completed                 # written on clean exit; presence = "skip"
```

`build-hash ⊂ run-hash` means: two runs with different seeds share a build
but not a run dir. The build cache is stronger than the run cache.

**When can a re-run actually be bit-identical?**

- Serial CPU + fixed seed: yes, by construction.
- **GPU: no.** CUDA atomic reductions and non-deterministic warp
  scheduling make bit-identity impossible without disabling optimizations
  we care about. GPU runs reproduce **statistical** results across
  re-runs, not byte-identical HDF5. The cache still works — a completed
  run is a completed run — we just do not claim byte equality across
  machines.
- MPI: reduction order depends on rank count, which is already in the
  hash.

Paper-shaped notebooks should be honest about this in an `mo.md` cell.
"This notebook is reproducible" and "these figures are bit-identical
across machines" are different claims; conflating them is what a referee
will catch.

**Notebook re-execution cost model (Shendruk benchmark as example):**

- First execution, cold cache: 4 builds (one per `nx`) + 4 sim runs, each
  ~12 s on GPU at 10 k steps ≈ minutes.
- Second execution, warm cache: hash lookups, no compute. Seconds.
- Third execution on a different machine: cache miss on both builds and
  runs → same as first. If sharing artifacts matters, publish the
  `<cache-root>/runs/` directory alongside the notebook (Zenodo,
  institutional bucket, etc.) — out of scope for now but the design
  supports it because run dirs are self-contained and named by hash.

## Tier 2 — Orchestrate `LbmSolver` and `QTensorSolver` from Python

Same shadow-header + build pipeline as Tier 1, but the output artifact is a
`pybind11` module rather than an executable. `bindings.py` on the Python
side does: hash config → check cache → cmake configure `python/cpp/` with
the shadow includes → build → `importlib` load the resulting `.so`.

### Bindings scope (`python/cpp/bindings.cc`)

Wrap the objects a physicist would need to reproduce the integration tests:

- `FluidFields`, `QTensorFields` — expose the raw storage as **read-only
  numpy views** using `py::array_t<double>` with the strides matching the
  host layout (direction-fastest for `f`, row-major xyz otherwise). Write
  access via explicit `set_from_numpy(arr)`, not silent slicing, so
  round-trips through Python don't accidentally end up on a device copy.
- `LbmSolver<BC>` — `Step()`, `HandleBoundaries()`, access to `ff`.
- `QTensorSolver<BC>` — `Step()`, `SetActiveStressAndComputeBodyForce()`,
  access to `qf`.
- `ActiveNematicSim<BC>` — full loop for parity with Tier 1 when the user
  wants Python-level `for step in range(N): sim.step()`.

Bindings should **not** try to be generic over BC at Python-import time.
Each built module carries one instantiated `BC` (`SlitConfig`,
`ChannelConfig`, etc.), matching the shadow config, and exposes it as
`module.CONFIG_NAME`. If the user wants to run two BCs, they build two
modules — just like they'd run two binaries. This falls out of the
compile-time templating design; do not paper over it with a runtime-BC
virtual base.

### Type stubs

Emit a `.pyi` alongside each per-config `.so`, generated from the pybind11
docstrings via `pybind11-stubgen`. This is what makes `ty` / `pyright`
useful on user code — without stubs, everything through `bindings.py` is
`Any`.

### GPU builds

For a GPU config, the pybind11 module links against the same CUDA object
files the main binary does. Do not try to expose device pointers to Python
— all numpy views go through host mirrors (`DeviceFields::DownloadToHost()`
already exists for this on the CPU side; add a `SyncToHost()` shim that
handles both).

### MPI builds

Skip in the first pass. Python orchestration under `mpirun -n N python
script.py` is possible via `mpi4py`, but the design decisions (who owns
rank 0's Python interpreter, how does an interactive REPL survive halo
exchanges) are enough surface area to earn their own milestone. Tier 2 v1
is serial-only.

## Testing

Two suites, kept separate:

- **`../tests/` (existing GoogleTest)** — untouched. Everything the C++
  tests currently verify must continue to work; the Python frontend is
  downstream.
- **`python/tests/` (new pytest suite)** — exercises the frontend itself:
  - `test_codegen.py` — snapshot test: rendering the current
    `params.h`-equivalent `ParamsSpec` produces a header that diff-matches
    `../src/params.h`. If someone edits `params.h`, this fails, and the fix
    is to add the new field to `ParamsSpec`.
  - `test_codegen_roundtrip.py` — `hypothesis` property test: for arbitrary
    valid `ParamsSpec`, `parse(render(spec)) == spec`. Catches missing
    fields the snapshot test misses because it only sees one canonical set.
  - `test_preset_docstrings.py` — for each `BoundaryConfig` classmethod
    preset, parse the corresponding `struct` out of `../src/sim_config.h`
    and assert the classmethod's docstring contains it verbatim. If a C++
    preset is edited without updating its Python twin, this fails and the
    fix is to sync the docstring (and, likely, the classmethod body).
  - `test_build_cache.py` — same config produces same hash produces same
    build dir; changing one field produces a new hash.
  - `test_smoke_serial_cpu.py` — configure a tiny grid (nx=ny=nz=8, 100
    steps), run to completion, assert `data/*.vtkhdf` files exist and the
    log parser can read them.
  - `test_smoke_bindings.py` — same but through the Tier-2 bindings: step
    10 times, check mass conservation to 1e-12.

The Python smoke tests must be **fast** — target <30 s each — because they
build the C++ side. Use tiny grids and step counts. Real physics validation
stays in the C++ integration tests.

Ruff + ty are wired into the same suite via `uv run ruff check`,
`uv run ruff format --check`, and `uv run ty check`. They run as separate
CI jobs (fail-independently) rather than pytest hooks, so a lint failure
doesn't hide a test failure or vice versa.

## Packaging

`pyproject.toml` uses `scikit-build-core` as the build backend and declares
uv-native dev workflow in `[tool.uv]`:

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "pybind11>=2.13"]
build-backend = "scikit_build_core.build"

[project]
name = "lbman3d"
version = "0.3.0.dev0"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0",
    "polars>=1.0",
    "pydantic>=2.7",
    "typer>=0.12",
    "rich>=13.7",
    "plotly>=5.20",
    "watchfiles>=0.22",
    "platformdirs>=4.2",
    "h5py>=3.11",              # VTKHDF reading
]

[project.optional-dependencies]
marimo   = ["marimo>=0.9"]
trame    = ["trame", "trame-vtk", "trame-vuetify"]
pyvista  = ["pyvista[all]>=0.44"]

[project.scripts]
lbman3d = "lbman3d.cli:app"    # typer app: run, monitor, render, ...

[dependency-groups]            # PEP 735 — modern replacement for extras=[dev]
dev = [
    "pytest>=8",
    "pytest-xdist",            # parallel test runs
    "hypothesis>=6.100",
    "ruff>=0.6",
    "ty",                      # Astral's type checker
    "pybind11-stubgen>=2.5",
]

[tool.scikit-build]
cmake.source-dir = ".."
cmake.build-type = "Release"
wheel.packages = ["src/lbman3d"]

[tool.uv]
package = true                 # this project IS a package, not just a script env
required-version = ">=0.5"
```

Contrast with the old world:

- **`uv sync`** replaces `pip install -e .[dev]` + `venv` creation.
  Reproducible via the committed `uv.lock`.
- **`uv run pytest`** replaces `source .venv/bin/activate && pytest`.
- **`uv build`** produces a wheel via scikit-build-core → this is the same
  wheel `pip install lbman3d` would build.
- **`uv tool run ruff check`** lets you run ruff without installing it into
  the project env at all — nice for pre-commit hooks.

Two subtleties:

1. **`uv sync` builds the Tier-2 module for the default config.** That is
   intentionally a boring default (fully periodic, small grid) so
   `uv sync` on a fresh clone Just Works. The user then re-builds their
   own configuration via `lbman3d.build(params, sim)` in their script.
2. **CUDA is a hard problem for wheels.** Do not attempt a manylinux wheel
   with CUDA baked in. `uv sync` on a CUDA-less machine should succeed
   with a CPU-only build; GPU users build from source with the CMake flags
   they'd use anyway. Route this via `LBM_FORCE_CPU` in
   `[tool.scikit-build.cmake.define]` when a `LBMAN3D_CPU=1` env var is
   set.

## Ruff & ty configuration

`ruff` config lives in `pyproject.toml` under `[tool.ruff]`. Start with a
strict-but-sensible ruleset — the code is greenfield so there is no legacy
to grandfather in:

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = [
    "E", "F", "W",          # pycodestyle + pyflakes
    "I",                    # isort
    "N",                    # pep8-naming
    "UP",                   # pyupgrade — keeps us on modern syntax
    "B",                    # bugbear — real bugs, not style
    "SIM", "RUF",           # simplify + ruff-specific
    "PL",                   # pylint (subset)
    "NPY",                  # numpy-specific
    "PD",                   # pandas-specific (fires on polars too usefully)
]
ignore = [
    "E501",                 # line length — ruff format handles it
    "PLR0913",              # too-many-arguments — pydantic BaseModels legitimately have many
]

[tool.ruff.format]
docstring-code-format = true
```

`ty` config also in `pyproject.toml`:

```toml
[tool.ty]
python-version = "3.12"
strict = ["src/lbman3d/**"]     # strict on our code, permissive on examples
```

The type checker sees `.pyi` stubs generated from the per-config `.so`
modules. When those don't exist yet (i.e. before `uv sync`), `ty` will
correctly infer `Unknown` for anything under `lbman3d.bindings.*`, which is
what we want.

## CI

Single `.github/workflows/python.yml`, three jobs, run in parallel:

```yaml
jobs:
  lint:
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .

  typecheck:
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen
      - run: uv run ty check src/lbman3d/

  test:
    strategy:
      matrix:
        os: [ubuntu-latest]         # add macos-latest once GPU story is clear
        backend: [cpu, mpi-cpu]     # gpu not covered by GH runners
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen
      - run: uv run pytest -n auto python/tests/
```

`uv sync --frozen` refuses to touch the lockfile — if a dependency shifted,
CI fails, not a run. The existing GitHub Actions for the C++ tests are
untouched.

## Non-goals — do not build these yet

Listing these so future-me does not talk themselves into scope creep:

- **A `LbmSim(nx=…, ny=…)` runtime constructor.** That's the static-dims
  policy refactor from [`../src/mpi/CLAUDE.md`](../src/mpi/CLAUDE.md), and
  the C++ CLAUDE.md says explicitly: add it only when a measurement
  justifies it, not preemptively. Python is not that justification.
- **Cluster job schedulers / sweep frameworks.** `launch_many()` covers
  enumerated small sweeps on one machine (four plate separations, five
  seeds — the notebook use case). Anything beyond that — SLURM
  submission, checkpointing, hyperparameter search, cluster-wide restart
  — is a separate `lbman3d-sweep` package layered on top, not baked into
  the frontend. The load-bearing use case here is "notebook runs 4–8
  configs on my workstation," not "sweep 500 configs on a cluster."
- **Resumption of interrupted runs.** A partial run under
  `<cache-root>/runs/<hash>/` without `.completed` is deleted and
  restarted from step 0. The C++ side has no checkpoint format; adding
  one is a large surface for a use case (killing 3-hour runs) that a
  paper-shaped notebook does not exercise.
- **Runtime hot-reload of physics parameters mid-run.** The solver has no
  reload hook and adding one is a large surface. Config is compile-time,
  full stop.
- **A Jupyter/marimo kernel that IS the solver.** Solver-in-Python-process
  means the C++ side has to survive Python signal handling, which the
  CUDA runtime handles poorly. Keep the solver in a subprocess; the
  notebook observes, it does not host.
- **MPI + Python orchestration.** See "MPI builds" above. Separate
  milestone.
- **Backports to older Python.** 3.12+ only. If a collaborator complains,
  point them at `uv python install 3.12` — uv manages the interpreter, so
  the collaborator does not need root or conda.

## Milestone order (suggested)

1. C++-side changes bundled in one PR: `LBM_EXTRA_INCLUDE` CMake option
   **and** surface the RNG seed of the noise IC as a `params.h` constant
   (so `ParamsSpec.seed` has something to shadow). Both are trivial;
   neither has a Python dependency.
2. `pyproject.toml` + `uv.lock` + ruff/ty config + CI skeleton (all three
   jobs; test job has one dummy `test_import.py`). Commits the
   modernization stack before any real code lands, so every subsequent PR
   is already gated.
3. `config.py` (pydantic, including `seed` and `omp_threads`) +
   `codegen.py` + snapshot test + hypothesis round-trip test. No build
   invocation yet — this proves round-tripping.
4. `build.py` + `runner.py` (single-run `launch()`) + serial-CPU smoke
   test.
5. **Content-addressed output cache + `launch_many()` / `wait_all()` +
   `test_output_cache.py` + `test_launch_many.py`.** This is the milestone
   that unlocks the notebook workflow.
6. GPU and MPI-CPU backends in the runner. Same tests, more backends.
7. `logparse.py` extracted from `monitor.py` (now polars-backed);
   `monitor/cli.py` (typer) replaces the top-level script.
   `../monitor.py` becomes a one-line shim (deprecation period).
8. `monitor/marimo.py` + example notebook.
9. ParaView static frames via `pvpython`.
10. **`lbman3d.analysis` subpackage** — correlation functions, defect
    counting, director histograms. Signatures take `Run` handles;
    outputs are polars/numpy.
11. **`examples/marimo_shendruk_benchmark.py`** — the paper-shaped
    reproducibility demonstrator: markdown prose, `launch_many()` over
    four plate separations, `analysis.velocity_corr_ratio()`, a plotly
    figure. This is the milestone that closes the loop and validates
    that steps 1–10 hang together in the way an end-user experiences.
12. Tier 2 pybind11 bindings for serial CPU + `pybind11-stubgen` for
    `.pyi` generation.
13. Tier 2 for serial GPU.
14. trame integration (optional; only if static frames are insufficient
    in practice).

Each step is one PR. Steps 1–6 are the load-bearing ones; steps 10–11
are what turn a working pipeline into a workflow you can publish against.
