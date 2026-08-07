#include <chrono>
#include <iostream>
#include "format_compat.h"
#include <sim_config.h>
#include "active_nematic.h"
#include "mpi/mpi_context.h"
#include <params.h>

inline constexpr int kWarmupSteps    = 10;
inline constexpr int kBenchmarkSteps = 1000;

// Reduce a local millisecond count to the maximum across all ranks. Wall time
// of a step is bounded by the slowest rank, so max is the meaningful
// aggregate. Trivially the identity on CPU-only builds (single rank).
static double MaxAcrossRanks(double local_ms) {
    double global_ms;
    MPIContext::MaxDoubles(&local_ms, &global_ms);
    return global_ms;
}

int main() {
    // Construct sim first — its MPIContext ctor is what initializes MPI. Any
    // rank-aware IO must come after this.
    ActiveNematicSim<SimBC> sim{};

    if (MPIContext::IsRoot())
        std::cout << "Running on: " << InitializeComputeBackend(sim.mpi()) << "\n";

    using clock = std::chrono::high_resolution_clock;
    using ms    = std::chrono::duration<double, std::milli>;

    for (int t = 0; t < kWarmupSteps; ++t)
        sim.Step();

    double ms_qtensor = 0, ms_lbm = 0;
    int num_successful_steps = 0;
    for (int t = 0; t < kBenchmarkSteps; ++t) {
        auto t0 = clock::now();
        sim.QTensorStep();
        auto t1 = clock::now();
        sim.LBMStep();
        auto t2 = clock::now();

        // Per-step wall time is bounded by the slowest rank (all ranks sync
        // in the halo exchanges). Take the max so the reported timing matches
        // what wall-clock `time` reports for the whole run.
        ms_qtensor += MaxAcrossRanks(ms(t1 - t0).count());
        ms_lbm     += MaxAcrossRanks(ms(t2 - t1).count());
        num_successful_steps += 1;
    }

    if (MPIContext::IsRoot()) {
        double ms_total = ms_qtensor + ms_lbm;
        const auto& mpi  = sim.mpi();
        const auto& grid = sim.grid();
        std::cout << compat::format(
            "NX={} NY={} NZ={} kNumOMPThreads={}\n"
            "MPI: world_size={} dims=[{},{},{}] rank0 local=[{},{},{}]\n"
            "Steps(successful): {}({})\n"
            "  QTensor : {:.3f} ms/step ({:.1f}%)\n"
            "  LBM     : {:.3f} ms/step ({:.1f}%)\n"
            "  Total   : {:.3f} ms/step\n",
            Params::nx, Params::ny, Params::nz, Params::kNumOMPThreads,
            mpi.world_size, mpi.dims[0], mpi.dims[1], mpi.dims[2],
            grid.local_nx, grid.local_ny, grid.local_nz,
            kBenchmarkSteps, num_successful_steps,
            ms_qtensor / kBenchmarkSteps, 100.0 * ms_qtensor / ms_total,
            ms_lbm     / kBenchmarkSteps, 100.0 * ms_lbm     / ms_total,
            ms_total   / kBenchmarkSteps);
    }

    return 0;
}
