#include <iostream>
#include <ranges>
#include "format_compat.h"
#include <sim_config.h>
#include "mpi/mpi_context.h"
#include "active_nematic.h"
#include <params.h>

int main() {
    ActiveNematicSim<SimBC> sim{};

#ifdef SIM_WITH_CUDA
    // On the GPU build, the simulation fields live on the GPU memory and the 
    // Q-tensor and the moments are only copied over to the CPU 
    // (via `SnapshotToHost`) for exporting and/or logging.
    // This allows an optimization over the CPU workflow, where, once the fields are
    // copied over, _the GPU can continue time-stepping while the CPU exports/logs. 
    // 
    // This pattern requires the "snapshot" to be a real copy that freezes a
    // point-in-time view of the state, which only exists on GPU (device →
    // host copy). Under CPU there is no separate host memory: sim.Step()
    // mutates fluid_/qtensor_ in place, so a snapshot at the top of the batch
    // can't reference the pre-Step state after the do-while has run.
    // The #else branch below uses the sync pattern for correctness on CPU.
    int t = 0;
    bool save_tick = true;
    bool log_tick = true;
    while (t < kNumSteps) {
        sim.SnapshotToHost();
        bool save_tick_next, log_tick_next;
        do {
            sim.Step();
            t++;
            save_tick_next = (t % kSaveInterval == 0);
            // kDebugLogging overrides kLogInterval (logs every step) and, separately,
            // causes SimIO::ExportVTKHDF to include the raw D3Q15 populations f0..f14.
            log_tick_next = (Params::kDebugLogging || t % kLogInterval == 0);
        } while ((t < kNumSteps) && !save_tick_next && !log_tick_next);

        if (save_tick) {
            if (MPIContext::IsRoot())
                std::cout << compat::format("Step {}", sim.GetHostSnapshotTimeStep()) << "\n";
            sim.Export("data");
        }
        if (log_tick) {
            if (!sim.Log()) {
                if (MPIContext::IsRoot())
                    std::cerr << compat::format("Simulation diverged at step {} — exiting.\n", sim.GetHostSnapshotTimeStep());
                return 1;
            }
        }
        save_tick = save_tick_next;
        log_tick = log_tick_next;
    }
#else
    for (int t : std::views::iota(0, kNumSteps)) {
        const bool save_tick = (t % kSaveInterval == 0);
        // kDebugLogging overrides kLogInterval (logs every step) and, separately,
        // causes SimIO::ExportVTKHDF to include the raw D3Q15 populations f0..f14.
        const bool log_tick = (Params::kDebugLogging || t % kLogInterval == 0);
        if (save_tick || log_tick) {
            sim.SnapshotToHost(); // This is a no-op on CPU, but it stamps host_snapshot_step_ so the guard in Log/Export passes
        }
        if (save_tick) {
            if (MPIContext::IsRoot()) {
                int bar_width = 50;
                double progress = static_cast<double>(t) / kNumSteps;
                int filled = static_cast<int>(progress * bar_width);
                std::string bar(filled, '#');
                bar.resize(bar_width, '-');
                std::cout << compat::format("[{}] {:.1f}% ({}/{})\r", bar, progress * 100, t, kNumSteps);
                std::cout.flush();
            }
            sim.Export("data");
        }
        if (log_tick) {
            if (!sim.Log()) {
                if (MPIContext::IsRoot())
                    std::cerr << compat::format("Simulation diverged at step {} — exiting.\n", t);
                return 1;
            }
        }
        sim.Step();
    }
#endif

    return 0;
}
