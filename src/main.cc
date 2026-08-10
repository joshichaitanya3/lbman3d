#include <iostream>
#include <ranges>
#include "format_compat.h"
#include <sim_config.h>
#include "mpi/mpi_context.h"
#include "active_nematic.h"
#include <params.h>

int main() {
    ActiveNematicSim<SimBC> sim{};
    for (int t : std::views::iota(0, kNumSteps)) {
        if (t % kSaveInterval == 0) {
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
        // kDebugLogging overrides kLogInterval (logs every step) and, separately,
        // causes SimIO::ExportVTKHDF to include the raw D3Q15 populations f0..f14.
        if (Params::kDebugLogging || t % kLogInterval == 0) {
            if (!sim.Log()) {
                if (MPIContext::IsRoot())
                    std::cerr << compat::format("Simulation diverged at step {} — exiting.\n", t);
                return 1;
            }
        }
        sim.Step();
    }
    return 0;
}
