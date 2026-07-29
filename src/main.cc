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
            if (MPIContext::IsRoot())
                std::cout << compat::format("Step {}", t) << "\n";
            sim.Export("data", VTKHDF);
            if constexpr (!Params::kDebugLogging) {
                if (!sim.Log()) {
                    std::cerr << compat::format("Simulation diverged at step {} — exiting.\n", t);
                    return 1;
                }
            }
        }
        sim.Step();
        if constexpr (Params::kDebugLogging) {
            if (!sim.Log()) {
                if (MPIContext::IsRoot())
                    std::cerr << compat::format("Simulation diverged at step {} — exiting.\n", t);
                return 1;
            }
        }
    }
    return 0;
}
