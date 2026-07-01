#include <chrono>
#include <iostream>
#include "format_compat.h"
#include "sim_config.h"
#include "active_nematic.h"
#include "params.h"

inline constexpr int kWarmupSteps    = 10;
inline constexpr int kBenchmarkSteps = 1000;

int main() {
    using clock = std::chrono::high_resolution_clock;
    using ms    = std::chrono::duration<double, std::milli>;

    ActiveNematicSim<SimBC> sim{Grid<SimBC>(Params::nx, Params::ny, Params::nz)};

    for (int t = 0; t < kWarmupSteps; ++t)
        sim.Step();

    double ms_qtensor = 0, ms_lbm = 0;
    for (int t = 0; t < kBenchmarkSteps; ++t) {
        auto t0 = clock::now();
        sim.QTensorStep();
        auto t1 = clock::now();
        sim.LBMStep();
        auto t2 = clock::now();

        ms_qtensor += ms(t1 - t0).count();
        ms_lbm     += ms(t2 - t1).count();
    }

    double ms_total = ms_qtensor + ms_lbm;
    std::cout << compat::format(
        "NX={} NY={} NZ={} numprocs={}\n"
        "Steps: {}\n"
        "  QTensor : {:.3f} ms/step ({:.1f}%)\n"
        "  LBM     : {:.3f} ms/step ({:.1f}%)\n"
        "  Total   : {:.3f} ms/step\n",
        Params::nx, Params::ny, Params::nz, Params::numprocs,
        kBenchmarkSteps,
        ms_qtensor / kBenchmarkSteps, 100.0 * ms_qtensor / ms_total,
        ms_lbm     / kBenchmarkSteps, 100.0 * ms_lbm     / ms_total,
        ms_total   / kBenchmarkSteps);

    return 0;
}
