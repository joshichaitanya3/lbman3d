#include <chrono>
#include <iostream>
#include <vector>
#include "format_compat.h"
#include <sim_config.h>
#include "active_nematic.h"
#include "mpi/mpi_context.h"
#include <params.h>
#ifdef SIM_WITH_CUDA
#include <cuda_runtime.h>
#endif

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

#ifdef SIM_WITH_CUDA
    // On GPU, QTensorStep/LBMStep only enqueue kernels on the default stream
    // and return in microseconds. The chrono deltas below therefore measure
    // launch-side CPU wall, not GPU work — they only *look* ms-scale because
    // the driver's launch queue eventually fills and subsequent
    // cudaLaunchKernel calls block the CPU at the GPU's drain rate. To get
    // the real GPU throughput, we bracket the whole loop with a cudaEvent
    // pair on the default stream — the events fire on the GPU's timeline
    // when the stream reaches them, so their elapsed time is the actual GPU
    // wall clock between the two markers, ignoring any CPU-side scheduling.
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);
    // Recording gpu_start now enqueues it *behind* any warmup kernels still
    // in the stream — it won't fire until those complete, so the elapsed
    // time we read at the end correctly excludes warmup.
    cudaEventRecord(gpu_start, 0);

    // Per-phase GPU breakdown: one event after each phase per iteration.
    // Recording an event on the default stream does *not* force a sync —
    // it just enqueues a timestamp marker. We read all elapsed times after
    // the loop, once every event has fired, so the async pipeline is
    // preserved during the timed section.
    std::vector<cudaEvent_t> ev_qt(kBenchmarkSteps), ev_lbm(kBenchmarkSteps);
    for (int i = 0; i < kBenchmarkSteps; ++i) {
        cudaEventCreate(&ev_qt[i]);
        cudaEventCreate(&ev_lbm[i]);
    }
#endif

    // Wall clock around the whole loop, once — closer to what `time ./benchmark`
    // reports than the sum of per-launch chrono deltas, and doesn't accumulate
    // launch-overhead artefacts.
    auto loop_t0 = clock::now();

    double ms_qtensor_cpu = 0, ms_lbm_cpu = 0;
    int num_successful_steps = 0;
    for (int t = 0; t < kBenchmarkSteps; ++t) {
        auto t0 = clock::now();
        sim.QTensorStep();
#ifdef SIM_WITH_CUDA
        cudaEventRecord(ev_qt[t], 0);
#endif
        auto t1 = clock::now();
        sim.LBMStep();
#ifdef SIM_WITH_CUDA
        cudaEventRecord(ev_lbm[t], 0);
#endif
        auto t2 = clock::now();

        // Per-step wall time is bounded by the slowest rank (all ranks sync
        // in the halo exchanges). Take the max so the reported timing matches
        // what wall-clock `time` reports for the whole run.
        ms_qtensor_cpu += MaxAcrossRanks(ms(t1 - t0).count());
        ms_lbm_cpu     += MaxAcrossRanks(ms(t2 - t1).count());
        num_successful_steps += 1;
    }

    auto loop_t1 = clock::now();
    const double ms_loop_cpu = MaxAcrossRanks(ms(loop_t1 - loop_t0).count());

#ifdef SIM_WITH_CUDA
    cudaEventRecord(gpu_end, 0);
    // The GPU is still processing the tail of the queue at this point;
    // block until gpu_end fires so cudaEventElapsedTime returns a valid
    // value. This sync only affects the reported number — it doesn't
    // change the work the GPU did.
    cudaEventSynchronize(gpu_end);
    float ms_gpu_f = 0.f;
    cudaEventElapsedTime(&ms_gpu_f, gpu_start, gpu_end);
    const double ms_loop_gpu = MaxAcrossRanks(static_cast<double>(ms_gpu_f));

    // Aggregate per-phase GPU time. gpu_end's sync above guarantees every
    // per-iteration event has also fired (they were recorded before gpu_end
    // on the same stream), so all cudaEventElapsedTime calls here are safe.
    // For iter N: QT phase = elapsed(prev LBM end, this QT end); LBM phase =
    // elapsed(this QT end, this LBM end). Iter 0's QT reference is gpu_start.
    double ms_qtensor_gpu = 0.0, ms_lbm_gpu = 0.0;
    for (int t = 0; t < kBenchmarkSteps; ++t) {
        float qt_ms = 0.f, lbm_ms = 0.f;
        cudaEventElapsedTime(&qt_ms,
                             (t == 0) ? gpu_start : ev_lbm[t - 1],
                             ev_qt[t]);
        cudaEventElapsedTime(&lbm_ms, ev_qt[t], ev_lbm[t]);
        ms_qtensor_gpu += static_cast<double>(qt_ms);
        ms_lbm_gpu     += static_cast<double>(lbm_ms);
    }
    ms_qtensor_gpu = MaxAcrossRanks(ms_qtensor_gpu);
    ms_lbm_gpu     = MaxAcrossRanks(ms_lbm_gpu);

    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_end);
    for (int i = 0; i < kBenchmarkSteps; ++i) {
        cudaEventDestroy(ev_qt[i]);
        cudaEventDestroy(ev_lbm[i]);
    }
#endif

    if (MPIContext::IsRoot()) {
        const double ms_total_cpu = ms_qtensor_cpu + ms_lbm_cpu;
        const auto& mpi  = sim.mpi();
        const auto& grid = sim.grid();
        std::cout << compat::format(
            "NX={} NY={} NZ={} kNumOMPThreads={}\n"
            "MPI: world_size={} dims=[{},{},{}] rank0 local=[{},{},{}]\n"
            "Steps(successful): {}({})\n",
            Params::nx, Params::ny, Params::nz, Params::kNumOMPThreads,
            mpi.world_size, mpi.dims[0], mpi.dims[1], mpi.dims[2],
            grid.local_nx, grid.local_ny, grid.local_nz,
            kBenchmarkSteps, num_successful_steps);
#ifdef SIM_WITH_CUDA
        const double ms_total_gpu = ms_qtensor_gpu + ms_lbm_gpu;
        std::cout << compat::format(
            "  QTensor CPU-wall : {:.3f} ms/step ({:.1f}%)  [async launch, not GPU work]\n"
            "  LBM     CPU-wall : {:.3f} ms/step ({:.1f}%)  [async launch, not GPU work]\n"
            "  Sum of launches  : {:.3f} ms/step             [proxy for GPU throughput only when queue saturates]\n"
            "  Full-loop CPU    : {:.3f} ms/step             [matches `time ./benchmark` minus startup]\n"
            "  QTensor GPU-wall : {:.3f} ms/step ({:.1f}%)  ← actual GPU work in Q-tensor phase\n"
            "  LBM     GPU-wall : {:.3f} ms/step ({:.1f}%)  ← actual GPU work in LBM phase\n"
            "  GPU-event total  : {:.3f} ms/step             ← actual GPU throughput\n",
            ms_qtensor_cpu / kBenchmarkSteps, 100.0 * ms_qtensor_cpu / ms_total_cpu,
            ms_lbm_cpu     / kBenchmarkSteps, 100.0 * ms_lbm_cpu     / ms_total_cpu,
            ms_total_cpu   / kBenchmarkSteps,
            ms_loop_cpu    / kBenchmarkSteps,
            ms_qtensor_gpu / kBenchmarkSteps, 100.0 * ms_qtensor_gpu / ms_total_gpu,
            ms_lbm_gpu     / kBenchmarkSteps, 100.0 * ms_lbm_gpu     / ms_total_gpu,
            ms_loop_gpu    / kBenchmarkSteps);
#else
        std::cout << compat::format(
            "  QTensor : {:.3f} ms/step ({:.1f}%)\n"
            "  LBM     : {:.3f} ms/step ({:.1f}%)\n"
            "  Total   : {:.3f} ms/step\n",
            ms_qtensor_cpu / kBenchmarkSteps, 100.0 * ms_qtensor_cpu / ms_total_cpu,
            ms_lbm_cpu     / kBenchmarkSteps, 100.0 * ms_lbm_cpu     / ms_total_cpu,
            ms_total_cpu   / kBenchmarkSteps);
#endif
    }

    return 0;
}
