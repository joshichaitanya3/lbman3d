#ifndef LBM_AN_ACTIVE_NEMATIC_H_
#define LBM_AN_ACTIVE_NEMATIC_H_

#include <memory>
#include <stdexcept>
#include <string>

#include "offsets.h"
#include <params.h>
#include "boundary.h"
#include "boundary_names.h"
#include "fluid_fields.h"
#include "qtensor_fields.h"
#include "analysis_fields.h"
#include "lbm_solver.h"
#include "qtensor_solver.h"
#include "sim_io.h"
#include "analysis/defect_fields.h"
#include "analysis/defect_finder.h"
#include "analysis/defect_analyzer.h"
#include "analysis/defect_analysis_config.h"
#include "device_fields.h"
#include "device_solver.h"
#include "local_grid.h"
#include "mpi/mpi_context.h"
#include "mpi/halo_exchange_lbm.h"
#include "mpi/halo_exchange_qtensor.h"
#ifdef SIM_WITH_CUDA
#include "cuda/halo_exchange_qtensor_nvshmem.h"
#endif


// Orchestrates LbmSolver + QTensorSolver + SimIO for 3D active nematics.
//
// To use a custom activity model, inject a QTensorSolver subclass:
//
//   ActiveNematicSim<PeriodicBC> sim{std::make_unique<VaryingAlpha>()};
//
// To run without any Q-tensor dynamics use LbmSolver directly.
template<typename BC>
class ActiveNematicSim {
    static_assert(bc_periodicity_consistent_v<BC>,
        "BC has an axis with mixed Periodic/non-Periodic between its QBC and UBC "
        "slots. The finder walks QBC-periodicity; DefectFields sizes off "
        "UBC-periodicity. These must agree per axis.");

    MPIContext         mpi_; // Needs to be the first member
    LocalGrid          grid_;
    // Backend init (VII-b/c): cudaSetDevice to this rank's GPU, NVSHMEM
    // bootstrap + symmetric heap sizing under LBM_ENABLE_NVSHMEM. Declared
    // before d_fields_ so it runs before thrust::device_vector allocations
    // in DeviceFields — otherwise those pin to device 0 while kernels bind
    // to local_rank.
    BackendInfo        backend_info_;
    HaloExchangeQTensor qtensor_halo_;
    HaloExchangeLBM    lbm_halo_;
    FluidFields        fluid_;
    QTensorFields  qtensor_;
    DeviceFields   d_fields_;
    #ifdef LBM_ENABLE_NVSHMEM
    // NVSHMEM Q-tensor halo. Must be declared after d_fields_ (which forces
    // backend init / symmetric-heap sizing to run first) and before d_solver_
    // so ExchangeQTensor is available on the first Step().
    HaloExchangeQTensorNvshmem qtensor_halo_nvshmem_;
    #endif
    DeviceSolver<BC> d_solver_;
    AnalysisFields af_;
    DefectFields   df_{periodicity_by_axis<BC>};
    DefectFinder<BC> finder_;
    DefectAnalyzer<BC> da_;
    LbmSolver<BC>  lbm_;
    std::unique_ptr<QTensorSolver<BC>> qtensor_solver_;
    SimIO          io_;
    int            time_step_ = 0;
    int host_snapshot_step_ = -1;

    void Initialize() {
        lbm_.Initialize(fluid_);
        qtensor_solver_->Initialize(qtensor_);
        #ifdef SIM_WITH_CUDA
        d_fields_.Initialize(fluid_, qtensor_);
        d_solver_.Initialize(d_fields_);
        #endif // SIM_WITH_CUDA
    }
    int num_files_exported = 0;
public:
    // Expose mpi_/grid_ for reporting (benchmark banners, log lines). The
    // fields are const-refs so callers can query world_size/dims/local_n*
    // without depending on the internals being non-const.
    const MPIContext& mpi() const { return mpi_; }
    const LocalGrid&  grid() const { return grid_; }
    const BackendInfo& backend_info() const { return backend_info_; }
    std::string backend_summary() const { return FormatBackendSummary(backend_info_); }

    // Default: constant-alpha active nematic.
    // Supply a QTensorSolver subclass to override the activity model.
    explicit ActiveNematicSim(std::unique_ptr<QTensorSolver<BC>> solver = nullptr)
        : mpi_(periodicity_by_axis<BC>),
          grid_(mpi_.MakeLocalGrid()),
          backend_info_(InitializeComputeBackend(mpi_, grid_)),
          qtensor_halo_(grid_, mpi_),
          lbm_halo_(grid_, mpi_, is_wall_by_face<BC>),
          fluid_(grid_),
          qtensor_(grid_),
          d_fields_(grid_),
          #ifdef LBM_ENABLE_NVSHMEM
          qtensor_halo_nvshmem_(grid_, mpi_),
          #endif
          qtensor_solver_(solver ? std::move(solver)
                                 : std::make_unique<QTensorSolver<BC>>())
    {
        Initialize();
        io_.LogSetupSummary(BC::name, FormatBackendSummary(backend_info_));
    }

    void QTensorStep() {
        #ifdef SIM_WITH_CUDA
        #ifdef LBM_ENABLE_NVSHMEM
        // VII-e: fill Q + velocity ghost cells with the neighbour's owned
        // values so d_solver_.QTensorStep's phase-1 stencil reads valid data
        // across the seam. Enqueued on the default stream — same stream the
        // subsequent QTensorStep kernels run on, so the barrier at the end of
        // ExchangeQTensor also serialises them behind the halo.
        //
        // VII-f will add an ExchangePassiveStresses call between
        // GpuQTensorStep (phase 1) and GpuComputeBodyForce (phase 2); today
        // d_solver_.QTensorStep fuses both, so phase 2's neighbour reads on Σ/τ
        // are the outstanding correctness gap under a multi-rank split — see
        // src/cuda/CLAUDE.md's VII-e/f rows.
        qtensor_halo_nvshmem_.ExchangeQTensor(d_fields_);
        #endif
        d_solver_.QTensorStep(d_fields_);
        #else
        qtensor_halo_.ExchangeQTensor(qtensor_, fluid_);
        qtensor_solver_->StepAndSetupBodyForce(qtensor_, fluid_);

        qtensor_halo_.ExchangePassiveStresses(qtensor_);
        qtensor_solver_->SetActiveStressAndComputeBodyForce(fluid_, qtensor_);
        #endif
    }
    void LBMStep() {
        #ifdef SIM_WITH_CUDA
        d_solver_.LBMStep(d_fields_);
        #else
        lbm_.LatticeBoltzmannStep(fluid_);
        lbm_halo_.ExchangeLBM(fluid_);
        #endif
    }
    // Q-tensor FD step + active force + LBM step.
    void Step() {
        QTensorStep();
        LBMStep();
        ++time_step_;
    }

    bool HostFieldsAreUpToDate() const {
        return host_snapshot_step_ >= 0;
    }
    void SnapshotToHost() {
        #ifdef SIM_WITH_CUDA
        d_fields_.CopyToHost(fluid_, qtensor_);
        #endif
        host_snapshot_step_ = time_step_;
    }

    // Log()/Export() operate on the host snapshot. Fail loud rather than
    // silently reading stale data if the caller forgot to SnapshotToHost.
    // Runs once per event, so the throw path is a boundary guard, not a
    // hot-path assert — it must survive Release, which strips assert().
    void EnsureHostFieldsCurrent() const {
        if (!HostFieldsAreUpToDate()) {
            throw std::runtime_error(
                "ActiveNematicSim::Log/Export called with stale host fields "
                "(snapshot at step " + std::to_string(host_snapshot_step_) +
                ", time_step_ is " + std::to_string(time_step_) +
                "). Call SnapshotToHost() before Log/Export.");
        }
    }

    // Returns false if the simulation has diverged (NaN detected).
    bool Log() {
        EnsureHostFieldsCurrent();
        double nematic_energy = 0.0;
        if constexpr (Params::kTrackNematicEnergy) {
            nematic_energy = TotalNematicFreeEnergy<BC>(qtensor_);
        }
        return io_.Log(fluid_, af_, df_, host_snapshot_step_, nematic_energy);
    }

    // Write one VTKHDF frame (plus the disclination mesh on non-MPI builds).
    // With kDebugLogging the frame also carries the raw D3Q15 populations
    // f0..f14, so nothing needs a second output format to inspect them.
    void Export(const std::string& path) {
        EnsureHostFieldsCurrent();
        QtensorToOrderDirector(qtensor_, af_);
        #ifndef LBM_ENABLE_MPI // MPI-parallel defect-detection not yet implemented
        finder_.FindDefects(af_, df_);
        da_.AnalyzeDefects(df_, qtensor_);
        #endif
        io_.ExportVTKHDF<BC>(fluid_, af_, path, time_step_, mpi_, grid_);
        #ifndef LBM_ENABLE_MPI // MPI-parallel defect-export not yet implemented
        io_.ExportDisclinations<BC>(df_, path, time_step_, mpi_, grid_);
        #endif
        num_files_exported++;
    }

    int GetTimeStep() const { return time_step_; }
    int GetHostSnapshotTimeStep() const { return host_snapshot_step_; }
};

#endif // LBM_AN_ACTIVE_NEMATIC_H_
