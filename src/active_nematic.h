#ifndef LBM_AN_ACTIVE_NEMATIC_H_
#define LBM_AN_ACTIVE_NEMATIC_H_

#include <memory>
#include <string>

#include "offsets.h"
#include <params.h>
#include "boundary.h"
#include "fluid_fields.h"
#include "qtensor_fields.h"
#include "analysis_fields.h"
#include "lbm_solver.h"
#include "qtensor_solver.h"
#include "sim_io.h"
#include "analysis/defect_fields.h"
#include "analysis/defect_finder.h"
#include "device_fields.h"
#include "device_solver.h"
#include "local_grid.h"
#include "mpi/mpi_context.h"
#include "mpi/halo_exchange_lbm.h"
#include "mpi/halo_exchange_qtensor.h"


// Orchestrates LbmSolver + QTensorSolver + SimIO for 3D active nematics.
//
// To use a custom activity model, inject a QTensorSolver subclass:
//
//   ActiveNematicSim<PeriodicBC> sim{std::make_unique<VaryingAlpha>()};
//
// To run without any Q-tensor dynamics use LbmSolver directly.
template<typename BC>
class ActiveNematicSim {
    MPIContext         mpi_; // Needs to be the first member
    LocalGrid          grid_;
    HaloExchangeQTensor qtensor_halo_;
    HaloExchangeLBM    lbm_halo_;
    FluidFields        fluid_;
    QTensorFields  qtensor_;
    DeviceFields   d_fields_;
    DeviceSolver<BC> d_solver_;
    AnalysisFields af_;
    DefectFields   df_;
    DefectFinder<BC> finder_;
    LbmSolver<BC>  lbm_;
    std::unique_ptr<QTensorSolver<BC>> qtensor_solver_;
    SimIO          io_;
    int            time_step_ = 0;

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

    // Default: constant-alpha active nematic.
    // Supply a QTensorSolver subclass to override the activity model.
    explicit ActiveNematicSim(std::unique_ptr<QTensorSolver<BC>> solver = nullptr)
        : mpi_(periodicity_by_axis<BC>),
          grid_(mpi_.MakeLocalGrid()),
          qtensor_halo_(grid_, mpi_),
          lbm_halo_(grid_, mpi_, is_wall_by_face<BC>),
          fluid_(grid_),
          qtensor_(grid_),
          d_fields_(grid_),
          qtensor_solver_(solver ? std::move(solver)
                                 : std::make_unique<QTensorSolver<BC>>())
    {
        Initialize();
        io_.LogSetupSummary(BC::name, InitializeComputeBackend(mpi_));
    }

    void QTensorStep() {
        #ifdef SIM_WITH_CUDA
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

    // Returns false if the simulation has diverged (NaN detected).
    bool Log() {
        #ifdef SIM_WITH_CUDA
        d_fields_.CopyToHost(fluid_, qtensor_);
        #endif
        double nematic_energy = 0.0;
        if constexpr (Params::kTrackNematicEnergy) {
            nematic_energy = TotalNematicFreeEnergy<BC>(qtensor_);
        }
        return io_.Log(fluid_, af_, df_, time_step_, nematic_energy);
    }

    // Write one VTKHDF frame (plus the disclination mesh on non-MPI builds).
    // With kDebugLogging the frame also carries the raw D3Q15 populations
    // f0..f14, so nothing needs a second output format to inspect them.
    void Export(const std::string& path) {
        #ifdef SIM_WITH_CUDA
        d_fields_.CopyToHost(fluid_, qtensor_);
        #endif
        QtensorToOrderDirector(qtensor_, af_);
        #ifndef LBM_ENABLE_MPI // MPI-parallel defect-detection not yet implemented
        finder_.FindDefects(af_, df_);
        #endif

        io_.ExportVTKHDF(fluid_, af_, path, num_files_exported, mpi_, grid_);
        #ifndef LBM_ENABLE_MPI // MPI-parallel defect-export not yet implemented
        io_.ExportDisclinations(df_, path, num_files_exported, mpi_, grid_);
        #endif
        num_files_exported++;
    }

    int GetTimeStep() const { return time_step_; }
};

#endif // LBM_AN_ACTIVE_NEMATIC_H_
