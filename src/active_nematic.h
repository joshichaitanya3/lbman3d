#ifndef LBM_AN_ACTIVE_NEMATIC_H_
#define LBM_AN_ACTIVE_NEMATIC_H_

#include <memory>
#include <string>

#include "grid.h"
#include "params.h"
#include "fluid_fields.h"
#include "qtensor_fields.h"
#include "analysis_fields.h"
#include "lbm_solver.h"
#include "qtensor_solver.h"
#include "sim_io.h"
#include "analysis/defect_fields.h"
#include "analysis/defect_finder.h"

enum ExportFormat { CSV, VTKHDF };

// Orchestrates LbmSolver + QTensorSolver + SimIO for 2D active nematics.
//
// To use a custom activity model, inject a QTensorSolver subclass:
//
//   ActiveNematicSim<PeriodicBC> sim{grid, std::make_unique<VaryingAlpha>(grid)};
//
// To run without any Q-tensor dynamics use LbmSolver directly.
template<typename BC>
class ActiveNematicSim {
    FluidFields    fluid_;
    QTensorFields  qtensor_;
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
    }
    int num_files_exported = 0;
public:
    // Default: constant-alpha active nematic.
    // Supply a QTensorSolver subclass to override the activity model.
    explicit ActiveNematicSim(Grid<BC> grid,
                              std::unique_ptr<QTensorSolver<BC>> solver = nullptr)
        : lbm_(grid),
          qtensor_solver_(solver ? std::move(solver)
                                 : std::make_unique<QTensorSolver<BC>>(grid)),
          finder_(grid)
    {
        Initialize();
        io_.LogSetupSummary(Grid<BC>::GridType());
    }

    // Q-tensor FD step + active force + LBM step.
    void Step() {
        qtensor_solver_->Step(qtensor_, fluid_);
        lbm_.LatticeBoltzmannStep(fluid_);
        ++time_step_;
    }

    // Returns false if the simulation has diverged (NaN detected).
    bool Log() { return io_.Log(fluid_, af_, df_, time_step_); }

    void Export(const std::string& path, ExportFormat fmt) {
        // io_.Export(fluid_, qtensor_, path, time_step_);
        QtensorToOrderDirector(qtensor_, af_);
        finder_.FindDefects(qtensor_, af_, df_);

        switch (fmt)
        {
        case CSV:
            io_.ExportCSV(fluid_, qtensor_, path, time_step_);
            num_files_exported++;
            break;
        case VTKHDF:
            io_.ExportVTKHDF(fluid_, qtensor_, af_, path, num_files_exported, static_cast<double>(time_step_)*Params::DT);
            io_.ExportDisclinations(df_, path, num_files_exported, static_cast<double>(time_step_)*Params::DT);
            num_files_exported++;
            break;
        default:
            break;
        }
    }

    int GetTimeStep() const { return time_step_; }
};

#endif // LBM_AN_ACTIVE_NEMATIC_H_
