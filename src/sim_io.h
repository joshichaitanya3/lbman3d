#ifndef LBM_AN_SIM_IO_H_
#define LBM_AN_SIM_IO_H_

#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <params.h>
#include "fluid_fields.h"
#include "qtensor_fields.h"
#include "analysis_fields.h"
#include "analysis/defect_fields.h"
#include "analysis/defect_finder.h"
#include "mpi/mpi_context.h"

// Handles all logging and file export.
// The past-velocity tracking fields it reads live in AnalysisFields, keeping
// FluidFields pure simulation state with no IO bookkeeping in it.
class SimIO {
    std::ofstream log_file_;
    
public:
    SimIO();
    ~SimIO();

    // Print all Params values, BC name, and compute backend (CPU/GPU) info to
    // the log.  Call once after construction.
    void LogSetupSummary(std::string_view bc_name, std::string_view backend_info);

    // Log mass, momentum, velocity error; update internal past-velocity state.
    // nematic_energy is the caller-computed TotalNematicFreeEnergy (see
    // analysis_fields.h) — not recomputed here since it's only meaningful
    // occasionally, not worth templating Log() on BC.
    // Returns false (and logs a message) if any quantity is NaN.
    bool Log(const FluidFields& ff, AnalysisFields& af, const DefectFields& df, int time_step, double nematic_energy);

    void ExportVTKHDF(const FluidFields& ff, AnalysisFields& af,
                const std::string& path, int step, const MPIContext& ctx, const LocalGrid& grid);
    
    void ExportDisclinations(const DefectFields& df,
                const std::string& path, int step, const MPIContext& ctx, const LocalGrid&);

};

#endif // LBM_AN_SIM_IO_H_
