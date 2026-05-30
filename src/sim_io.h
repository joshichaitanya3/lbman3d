#ifndef LBM_AN_SIM_IO_H_
#define LBM_AN_SIM_IO_H_

#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "params.h"
#include "fluid_fields.h"
#include "qtensor_fields.h"
#include "analysis_fields.h"
#include "analysis/defect_finder.h"


// Handles all logging and CSV export.
// Owns diagnostic tracking fields (rho_past, ux_past, uy_past) to keep
// FluidFields as pure simulation state with no IO bookkeeping in it.
class SimIO {
    std::ofstream log_file_;
    
public:
    SimIO();
    ~SimIO();

    // Print all Params values and BC name to the log.  Call once after construction.
    void LogSetupSummary(std::string_view bc_name);

    // Log mass, momentum, velocity error; update internal past-velocity state.
    // Returns false (and logs a message) if any quantity is NaN.
    bool Log(const FluidFields& ff, AnalysisFields& af, int time_step);

    // Write per-field CSV files to `path/`.  Updates internal rho_past state.
    void ExportCSV(const FluidFields& ff, const QTensorFields& qf,
                const std::string& path, int step);

    void ExportVTKHDF(const FluidFields& ff, const QTensorFields& qf, AnalysisFields& af,
                const std::string& path, int step, double time);
    
    void ExportDisclinations(const DefectFields& df,
                const std::string& path, int step, double time);
    
    // Write one CSV per lattice direction to `path/`.
    void ExportDistributionCSV(const FluidFields& ff,
        const std::string& path, int step);

};

#endif // LBM_AN_SIM_IO_H_
