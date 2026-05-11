#ifndef LBM_AN_SIM_IO_H_
#define LBM_AN_SIM_IO_H_

#include <fstream>
#include <string>
#include <string_view>
#include <vector>
#include <mdspan/mdspan.hpp>

#include "params.h"
#include "fluid_fields.h"
#include "qtensor_fields.h"

// Handles all logging and CSV export.
// Owns diagnostic tracking fields (rho_past, ux_past, uy_past) to keep
// FluidFields as pure simulation state with no IO bookkeeping in it.
class SimIO {
    std::ofstream log_file_;

    // Previous-step snapshots for per-export diagnostics
    std::vector<double> rho_past_data_, ux_past_data_, uy_past_data_, uz_past_data_;
    std::vector<double> vel, director, order; // For VTKHDF export
    using ext_t = Kokkos::extents<int, Params::nx, Params::ny, Params::nz>;
    Kokkos::mdspan<double, ext_t> rho_past_, ux_past_, uy_past_, uz_past_;
public:
    SimIO();
    ~SimIO();

    // Print all Params values and BC name to the log.  Call once after construction.
    void LogSetupSummary(std::string_view bc_name);

    // Log mass, momentum, velocity error; update internal past-velocity state.
    // Returns false (and logs a message) if any quantity is NaN.
    bool Log(const FluidFields& ff, int time_step);

    // Write per-field CSV files to `path/`.  Updates internal rho_past state.
    void ExportCSV(const FluidFields& ff, const QTensorFields& qf,
                const std::string& path, int step);

    void ExportVTKHDF(const FluidFields& ff, const QTensorFields& qf,
                const std::string& path, int step, double time);
    // Write one CSV per lattice direction to `path/`.
    void ExportDistributionCSV(const FluidFields& ff,
        const std::string& path, int step);
    
    double HalfTrQ2(
        double Qxx,
        double Qxy,
        double Qxz,
        double Qyy,
        double Qyz);
    
    double DetQ(
        double Qxx,
        double Qxy,
        double Qxz,
        double Qyy,
        double Qyz);
        
    /* !\brief Efficiently compute the scalar order parameter S and director n from 
    the five independent components of a symmetric traceless Q-tensor field.

    \param components list of 5 ndarrays [Qxx, Qxy, Qxz, Qyy, Qyz], each of shape (nx, ny, nz)
    \param dtype numpy dtype, optional
    \param Output dtype for S and the director components.
    \param Defaults to the dtype of the input arrays.
    
    The Q-tensor at each grid point is:
    
    Q = [[Qxx,  Qxy,  Qxz],
    [Qxy,  Qyy,  Qyz],
    [Qxz,  Qyz, -Qxx-Qyy]]
    
    S is defined as the largest eigenvalue of Q.  The director n is the
    eigenvector corresponding to S.  Because n and -n are physically
    equivalent, no sign convention is enforced here.
    
    The technique used in this method does not require diagonalization of
    the Q-tensor (see ../docs/order_parameter_calculation.pdf).
    
    The calculation is always performed in float64 for numerical
    accuracy.  The results are then cast to `dtype` before returning.
    It is therefore recommended to load the Q-tensor components as float64
    and pass dtype=np.float32 here when reduced output precision is desired.
    
    */
    void QtensorToOrderDirector(const QTensorFields& qf);
};

#endif // LBM_AN_SIM_IO_H_
