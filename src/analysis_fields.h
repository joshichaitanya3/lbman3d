#ifndef LBM_AN_ANALYSIS_FIELDS_H_
#define LBM_AN_ANALYSIS_FIELDS_H_

#include <vector>
#include <mdspan/mdspan.hpp>
#include "params.h"
#include "qtensor_fields.h"

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
    

// Owns analysis state: director vector, scalar order, and past density+velocity
struct AnalysisFields {
    // Previous-step snapshots for per-export diagnostics
    std::vector<double> rho_past_data_, ux_past_data_, uy_past_data_, uz_past_data_;
    // Order parameter and director fields populated by QtensorToOrderDirector
    std::vector<double> order_data_, director_data_;
    using ext3_t  = Kokkos::extents<int, Params::nz, Params::ny, Params::nx>;
    using ext4_t = Kokkos::extents<int, Params::nz, Params::ny, Params::nx, 3>;
    Kokkos::mdspan<double, ext3_t>  rho_past_, ux_past_, uy_past_, uz_past_;
    Kokkos::mdspan<double, ext3_t>  order_;
    Kokkos::mdspan<double, ext4_t> director_;

    AnalysisFields();
   
};

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
void QtensorToOrderDirector(const QTensorFields& qf, AnalysisFields& af);

#endif // LBM_AN_ANALYSIS_FIELDS_H_
