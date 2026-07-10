#ifndef LBM_AN_ANALYSIS_FIELDS_H_
#define LBM_AN_ANALYSIS_FIELDS_H_

#include <vector>
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
// Flat, row-major storage indexed via idx(x,y,z) from physics_helpers.h
// (director_ uses dirIdx(x,y,z,c) instead — see above). Purely a host/CPU
// structure — never touched by GPU code.
struct AnalysisFields {
    // Previous-step snapshots for per-export diagnostics
    std::vector<double> rho_past_, ux_past_, uy_past_, uz_past_;
    // Order parameter and director fields populated by QtensorToOrderDirector
    std::vector<double> order_;
    std::vector<double> director_;  // AoS: [nz, ny, nx, 3]

    AnalysisFields();

};

// AoS index into director_: [nz, ny, nx, 3], deliberately interleaved rather
// than split into 3 SoA arrays like the Q-tensor components, so VTKHDF
// export can pass the buffer straight through via WriteVectorField, and so
// defect-finding code (which reads the full 3-vector at each point) gets all
// 3 components contiguous. c is the director component (0=x, 1=y, 2=z).
// Not idx(x,y,z,i) from physics_helpers.h: that assumes an ndir-sized last
// dimension, not 3.
inline int dirIdx(int x, int y, int z, int c) {
    return (((z + Params::nz) % Params::nz) * Params::ny * Params::nx
          + ((y + Params::ny) % Params::ny) * Params::nx
          + (x + Params::nx) % Params::nx) * 3 + c;
}

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
