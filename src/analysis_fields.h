#ifndef LBM_AN_ANALYSIS_FIELDS_H_
#define LBM_AN_ANALYSIS_FIELDS_H_

#include <vector>
#include "params.h"
#include "qtensor_fields.h"
#include "physics_helpers.h"
#include "grid.h"

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
    

// Landau-de Gennes free energy density at a single point:
//   A/2 TrQ² + B/3 TrQ³ + C/4 (TrQ²)² + elastic
// Elastic uses the IBP form -L/2 Q:∇²Q, equal to L/2 (∇Q)² up to surface
// terms. lap_Q* are the seven-point Laplacians of the corresponding Q
// component (see StepAndSetupBodyForce for the stencil).
double NematicFreeEnergyDensity(
    double Qxx, double Qxy, double Qxz, double Qyy, double Qyz,
    double lap_Qxx, double lap_Qxy, double lap_Qxz, double lap_Qyy, double lap_Qyz);

// Domain-integrated nematic free energy. Recomputes the Laplacian stencil
// itself (via Grid::QXoff/QYoff/QZoff), since that's only cheap to keep
// inline in the solver's hot loop, not worth carrying as per-step state.
// Call occasionally (e.g. from SimIO::Log), not every step.
template<typename BC>
double TotalNematicFreeEnergy(const QTensorFields& qf, const Grid<BC>& grid) {
    double total = 0.0;
    #pragma omp parallel for default(shared) num_threads(Params::numprocs) \
        schedule(static) reduction(+:total)
    for (int z = 0; z < Params::nz; ++z) {
        for (int y = 0; y < Params::ny; ++y) {
            for (int x = 0; x < Params::nx; ++x) {
                const int xm = grid.QXoff(x, -1), xp = grid.QXoff(x, 1);
                const int ym = grid.QYoff(y, -1), yp = grid.QYoff(y, 1);
                const int zm = grid.QZoff(z, -1), zp = grid.QZoff(z, 1);

                const double Qxx = qf.qxx[idx(x, y, z)];
                const double Qxy = qf.qxy[idx(x, y, z)];
                const double Qxz = qf.qxz[idx(x, y, z)];
                const double Qyy = qf.qyy[idx(x, y, z)];
                const double Qyz = qf.qyz[idx(x, y, z)];

                const double lap_Qxx = qf.qxx[idx(xp,y,z)] + qf.qxx[idx(xm,y,z)] + qf.qxx[idx(x,yp,z)] + qf.qxx[idx(x,ym,z)] + qf.qxx[idx(x,y,zp)] + qf.qxx[idx(x,y,zm)] - 6.0*Qxx;
                const double lap_Qxy = qf.qxy[idx(xp,y,z)] + qf.qxy[idx(xm,y,z)] + qf.qxy[idx(x,yp,z)] + qf.qxy[idx(x,ym,z)] + qf.qxy[idx(x,y,zp)] + qf.qxy[idx(x,y,zm)] - 6.0*Qxy;
                const double lap_Qxz = qf.qxz[idx(xp,y,z)] + qf.qxz[idx(xm,y,z)] + qf.qxz[idx(x,yp,z)] + qf.qxz[idx(x,ym,z)] + qf.qxz[idx(x,y,zp)] + qf.qxz[idx(x,y,zm)] - 6.0*Qxz;
                const double lap_Qyy = qf.qyy[idx(xp,y,z)] + qf.qyy[idx(xm,y,z)] + qf.qyy[idx(x,yp,z)] + qf.qyy[idx(x,ym,z)] + qf.qyy[idx(x,y,zp)] + qf.qyy[idx(x,y,zm)] - 6.0*Qyy;
                const double lap_Qyz = qf.qyz[idx(xp,y,z)] + qf.qyz[idx(xm,y,z)] + qf.qyz[idx(x,yp,z)] + qf.qyz[idx(x,ym,z)] + qf.qyz[idx(x,y,zp)] + qf.qyz[idx(x,y,zm)] - 6.0*Qyz;

                total += NematicFreeEnergyDensity(Qxx, Qxy, Qxz, Qyy, Qyz,
                                                   lap_Qxx, lap_Qxy, lap_Qxz, lap_Qyy, lap_Qyz);
            }
        }
    }
    return total;
}

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
