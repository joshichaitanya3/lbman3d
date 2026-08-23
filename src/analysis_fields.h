#ifndef LBM_AN_ANALYSIS_FIELDS_H_
#define LBM_AN_ANALYSIS_FIELDS_H_

#include <vector>
#include <params.h>
#include "qtensor_fields.h"
#include "qtensor_types.h"    // SymTrLessTensor5, reused for uniaxial-Q reconstruction
#include "physics_helpers.h"
#include "offsets.h"
#include "local_grid.h"

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
// itself (via QXoff/QYoff/QZoff, grid.h), since that's only cheap to keep
// inline in the solver's hot loop, not worth carrying as per-step state.
// Call occasionally (e.g. from SimIO::Log), not every step.
template<typename BC>
double TotalNematicFreeEnergy(const QTensorFields& qf) {
    double total = 0.0;

    const LocalGrid& g = qf.grid;
    #pragma omp parallel for default(shared) num_threads(Params::kNumOMPThreads) \
        schedule(static) reduction(+:total)
    for (int z = 0; z < g.local_nz; ++z) {
        for (int y = 0; y < g.local_ny; ++y) {
            for (int x = 0; x < g.local_nx; ++x) {
                const int idxp = g.halo_idx(x, y, z);
                const int xm = QXoff<BC>(x, -1), xp = QXoff<BC>(x, 1);
                const int ym = QYoff<BC>(y, -1), yp = QYoff<BC>(y, 1);
                const int zm = QZoff<BC>(z, -1), zp = QZoff<BC>(z, 1);

                const double Qxx = qf.qxx[idxp];
                const double Qxy = qf.qxy[idxp];
                const double Qxz = qf.qxz[idxp];
                const double Qyy = qf.qyy[idxp];
                const double Qyz = qf.qyz[idxp];

                const double lap_Qxx = qf.qxx[g.halo_idx(xp,y,z)] + qf.qxx[g.halo_idx(xm,y,z)] + qf.qxx[g.halo_idx(x,yp,z)] + qf.qxx[g.halo_idx(x,ym,z)] + qf.qxx[g.halo_idx(x,y,zp)] + qf.qxx[g.halo_idx(x,y,zm)] - 6.0*Qxx;
                const double lap_Qxy = qf.qxy[g.halo_idx(xp,y,z)] + qf.qxy[g.halo_idx(xm,y,z)] + qf.qxy[g.halo_idx(x,yp,z)] + qf.qxy[g.halo_idx(x,ym,z)] + qf.qxy[g.halo_idx(x,y,zp)] + qf.qxy[g.halo_idx(x,y,zm)] - 6.0*Qxy;
                const double lap_Qxz = qf.qxz[g.halo_idx(xp,y,z)] + qf.qxz[g.halo_idx(xm,y,z)] + qf.qxz[g.halo_idx(x,yp,z)] + qf.qxz[g.halo_idx(x,ym,z)] + qf.qxz[g.halo_idx(x,y,zp)] + qf.qxz[g.halo_idx(x,y,zm)] - 6.0*Qxz;
                const double lap_Qyy = qf.qyy[g.halo_idx(xp,y,z)] + qf.qyy[g.halo_idx(xm,y,z)] + qf.qyy[g.halo_idx(x,yp,z)] + qf.qyy[g.halo_idx(x,ym,z)] + qf.qyy[g.halo_idx(x,y,zp)] + qf.qyy[g.halo_idx(x,y,zm)] - 6.0*Qyy;
                const double lap_Qyz = qf.qyz[g.halo_idx(xp,y,z)] + qf.qyz[g.halo_idx(xm,y,z)] + qf.qyz[g.halo_idx(x,yp,z)] + qf.qyz[g.halo_idx(x,ym,z)] + qf.qyz[g.halo_idx(x,y,zp)] + qf.qyz[g.halo_idx(x,y,zm)] - 6.0*Qyz;

                total += NematicFreeEnergyDensity(Qxx, Qxy, Qxz, Qyy, Qyz,
                                                   lap_Qxx, lap_Qxy, lap_Qxz, lap_Qyy, lap_Qyz);
            }
        }
    }
    return total;
}

// Owns analysis state: director vector, scalar order, and past density+velocity
// Flat, row-major storage indexed via g.halo_idx(x,y,z) from physics_helpers.h
// (director_ uses dirIdx(x,y,z,c) instead — see above). Purely a host/CPU
// structure — never touched by GPU code.
struct AnalysisFields {
    // Previous-step snapshots for per-export diagnostics
    LocalGrid grid;
    std::vector<double> ux_past_, uy_past_, uz_past_;
    // Order parameter and director fields populated by QtensorToOrderDirector
    std::vector<double> order_;
    std::vector<double> director_;  // AoS: [nz, ny, nx, 3]
    explicit AnalysisFields(LocalGrid g = LocalGrid::SingleRank());

};

// AoS index into director_: [nz, ny, nx, 3], deliberately interleaved rather
// than split into 3 SoA arrays like the Q-tensor components, so VTKHDF
// export can pass the buffer straight through via WriteVectorField, and so
// defect-finding code (which reads the full 3-vector at each point) gets all
// 3 components contiguous. c is the director component (0=x, 1=y, 2=z).
// Not g.halo_idx(x,y,z,i) from physics_helpers.h: that assumes an ndir-sized last
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
struct OrderDirector {
    double S;
    double nx, ny, nz;
};

OrderDirector QtensorToOrderDirectorPoint(
    double Qxx, double Qxy, double Qxz, double Qyy, double Qyz);

void QtensorToOrderDirector(const QTensorFields& qf, AnalysisFields& af);

/*!
 * Inverse of QtensorToOrderDirectorPoint, exact on the uniaxial subset:
 *
 *     Q_αβ = (3S/2) · (n_α · n_β − δ_αβ/3)
 *
 * where S is the **largest eigenvalue** of Q — the same quantity
 * QtensorToOrderDirectorPoint returns, not the scalar prefactor in the
 * classical convention Q = s(nnᵀ − I/3).
 *
 * Derivation: for uniaxial Q with director n̂ and largest eigenvalue S,
 * Q = S·nnᵀ + λ_⊥·(I − nnᵀ); tracelessness fixes λ_⊥ = −S/2, hence
 * Q = (3S/2)(nnᵀ − I/3).
 *
 * QtensorToOrderDirectorPoint discards the biaxial degrees of freedom
 * (intermediate eigenvalue + its axis) that a general symmetric-traceless Q
 * carries, so this inverse only recovers Q where the discarded pieces vanish
 * — i.e. in the uniaxial regime, and only when the largest eigenvalue is
 * positive (n̂ is the director). That is exact away from disclination cores,
 * which is where the β pipeline samples Q (ring radius R ≥ 2 lattice units,
 * well beyond ξ). Not suitable for reconstructing Q at defect cores.
 */
// Returns the 5 independent components of the reconstructed uniaxial Q.
// SymTrLessTensor5 is the codebase-wide container for symmetric-traceless
// tensors (see qtensor_types.h); reusing it keeps this in the same shape
// as the QStencil / model.h intermediaries and avoids a duplicate struct.
SymTrLessTensor5 OrderDirectorToQtensorPoint(
    double S, double nx, double ny, double nz);

// Bulk reconstruction: reads af.order_ and af.director_, writes qf.qxx..qyz.
// The QTensorFields grid is not required to match AnalysisFields'; both must
// share nx/ny/nz. Halo cells are not touched (they'll be zero from
// QTensorFields' constructor).
void OrderDirectorToQtensor(const AnalysisFields& af, QTensorFields& qf);

#endif // LBM_AN_ANALYSIS_FIELDS_H_
