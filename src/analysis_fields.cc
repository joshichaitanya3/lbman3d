#include "analysis_fields.h"
#include <params.h>
#include "physics_helpers.h"
#include <cmath>
#include <algorithm>

using namespace Params;

AnalysisFields::AnalysisFields() :
    rho_past_(nx * ny * nz, kDensity),
    ux_past_ (nx * ny * nz, 0.0),
    uy_past_ (nx * ny * nz, 0.0),
    uz_past_ (nx * ny * nz, 0.0),
    order_   (nx * ny * nz, 0.0),
    director_(nx * ny * nz * 3, 0.0)
{}


double HalfTrQ2(
    double Qxx,
    double Qxy,
    double Qxz,
    double Qyy,
    double Qyz) {
    return (Qxx*Qxx + Qxx*Qyy + Qyy*Qyy + Qxy*Qxy + Qxz*Qxz + Qyz*Qyz);
}

double DetQ(
    double Qxx,
    double Qxy,
    double Qxz,
    double Qyy,
    double Qyz) {

    return (-(Qxx + Qyy) * (Qxx*Qyy - Qxy*Qxy) - Qyy*Qxz*Qxz - Qxx*Qyz*Qyz + 2*Qxy*Qxz*Qyz);
}

double NematicFreeEnergyDensity(
    double Qxx, double Qxy, double Qxz, double Qyy, double Qyz,
    double lap_Qxx, double lap_Qxy, double lap_Qxz, double lap_Qyy, double lap_Qyz) {

    const double TrQ2 = 2.0 * HalfTrQ2(Qxx, Qxy, Qxz, Qyy, Qyz);

    const double kone_thirds = 1.0/3.0;
    const double Q2_xx = Qxx*Qxx + Qxy*Qxy + Qxz*Qxz - kone_thirds * TrQ2;
    const double Q2_xy = Qxx*Qxy + Qxy*Qyy + Qxz*Qyz;
    const double Q2_xz = Qxy*Qyz - Qxz*Qyy;
    const double Q2_yy = Qxy*Qxy + Qyy*Qyy + Qyz*Qyz - kone_thirds * TrQ2;
    const double Q2_yz = Qxy*Qxz - Qyz*Qxx;

    const double TrQ3 = 2.0*Qxx*Q2_xx + 2.0*Qyy*Q2_yy + Qxx*Q2_yy + Qyy*Q2_xx
                       + 2.0*(Qxy*Q2_xy + Qxz*Q2_xz + Qyz*Q2_yz);
    const double Q_lap_Q = 2.0*Qxx*lap_Qxx + 2.0*Qyy*lap_Qyy + Qxx*lap_Qyy + Qyy*lap_Qxx
                         + 2.0*(Qxy*lap_Qxy + Qxz*lap_Qxz + Qyz*lap_Qyz);

    return 0.5*A*TrQ2 + (B/3.0)*TrQ3 + 0.25*C*TrQ2*TrQ2 - 0.5*L*Q_lap_Q;
}

void QtensorToOrderDirector(const QTensorFields& qf, AnalysisFields& af) {

    // #pragma omp parallel for num_threads(numprocs) schedule(static)
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const double Qxx = qf.qxx[idx(x, y, z)];
                const double Qxy = qf.qxy[idx(x, y, z)];
                const double Qxz = qf.qxz[idx(x, y, z)];
                const double Qyy = qf.qyy[idx(x, y, z)];
                const double Qyz = qf.qyz[idx(x, y, z)];
                const double p = HalfTrQ2(Qxx, Qxy, Qxz, Qyy, Qyz);
                const double q = DetQ(Qxx, Qxy, Qxz, Qyy, Qyz);

                const double r = 2.0 * std::sqrt(p/3.0);

                const double S = r * std::cos(1.0/3.0 * std::acos(std::clamp(4*q/(r*r*r), -1.0, 1.0)));

                double nhatx = Qxz*(Qyy-S) - Qxy*Qyz;
                double nhaty = Qyz*(Qxx-S) - Qxy*Qxz;
                double nhatz = Qxy*Qxy - (Qxx-S)*(Qyy-S);
                const double norm_inv = 1.0 / std::sqrt(nhatx*nhatx + nhaty*nhaty + nhatz*nhatz);
                nhatx *= norm_inv;
                nhaty *= norm_inv;
                nhatz *= norm_inv;

                af.director_[dirIdx(x, y, z, 0)] = nhatx;
                af.director_[dirIdx(x, y, z, 1)] = nhaty;
                af.director_[dirIdx(x, y, z, 2)] = nhatz;
                af.order_[idx(x, y, z)] = S;
            }
        }
    }
}

