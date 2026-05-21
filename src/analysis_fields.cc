#include "analysis_fields.h"
#include "params.h"
#include <cmath>
using namespace Params;

AnalysisFields::AnalysisFields() :
    rho_past_data_(nx * ny * nz, kDensity),
    ux_past_data_ (nx * ny * nz, 0.0),
    uy_past_data_ (nx * ny * nz, 0.0),
    uz_past_data_ (nx * ny * nz, 0.0),
    order_data_   (nx * ny * nz, 0.0),
    director_data_(nx * ny * nz * 3, 0.0),
    rho_past_(rho_past_data_.data(), nz, ny, nx),
    ux_past_ (ux_past_data_.data(),  nz, ny, nx),
    uy_past_ (uy_past_data_.data(),  nz, ny, nx),
    uz_past_ (uz_past_data_.data(),  nz, ny, nx),
    order_   (order_data_.data(),    nz, ny, nx),
    director_(director_data_.data(), nz, ny, nx, 3)
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

void QtensorToOrderDirector(const QTensorFields& qf, AnalysisFields& af) {

    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const double Qxx = qf.qxx[z, y, x];
                const double Qxy = qf.qxy[z, y, x];
                const double Qxz = qf.qxz[z, y, x];
                const double Qyy = qf.qyy[z, y, x];
                const double Qyz = qf.qyz[z, y, x];
                const double p = HalfTrQ2(Qxx, Qxy, Qxz, Qyy, Qyz);
                const double q = DetQ(Qxx, Qxy, Qxz, Qyy, Qyz);

                const double r = 2.0 * std::sqrt(p/3.0);

                const double S = r * std::cos(1.0/3.0 * std::acos(4*q/(r*r*r)));

                double nhatx = Qxz*(Qyy-S) - Qxy*Qyz;
                double nhaty = Qyz*(Qxx-S) - Qxy*Qxz;
                double nhatz = Qxy*Qxy - (Qxx-S)*(Qyy-S);
                const double norm_inv = 1.0 / std::sqrt(nhatx*nhatx + nhaty*nhaty + nhatz*nhatz);
                nhatx *= norm_inv;
                nhaty *= norm_inv;
                nhatz *= norm_inv;

                af.director_[z, y, x, 0] = nhatx;
                af.director_[z, y, x, 1] = nhaty;
                af.director_[z, y, x, 2] = nhatz;
                af.order_[z, y, x] = S;
            }
        }
    }
}

