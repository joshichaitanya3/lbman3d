#include "params.h"
#include "boundary.h"

#include <cmath>
#include <random>
#include <ranges>

using namespace Params;

template<typename BC>
QTensorSolver<BC>::QTensorSolver(Grid<BC> grid) : grid_(std::move(grid)) {}

template<typename BC>
void QTensorSolver<BC>::Initialize(QTensorFields& qf) const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> noise_dist(-NOISE, NOISE);
    for (int z : std::views::iota(0, nz)) {
        for (int y : std::views::iota(0, ny)) {
            for (int x : std::views::iota(0, nx)) {
                qf.qxx(z, y, x) = 0.66 + noise_dist(gen);
                qf.qxy(z, y, x) = noise_dist(gen);
                qf.qxz(z, y, x) = noise_dist(gen);
                qf.qyy(z, y, x) = -0.33 + noise_dist(gen);
                qf.qyz(z, y, x) = noise_dist(gen);
            }
        }
    }
}

template<typename BC>
void QTensorSolver<BC>::StepAndSetupBodyForce(QTensorFields& qf, FluidFields& ff) const {

    auto compute_cell = [&](int x, int y, int z, int xm, int xp, int ym, int yp, int zm, int zp) {

        // Fields

        const double Qxx = qf.qxx(z, y, x);
        const double Qxy = qf.qxy(z, y, x);
        const double Qxz = qf.qxz(z, y, x);
        const double Qyy = qf.qyy(z, y, x);
        const double Qyz = qf.qyz(z, y, x);

        const double ux = ff.ux(z, y, x);
        const double uy = ff.uy(z, y, x);
        const double uz = ff.uz(z, y, x);

        // Polynomials
        const double TrQ2 = 2.0*(Qxx*Qxx + Qyy*Qyy+ Qxx*Qyy + Qxy*Qxy +Qxz*Qxz +Qyz*Qyz);
        
        // Q2 = Q_ik * Q_kj - 1/3 TrQ2 \delta_ij:
        const double kone_thirds = 1.0/3.0;
        const double Q2_xx = Qxx*Qxx + Qxy*Qxy + Qxz*Qxz - kone_thirds * TrQ2;
        const double Q2_xy = Qxx*Qxy + Qxy*Qyy + Qxz*Qyz;
        const double Q2_xz = Qxy*Qyz - Qxz*Qyy;
        const double Q2_yy = Qxy*Qxy + Qyy*Qyy + Qyz*Qyz - kone_thirds * TrQ2;
        const double Q2_yz = Qxy*Qxz - Qyz*Qxx;
        
        // First-order derivatives 

        // Velocity gradient tensor (central differences; uses the same Q stencil offsets
        // since HandleBoundaries has already set correct wall velocities from the
        // previous LBM step before StepAndSetupBodyForce is called)
        const double uxx = (ff.ux(z,y,xp) - ff.ux(z,y,xm)) / 2.0;
        const double uxy = (ff.ux(z,yp,x) - ff.ux(z,ym,x)) / 2.0;
        const double uxz = (ff.ux(zp,y,x) - ff.ux(zm,y,x)) / 2.0;

        const double uyx = (ff.uy(z,y,xp) - ff.uy(z,y,xm)) / 2.0;
        const double uyy = (ff.uy(z,yp,x) - ff.uy(z,ym,x)) / 2.0;
        const double uyz = (ff.uy(zp,y,x) - ff.uy(zm,y,x)) / 2.0;

        const double uzx = (ff.uz(z,y,xp) - ff.uz(z,y,xm)) / 2.0;
        const double uzy = (ff.uz(z,yp,x) - ff.uz(z,ym,x)) / 2.0;
        const double uzz = - (uxx + uyy); // from incompressibility
        
        /*
            * Exx:
            * Derivative(ux(x, y, z), x)
            * Exy:
            * Derivative(ux(x, y, z), y)/2 + Derivative(uy(x, y, z), x)/2
            * Exz:
            * Derivative(ux(x, y, z), z)/2 + Derivative(uz(x, y, z), x)/2
            * Eyy:
            * Derivative(uy(x, y, z), y)
            * Eyz:
            * Derivative(uy(x, y, z), z)/2 + Derivative(uz(x, y, z), y)/2
            */

        const double Exx = uxx;
        const double Exy = 0.5 * (uxy + uyx);
        const double Exz = 0.5 * (uxz + uzx);
        const double Eyy = uyy;
        const double Eyz = 0.5 * (uyz + uzy);

        /*
        Omegaxy = 1/2(Uxy - Uyx)
        */
        const double Wxy = 0.5 * (uxy - uyx);
        const double Wxz = 0.5 * (uxz - uzx);
        const double Wyx = -Wxy;
        const double Wyz = 0.5 * (uyz - uzy);
        const double Wzx = -Wxz;
        const double Wzy = -Wyz;

        // Q-tensor
        const double Qxxx = (qf.qxx(z, y, xp) - qf.qxx(z, y, xm)) / 2.0;
        const double Qxyx = (qf.qxy(z, y, xp) - qf.qxy(z, y, xm)) / 2.0;
        const double Qxzx = (qf.qxz(z, y, xp) - qf.qxz(z, y, xm)) / 2.0;
        const double Qyyx = (qf.qyy(z, y, xp) - qf.qyy(z, y, xm)) / 2.0;
        const double Qyzx = (qf.qyz(z, y, xp) - qf.qyz(z, y, xm)) / 2.0;

        const double Qxxy = (qf.qxx(z, yp, x) - qf.qxx(z, ym, x)) / 2.0;
        const double Qxyy = (qf.qxy(z, yp, x) - qf.qxy(z, ym, x)) / 2.0;
        const double Qxzy = (qf.qxz(z, yp, x) - qf.qxz(z, ym, x)) / 2.0;
        const double Qyyy = (qf.qyy(z, yp, x) - qf.qyy(z, ym, x)) / 2.0;
        const double Qyzy = (qf.qyz(z, yp, x) - qf.qyz(z, ym, x)) / 2.0;

        const double Qxxz = (qf.qxx(zp, y, x) - qf.qxx(zm, y, x)) / 2.0;
        const double Qxyz = (qf.qxy(zp, y, x) - qf.qxy(zm, y, x)) / 2.0;
        const double Qxzz = (qf.qxz(zp, y, x) - qf.qxz(zm, y, x)) / 2.0;
        const double Qyyz = (qf.qyy(zp, y, x) - qf.qyy(zm, y, x)) / 2.0;
        const double Qyzz = (qf.qyz(zp, y, x) - qf.qyz(zm, y, x)) / 2.0;

        // Laplacian (seven-point stencil)
        const double lap_Qxx = qf.qxx(z,y,xp) + qf.qxx(z,y,xm) + qf.qxx(z,yp,x) + qf.qxx(z,ym,x) + qf.qxx(zp,y,x) + qf.qxx(zm,y,x) - 6.0*Qxx;
        const double lap_Qxy = qf.qxy(z,y,xp) + qf.qxy(z,y,xm) + qf.qxy(z,yp,x) + qf.qxy(z,ym,x) + qf.qxy(zp,y,x) + qf.qxy(zm,y,x) - 6.0*Qxy;
        const double lap_Qxz = qf.qxz(z,y,xp) + qf.qxz(z,y,xm) + qf.qxz(z,yp,x) + qf.qxz(z,ym,x) + qf.qxz(zp,y,x) + qf.qxz(zm,y,x) - 6.0*Qxz;
        const double lap_Qyy = qf.qyy(z,y,xp) + qf.qyy(z,y,xm) + qf.qyy(z,yp,x) + qf.qyy(z,ym,x) + qf.qyy(zp,y,x) + qf.qyy(zm,y,x) - 6.0*Qyy;
        const double lap_Qyz = qf.qyz(z,y,xp) + qf.qyz(z,y,xm) + qf.qyz(z,yp,x) + qf.qyz(z,ym,x) + qf.qyz(zp,y,x) + qf.qyz(zm,y,x) - 6.0*Qyz;
        
        // Advection: -u · ∇Q
        const double adv_xx = -(ux * Qxxx + uy * Qxxy + uz * Qxxz);
        const double adv_xy = -(ux * Qxyx + uy * Qxyy + uz * Qxyz);
        const double adv_xz = -(ux * Qxzx + uy * Qxzy + uz * Qxzz);
        const double adv_yy = -(ux * Qyyx + uy * Qyyy + uz * Qyyz);
        const double adv_yz = -(ux * Qyzx + uy * Qyzy + uz * Qyzz);
        
        // ##############################################################################
        // #   corotation       [(Omega Q - Q Omega)_ij]
        // ##############################################################################
        
        const double cor_xx = 2.0 * (Qxy * Wxy + Qxz * Wxz);
        const double cor_xy = -Qxx * Wxy + Qxz * Wyz + Qyy * Wxy + Qyz * Wxz;
        const double cor_xz = -2.0 * Qxx * Wxz - Qxy * Wyz - Qyy * Wxz + Qyz * Wxy;
        const double cor_yy = 2.0 * (-Qxy * Wxy + Qyz * Wyz);
        const double cor_yz = -Qxx * Wyz -Qxy * Wxz -Qxz * Wxy - 2.0 * Qyy * Wyz;
        
        /* ##############################################################################
        // #   higher order order flow alignment   lambda [(E Q + Q E)_ij]
        1->xx, 2->xy, 3->xz, 4->yy, 5->yz
        QE_xx = e_1 Q_1 + e_2 Q_2 + e_3 Q_3
        QE_xy = e_2 Q_1 + e_4 Q_2 + e_5 Q_3
        QE_xz = e_3 Q_1 + e_5 Q_2 + (-e_1 - e_4) Q_3
        QE_yy = e_2 Q_2 + e_4 Q_4 + e_5 Q_5
        QE_yz = e_3 Q_2 + e_5 Q_4 + (-e_1 - e_4) Q_5
        
        Q:E (trace): (2 e_1 + e_4) Q_1 + 2 e_2 Q_2 + 2 e_3 Q_3 + e_1 Q_4 + 2 e_4 Q_4 + 2 e_5 Q_5
        // ##############################################################################
        */
        const double ktwo_thirds = 2.0/3.0;
        
        const double tr_QE = (2.0 * Exx + Eyy) * Qxx + (2.0 * Exy) * Qxy + (2.0 * Exz) * Qxz + (Exx + 2.0 * Eyy) * Qyy + 2.0 * Eyz * Qyz;

        const double aln2_xx = 2.0 * (Exx * Qxx + Exy * Qxy + Exz * Qxz) - ktwo_thirds * tr_QE;
        const double aln2_xy = Exy * Qxx + Eyy * Qxy + Eyz * Qxz
                                + Qxy * Exx + Qyy * Exy + Qyz * Exz;
        const double aln2_xz = Exz * Qxx + Eyz * Qxy + (-Exx - Eyy) * Qxz
                                + Qxz * Exx + Qyz * Exy + (-Qxx - Qyy) * Exz;
        
        const double aln2_yy = 2.0 * (Exy * Qxy + Eyy * Qyy + Eyz * Qyz) - ktwo_thirds * tr_QE;
        const double aln2_yz = Exz * Qxy + Eyz * Qyy + (-Exx - Eyy) * Qyz
                                + Qxz * Exy + Qyz * Eyy + (-Qxx - Qyy) * Eyz;
        
        // Molecular field H
        const double Hxx = L * lap_Qxx - A * Qxx - B * Q2_xx - C * Qxx * TrQ2;
        const double Hxy = L * lap_Qxy - A * Qxy - B * Q2_xy - C * Qxy * TrQ2;
        const double Hxz = L * lap_Qxz - A * Qxz - B * Q2_xz - C * Qxz * TrQ2;
        const double Hyy = L * lap_Qyy - A * Qyy - B * Q2_yy - C * Qyy * TrQ2;
        const double Hyz = L * lap_Qyz - A * Qyz - B * Q2_yz - C * Qyz * TrQ2;
        
        // Add the advective counter part of the back-flow to the body force, H:\nabla Q
        // since this does not come from the divergence of the stress tensor.
        // The backflow from the divergence will be added to this by SetActiveStressAndComputeBodyForce

        ff.fx(z, y, x) = -2.0 * (Hxx*Qxxx + Hxy*Qxyx + Hxz*Qxzx + Hyy*Qyyx + Hyz*Qyzx) + Hxx*Qyyx + Hyy*Qxxx;
        ff.fy(z, y, x) = -2.0 * (Hxx*Qxxy + Hxy*Qxyy + Hxz*Qxzy + Hyy*Qyyy + Hyz*Qyzy) + Hxx*Qyyy + Hyy*Qxxy;
        ff.fz(z, y, x) = -2.0 * (Hxx*Qxxz + Hxy*Qxyz + Hxz*Qxzz + Hyy*Qyyz + Hyz*Qyzz) + Hxx*Qyyz + Hyy*Qxxz;

        // Now, update the nematic stress tensor

        // Counterpart of higher order order flow alignment in the stress-tensor:
        // lambda [(H Q + Q H)_ij]
        // 1->xx, 2->xy, 3->xz, 4->yy, 5->yz
        // QH_xx = H_1 Q_1 + H_2 Q_2 + H_3 Q_3
        // QH_xy = H_2 Q_1 + H_4 Q_2 + H_5 Q_3
        // QH_xz = H_3 Q_1 + H_5 Q_2 + (-H_1 - H_4) Q_3
        // QH_yy = H_2 Q_2 + H_4 Q_4 + H_5 Q_5
        // QH_yz = H_3 Q_2 + H_5 Q_4 + (-H_1 - H_4) Q_5
        // We will call the symmetric-traceless part QH and the antisymmetric part Tau

        
        const double tr_QH = (2.0 * Hxx + Hyy) * Qxx + (2.0 * Hxy) * Qxy + (2.0 * Hxz) * Qxz + (Hxx + 2.0 * Hyy) * Qyy + 2.0 * Hyz * Qyz;

        const double QHxx = 2.0 * (Hxx * Qxx + Hxy * Qxy + Hxz * Qxz) - ktwo_thirds * tr_QH;
        const double QHxy = Hxy * Qxx + Hyy * Qxy + Hyz * Qxz
                                + Qxy * Hxx + Qyy * Hxy + Qyz * Hxz;
        const double QHxz = Hxz * Qxx + Hyz * Qxy + (-Hxx - Hyy) * Qxz
                                + Qxz * Hxx + Qyz * Hxy + (-Qxx - Qyy) * Hxz;
        
        const double QHyy = 2.0 * (Hxy * Qxy + Hyy * Qyy + Hyz * Qyz) - ktwo_thirds * tr_QH;
        const double QHyz = Hxz * Qxy + Hyz * Qyy + (-Hxx - Hyy) * Qyz
                                + Qxz * Hxy + Qyz * Hyy + (-Qxx - Qyy) * Hyz;
        
        const double Tauxx = 0.0; // Diagonal component of antisymmetric tensor
        const double Tauxy = (Hxy*Qxx + Hyy*Qxy + Hyz*Qxz) - (Qxy*Hxx + Qyy*Hxy + Qyz*Qxz);
        const double Tauxz = (Hxz*Qxx + Hyz*Qxy + (-Hxx - Hyy)*Qxz) - (Qxz*Hxx + Qyz*Hxy + (-Qxx - Qyy)*Hxz);
        const double Tauyy = 0.0; // Diagonal component of antisymmetric tensor
        const double Tauyz = (Hxz*Qxy + Hyz*Qyy + (-Hxx - Hyy)*Qyz) - (Qxz*Hxy + Qyz*Hyy + (-Qxx - Qyy)*Hyz);

        // Update nematic stress (passive + active)
        qf.Pxx(z, y, x) = -ktwo_thirds * LAMBDA * Hxx - LAMBDA * QHxx + Tauxx;
        qf.Pxy(z, y, x) = -ktwo_thirds * LAMBDA * Hxy - LAMBDA * QHxy + Tauxy;
        qf.Pxz(z, y, x) = -ktwo_thirds * LAMBDA * Hxz - LAMBDA * QHxz + Tauxz;
        qf.Pyy(z, y, x) = -ktwo_thirds * LAMBDA * Hyy - LAMBDA * QHyy + Tauyy;
        qf.Pyz(z, y, x) = -ktwo_thirds * LAMBDA * Hyz - LAMBDA * QHyz + Tauyz;

        // Now, we perform the timestep

        qf.qxx_new(z, y, x) = Qxx + DT*(adv_xx + cor_xx + LAMBDA * (ktwo_thirds * Exx + aln2_xx) + GAMMA * Hxx);
        qf.qxy_new(z, y, x) = Qxy + DT*(adv_xy + cor_xy + LAMBDA * (ktwo_thirds * Exy + aln2_xy) + GAMMA * Hxy);
        qf.qxz_new(z, y, x) = Qxz + DT*(adv_xz + cor_xz + LAMBDA * (ktwo_thirds * Exz + aln2_xz) + GAMMA * Hxz);
        qf.qyy_new(z, y, x) = Qyy + DT*(adv_yy + cor_yy + LAMBDA * (ktwo_thirds * Eyy + aln2_yy) + GAMMA * Hyy);
        qf.qyz_new(z, y, x) = Qyz + DT*(adv_yz + cor_yz + LAMBDA * (ktwo_thirds * Eyz + aln2_yz) + GAMMA * Hyz);

    };
    
    // Single parallel loop over the full domain. Ghost-node stencil offsets are
    // resolved inline; anchoring BCs are applied per-point after the FD step.
    #pragma omp parallel for num_threads(numprocs) schedule(static)
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const int xm = (x == 0)    ? QXoff(0, -1)   : x - 1;
                const int xp = (x == nx-1) ? QXoff(nx-1, 1) : x + 1;
                const int ym = (y == 0)    ? QYoff(0, -1)   : y - 1;
                const int yp = (y == ny-1) ? QYoff(ny-1, 1) : y + 1;
                const int zm = (z == 0)    ? QZoff(0, -1)   : z - 1;
                const int zp = (z == nz-1) ? QZoff(nz-1, 1) : z + 1;

                total_nematic_energy += compute_cell(x, y, z, xm, xp, ym, yp, zm, zp);

                // Apply anchoring BCs pointwise after the timestep.
                // X walls first, Z walls last — Z-wall writes take precedence at edges/corners.
                if (x == 0)    HandleQBoundaryPoint<typename BC::XLo>(qf, x, y, z);
                if (x == nx-1) HandleQBoundaryPoint<typename BC::XHi>(qf, x, y, z);
                if (y == 0)    HandleQBoundaryPoint<typename BC::YLo>(qf, x, y, z);
                if (y == ny-1) HandleQBoundaryPoint<typename BC::YHi>(qf, x, y, z);
                if (z == 0)    HandleQBoundaryPoint<typename BC::ZLo>(qf, x, y, z);
                if (z == nz-1) HandleQBoundaryPoint<typename BC::ZHi>(qf, x, y, z);
            }
        }
    }

    UpdateQnewWithQ(qf);
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-point Q-tensor boundary handler
//
// Neumann/Periodic: compile-time no-op — the stencil clamping/wrapping in
// QXoff/QYoff/QZoff already enforces ∂Q/∂n = 0 at the correct wall position.
//
// Anchoring<S,θ,φ>: overwrites q_new at (z,y,x) with the prescribed
// strong-anchoring value after the FD step. On the next step the Laplacian of
// adjacent interior cells reads this fixed value, giving the correct Dirichlet
// influence.
// ─────────────────────────────────────────────────────────────────────────────

template<typename BC>
template<typename WallSpec>
void QTensorSolver<BC>::HandleQBoundaryPoint(QTensorFields& qf, int x, int y, int z) const {
    using Q = typename WallSpec::QBC;
    if constexpr (is_anchoring_v<Q>) {
        const double sin_sq_phi   = 0.5 * (1.0 - std::cos(2.0 * Q::phi));
        const double cos_sq_phi   = 1.0 - sin_sq_phi;
        const double sin_phi_cos_phi = 0.5 * std::sin(2.0 * Q::phi);
        const double sin_sq_th    = 0.5 * (1.0 - std::cos(2.0 * Q::theta));
        const double sin_th_cos_th = 0.5 * std::sin(2.0 * Q::theta);

        const double S = Q::s;
        qf.qxx_new(z, y, x) = S * (cos_sq_phi * sin_sq_th - 1.0/3.0);
        qf.qxy_new(z, y, x) = S * (sin_phi_cos_phi * sin_sq_th);
        qf.qxz_new(z, y, x) = S * (std::cos(Q::phi) * sin_th_cos_th);
        qf.qyy_new(z, y, x) = S * (sin_sq_phi * sin_sq_th - 1.0/3.0);
        qf.qyz_new(z, y, x) = S * (std::sin(Q::phi) * sin_th_cos_th);
    }
}

template<typename BC>
void QTensorSolver<BC>::UpdateQnewWithQ(QTensorFields& qf) const {
    qf.SwapWithNew();
}

template<typename BC>
void QTensorSolver<BC>::SetActiveStressAndComputeBodyForce(FluidFields& ff, const QTensorFields& qf) const {
    auto compute_cell = [&](int x, int y, int z, int xm, int xp, int ym, int yp, int zm, int zp) {
        
        // First, add the active force.
        ff.fx(z, y, x) += -ALPHA * (
                           (qf.qxx(z, y, xp) - qf.qxx(z, y, xm))/2.0
                         + (qf.qxy(z, yp, x) - qf.qxy(z, ym, x))/2.0
                         + (qf.qxz(zp, y, x) - qf.qxz(zm, y, x))/2.0);

        ff.fy(z, y, x) += -ALPHA * (
                           (qf.qxy(z, y, xp) - qf.qxy(z, y, xm))/2.0
                         + (qf.qyy(z, yp, x) - qf.qyy(z, ym, x))/2.0
                         + (qf.qyz(zp, y, x) - qf.qyz(zm, y, x))/2.0);

        ff.fz(z, y, x) += -ALPHA * (
                           (qf.qxz(z, y, xp) - qf.qxz(z, y, xm))/2.0
                         + (qf.qyz(z, yp, x) - qf.qyz(z, ym, x))/2.0
                         - (qf.qxx(zp, y, x) - qf.qxx(zm, y, x))/2.0
                         - (qf.qyy(zp, y, x) - qf.qyy(zm, y, x))/2.0); // Since Pzz = -(Pxx + Pyy)
        
        // Now, add the passive stress and friction
        ff.fx(z, y, x) += ((qf.Pxx(z, y, xp) - qf.Pxx(z, y, xm))/2.0
                         + (qf.Pxy(z, yp, x) - qf.Pxy(z, ym, x))/2.0
                         + (qf.Pxz(zp, y, x) - qf.Pxz(zm, y, x))/2.0)
                         -MU * ff.ux(z, y, x);

        ff.fy(z, y, x) += ((qf.Pxy(z, y, xp) - qf.Pxy(z, y, xm))/2.0
                         + (qf.Pyy(z, yp, x) - qf.Pyy(z, ym, x))/2.0
                         + (qf.Pyz(zp, y, x) - qf.Pyz(zm, y, x))/2.0)
                         -MU * ff.uy(z, y, x);

        ff.fz(z, y, x) += ((qf.Pxz(z, y, xp) - qf.Pxz(z, y, xm))/2.0
                         + (qf.Pyz(z, yp, x) - qf.Pyz(z, ym, x))/2.0
                         - (qf.Pxx(zp, y, x) - qf.Pxx(zm, y, x))/2.0
                         - (qf.Pyy(zp, y, x) - qf.Pyy(zm, y, x))/2.0) // Since Pzz = -(Pxx + Pyy)
                         -MU * ff.uz(z, y, x);
    };

    #pragma omp parallel for default(shared) num_threads(numprocs) schedule(static)
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const int xm = (x == 0)    ? QXoff(0, -1)   : x - 1;
                const int xp = (x == nx-1) ? QXoff(nx-1, 1) : x + 1;
                const int ym = (y == 0)    ? QYoff(0, -1)   : y - 1;
                const int yp = (y == ny-1) ? QYoff(ny-1, 1) : y + 1;
                const int zm = (z == 0)    ? QZoff(0, -1)   : z - 1;
                const int zp = (z == nz-1) ? QZoff(nz-1, 1) : z + 1;
                compute_cell(x, y, z, xm, xp, ym, yp, zm, zp);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step drives three OMP parallel loops per timestep (one inside each of the
// two methods below, one inside LbmSolver::LatticeBoltzmannStep).
//
// Loop 1 (StepAndSetupBodyForce) and Loop 2 (SetActiveStressAndComputeBodyForce)
// cannot be merged: Loop 2 reads ∇·P at neighbor nodes, which Loop 1 writes,
// so a full barrier between them is load-bearing.
//
// Loop 2 and Loop 3 (LbmSolver::LatticeBoltzmannStep) CAN be merged.
// Loop 2 writes ff.fx/fy/fz; Loop 3 reads them. Both loops read qf.qxx/Pxx at
// neighbor nodes (written only by Loop 1, untouched after that). The per-node
// ordering "compute force, then collide/stream" is correct since ff.ux used for
// friction is the previous step's velocity — still present when Loop 2 runs.
//
// The bandwidth saving scales with grid size. At 512³, ff.fx/fy/fz total ~3.2 GB;
// in separate loops they are written then re-read, wasting ~6.4 GB of memory
// traffic per step. Merging avoids this entirely.
//
// To merge without coupling LbmSolver to QTensorFields, template
// LatticeBoltzmannStep with a zero-cost force functor:
//
//   template<typename ForceSetup = NoExtraForce>
//   void LatticeBoltzmannStep(FluidFields& ff, ForceSetup setup = {}) const;
//
// and call setup(ff, z, y, x) at the top of the inner loop before the forcing
// term. ActiveNematicSim::Step() passes a lambda capturing qtensor_ that calls
// QTensorSolver::ApplyForceAtNode (the per-node body of compute_cell in
// SetActiveStressAndComputeBodyForce). At 128×64×64 the benefit is small
// (ff.fx/fy/fz fit in L3 cache). At 512³ it is worth pursuing.
// ─────────────────────────────────────────────────────────────────────────────
template<typename BC>
void QTensorSolver<BC>::Step(QTensorFields& qf, FluidFields& ff) const {
    StepAndSetupBodyForce(qf, ff);
    SetActiveStressAndComputeBodyForce(ff, qf);
}
