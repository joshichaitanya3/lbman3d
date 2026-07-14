#include "params.h"
#include "boundary.h"
#include "physics_helpers.h"

#include <cmath>
#include <random>
#include <ranges>

using namespace Params;

template<typename BC>
void QTensorSolver<BC>::Initialize(QTensorFields& qf) const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> noise_dist(-NOISE, NOISE);
    for (int z : std::views::iota(0, nz)) {
        for (int y : std::views::iota(0, ny)) {
            for (int x : std::views::iota(0, nx)) {
                qf.qxx[idx(x, y, z)] = 0.66 + noise_dist(gen);
                qf.qxy[idx(x, y, z)] = noise_dist(gen);
                qf.qxz[idx(x, y, z)] = noise_dist(gen);
                qf.qyy[idx(x, y, z)] = -0.33 + noise_dist(gen);
                qf.qyz[idx(x, y, z)] = noise_dist(gen);
            }
        }
    }
}

template<typename BC>
void QTensorSolver<BC>::StepAndSetupBodyForce(QTensorFields& qf, FluidFields& ff) const {

    auto compute_cell = [&](int x, int y, int z) {

        // Fields

        const double Qxx = qf.qxx[idx(x, y, z)];
        const double Qxy = qf.qxy[idx(x, y, z)];
        const double Qxz = qf.qxz[idx(x, y, z)];
        const double Qyy = qf.qyy[idx(x, y, z)];
        const double Qyz = qf.qyz[idx(x, y, z)];

        const double ux = ff.ux[idx(x, y, z)];
        const double uy = ff.uy[idx(x, y, z)];
        const double uz = ff.uz[idx(x, y, z)];

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

        // Velocity gradient tensor (central differences with wall-aware ghost
        // values — see boundary_handler.h's VelocityGradientTensor; NOT the
        // Q-stencil xm/xp/ym/yp/zm/zp offsets above, which clamp for
        // Neumann and are wrong for NoSlip/MovingWall and the normal
        // component at SpecularReflection walls)
        const GradTensor nabla_u = VelocityGradientTensor<BC>(ff.ux.data(), ff.uy.data(), ff.uz.data(), x, y, z);
        const double uxx = nabla_u.ux_x;
        const double uxy = nabla_u.ux_y;
        const double uxz = nabla_u.ux_z;

        const double uyx = nabla_u.uy_x;
        const double uyy = nabla_u.uy_y;
        const double uyz = nabla_u.uy_z;

        const double uzx = nabla_u.uz_x;
        const double uzy = nabla_u.uz_y;
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

        // Q-tensor gradient + Laplacian (central/7-point stencil, wall-aware
        // ghost values — see boundary_handler.h's QGradientAndLaplacian; NOT
        // the QXoff/QYoff/QZoff Neumann-only clamp, which is wrong for
        // Anchoring walls)
        const QDerivs dQxx = QGradientAndLaplacian<QComp::XX, BC>(qf.qxx.data(), x, y, z);
        const QDerivs dQxy = QGradientAndLaplacian<QComp::XY, BC>(qf.qxy.data(), x, y, z);
        const QDerivs dQxz = QGradientAndLaplacian<QComp::XZ, BC>(qf.qxz.data(), x, y, z);
        const QDerivs dQyy = QGradientAndLaplacian<QComp::YY, BC>(qf.qyy.data(), x, y, z);
        const QDerivs dQyz = QGradientAndLaplacian<QComp::YZ, BC>(qf.qyz.data(), x, y, z);

        const double Qxxx = dQxx.dx, Qxxy = dQxx.dy, Qxxz = dQxx.dz;
        const double Qxyx = dQxy.dx, Qxyy = dQxy.dy, Qxyz = dQxy.dz;
        const double Qxzx = dQxz.dx, Qxzy = dQxz.dy, Qxzz = dQxz.dz;
        const double Qyyx = dQyy.dx, Qyyy = dQyy.dy, Qyyz = dQyy.dz;
        const double Qyzx = dQyz.dx, Qyzy = dQyz.dy, Qyzz = dQyz.dz;

        const double lap_Qxx = dQxx.lap;
        const double lap_Qxy = dQxy.lap;
        const double lap_Qxz = dQxz.lap;
        const double lap_Qyy = dQyy.lap;
        const double lap_Qyz = dQyz.lap;
        
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

        ff.fx[idx(x, y, z)] = -2.0 * (Hxx*Qxxx + Hxy*Qxyx + Hxz*Qxzx + Hyy*Qyyx + Hyz*Qyzx) + Hxx*Qyyx + Hyy*Qxxx;
        ff.fy[idx(x, y, z)] = -2.0 * (Hxx*Qxxy + Hxy*Qxyy + Hxz*Qxzy + Hyy*Qyyy + Hyz*Qyzy) + Hxx*Qyyy + Hyy*Qxxy;
        ff.fz[idx(x, y, z)] = -2.0 * (Hxx*Qxxz + Hxy*Qxyz + Hxz*Qxzz + Hyy*Qyyz + Hyz*Qyzz) + Hxx*Qyyz + Hyy*Qxxz;

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
        const double Tauxy = (Hxy*Qxx + Hyy*Qxy + Hyz*Qxz) - (Qxy*Hxx + Qyy*Hxy + Qyz*Hxz);
        const double Tauxz = (Hxz*Qxx + Hyz*Qxy + (-Hxx - Hyy)*Qxz) - (Qxz*Hxx + Qyz*Hxy + (-Qxx - Qyy)*Hxz);
        const double Tauyy = 0.0; // Diagonal component of antisymmetric tensor
        const double Tauyz = (Hxz*Qxy + Hyz*Qyy + (-Hxx - Hyy)*Qyz) - (Qxz*Hxy + Qyz*Hyy + (-Qxx - Qyy)*Hyz);

        // Update nematic stress (passive + active)
        qf.Pxx[idx(x, y, z)] = -ktwo_thirds * LAMBDA * Hxx - LAMBDA * QHxx + Tauxx;
        qf.Pxy[idx(x, y, z)] = -ktwo_thirds * LAMBDA * Hxy - LAMBDA * QHxy + Tauxy;
        qf.Pxz[idx(x, y, z)] = -ktwo_thirds * LAMBDA * Hxz - LAMBDA * QHxz + Tauxz;
        qf.Pyy[idx(x, y, z)] = -ktwo_thirds * LAMBDA * Hyy - LAMBDA * QHyy + Tauyy;
        qf.Pyz[idx(x, y, z)] = -ktwo_thirds * LAMBDA * Hyz - LAMBDA * QHyz + Tauyz;

        // Now, we perform the timestep

        qf.qxx_new[idx(x, y, z)] = Qxx + DT*(adv_xx + cor_xx + LAMBDA * (ktwo_thirds * Exx + aln2_xx) + GAMMA * Hxx);
        qf.qxy_new[idx(x, y, z)] = Qxy + DT*(adv_xy + cor_xy + LAMBDA * (ktwo_thirds * Exy + aln2_xy) + GAMMA * Hxy);
        qf.qxz_new[idx(x, y, z)] = Qxz + DT*(adv_xz + cor_xz + LAMBDA * (ktwo_thirds * Exz + aln2_xz) + GAMMA * Hxz);
        qf.qyy_new[idx(x, y, z)] = Qyy + DT*(adv_yy + cor_yy + LAMBDA * (ktwo_thirds * Eyy + aln2_yy) + GAMMA * Hyy);
        qf.qyz_new[idx(x, y, z)] = Qyz + DT*(adv_yz + cor_yz + LAMBDA * (ktwo_thirds * Eyz + aln2_yz) + GAMMA * Hyz);

    };

    // Single parallel loop over the full domain. Wall-aware ghost values
    // (QGradientAndLaplacian) are resolved inline per point, including at
    // boundary nodes — no separate post-step anchoring pass needed; see
    // boundary_handler.h's header comment for why that used to exist and
    // why it doesn't anymore.
    #pragma omp parallel for num_threads(numprocs) schedule(static)
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                compute_cell(x, y, z);
            }
        }
    }

    UpdateQnewWithQ(qf);
}

template<typename BC>
void QTensorSolver<BC>::UpdateQnewWithQ(QTensorFields& qf) const {
    qf.SwapWithNew();
}

template<typename BC>
void QTensorSolver<BC>::SetActiveStressAndComputeBodyForce(FluidFields& ff, const QTensorFields& qf) const {
    auto compute_cell = [&](int x, int y, int z, int xm, int xp, int ym, int yp, int zm, int zp) {
        
        // First, add the active force. Q's own gradient — wall-aware ghost
        // values (QGradientAndLaplacian), not the QXoff/QYoff/QZoff
        // Neumann-only clamp (still correct/used below for P, which has no
        // prescribed wall target of its own).
        const QDerivs dQxx = QGradientAndLaplacian<QComp::XX, BC>(qf.qxx.data(), x, y, z);
        const QDerivs dQxy = QGradientAndLaplacian<QComp::XY, BC>(qf.qxy.data(), x, y, z);
        const QDerivs dQxz = QGradientAndLaplacian<QComp::XZ, BC>(qf.qxz.data(), x, y, z);
        const QDerivs dQyy = QGradientAndLaplacian<QComp::YY, BC>(qf.qyy.data(), x, y, z);
        const QDerivs dQyz = QGradientAndLaplacian<QComp::YZ, BC>(qf.qyz.data(), x, y, z);

        ff.fx[idx(x, y, z)] += -ALPHA * (dQxx.dx + dQxy.dy + dQxz.dz);

        ff.fy[idx(x, y, z)] += -ALPHA * (dQxy.dx + dQyy.dy + dQyz.dz);

        ff.fz[idx(x, y, z)] += -ALPHA * (dQxz.dx + dQyz.dy - dQxx.dz - dQyy.dz); // Since Pzz = -(Pxx + Pyy)

        // Now, add the passive stress and friction
        ff.fx[idx(x, y, z)] += ((qf.Pxx[idx(xp, y, z)] - qf.Pxx[idx(xm, y, z)])/2.0
                         + (qf.Pxy[idx(x, yp, z)] - qf.Pxy[idx(x, ym, z)])/2.0
                         + (qf.Pxz[idx(x, y, zp)] - qf.Pxz[idx(x, y, zm)])/2.0)
                         -MU * ff.ux[idx(x, y, z)];

        ff.fy[idx(x, y, z)] += ((qf.Pxy[idx(xp, y, z)] - qf.Pxy[idx(xm, y, z)])/2.0
                         + (qf.Pyy[idx(x, yp, z)] - qf.Pyy[idx(x, ym, z)])/2.0
                         + (qf.Pyz[idx(x, y, zp)] - qf.Pyz[idx(x, y, zm)])/2.0)
                         -MU * ff.uy[idx(x, y, z)];

        ff.fz[idx(x, y, z)] += ((qf.Pxz[idx(xp, y, z)] - qf.Pxz[idx(xm, y, z)])/2.0
                         + (qf.Pyz[idx(x, yp, z)] - qf.Pyz[idx(x, ym, z)])/2.0
                         - (qf.Pxx[idx(x, y, zp)] - qf.Pxx[idx(x, y, zm)])/2.0
                         - (qf.Pyy[idx(x, y, zp)] - qf.Pyy[idx(x, y, zm)])/2.0) // Since Pzz = -(Pxx + Pyy)
                         -MU * ff.uz[idx(x, y, z)];
    };

    #pragma omp parallel for default(shared) num_threads(numprocs) schedule(static)
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const int xm = QXoff<BC>(x, -1);
                const int xp = QXoff<BC>(x, 1);
                const int ym = QYoff<BC>(y, -1);
                const int yp = QYoff<BC>(y, 1);
                const int zm = QZoff<BC>(z, -1);
                const int zp = QZoff<BC>(z, 1);
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
