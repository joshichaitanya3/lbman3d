#include <params.h>
#include "boundary.h"
#include "physics_helpers.h"
#include "model.h"
#include "local_grid.h"

#include <cmath>
#include <random>
#include <ranges>

using namespace Params;

template<typename BC>
void QTensorSolver<BC>::Initialize(QTensorFields& qf) const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> noise_dist(-NOISE, NOISE);

    const LocalGrid& g = qf.grid;
    for (int z : std::views::iota(0, g.local_nz)) {
        for (int y : std::views::iota(0, g.local_ny)) {
            for (int x : std::views::iota(0, g.local_nx)) {
                const int idxp = g.halo_idx(x, y, z);
                qf.qxx[idxp] = 0.66 + noise_dist(gen);
                qf.qxy[idxp] = noise_dist(gen);
                qf.qxz[idxp] = noise_dist(gen);
                qf.qyy[idxp] = -0.33 + noise_dist(gen);
                qf.qyz[idxp] = noise_dist(gen);
            }
        }
    }
}

template<typename BC>
void QTensorSolver<BC>::StepAndSetupBodyForce(QTensorFields& qf, FluidFields& ff) const {
    
    const LocalGrid& g = qf.grid;

    auto compute_cell = [&](int x, int y, int z) {
        
        // Fields
        const int idxp = g.halo_idx(x, y, z);

        const SymTrLessTensor5 Q{
            qf.qxx[idxp],
            qf.qxy[idxp],
            qf.qxz[idxp],
            qf.qyy[idxp],
            qf.qyz[idxp]
        };
        
        const Vec3 u{
            ff.ux[idxp],
            ff.uy[idxp],
            ff.uz[idxp]
        };
        
        const QDerivs dQxx = QGradientAndLaplacian<QComp::XX, BC>(qf.qxx.data(), x, y, z, g);
        const QDerivs dQxy = QGradientAndLaplacian<QComp::XY, BC>(qf.qxy.data(), x, y, z, g);
        const QDerivs dQxz = QGradientAndLaplacian<QComp::XZ, BC>(qf.qxz.data(), x, y, z, g);
        const QDerivs dQyy = QGradientAndLaplacian<QComp::YY, BC>(qf.qyy.data(), x, y, z, g);
        const QDerivs dQyz = QGradientAndLaplacian<QComp::YZ, BC>(qf.qyz.data(), x, y, z, g);
        
        const GradTensor nabla_u = VelocityGradientTensor<BC>(ff.ux.data(), ff.uy.data(), ff.uz.data(), x, y, z, g);

        const QStencil qs{
            Q, u, dQxx, dQxy, dQxz, dQyy, dQyz, nabla_u
        };
        
        SymTrLessTensor5 q_new, sigma;
        AntiSymTensor3 tau;
        Vec3 advective_backflow;

        PointwiseStepAndSetupBodyForce(
            qs,
            q_new,
            sigma,
            tau,
            advective_backflow
        );
        
        // Add the advective counter part of the back-flow to the body force, H:\nabla Q
        // since this does not come from the divergence of the stress tensor.
        // The backflow from the divergence will be added to this by SetActiveStressAndComputeBodyForce

        ff.fx[idxp] = advective_backflow.x;
        ff.fy[idxp] = advective_backflow.y;
        ff.fz[idxp] = advective_backflow.z;


        // Update nematic stress (passive + active), symmetric-traceless part
        qf.Sigma_xx[idxp] = sigma.xx;
        qf.Sigma_xy[idxp] = sigma.xy;
        qf.Sigma_xz[idxp] = sigma.xz;
        qf.Sigma_yy[idxp] = sigma.yy;
        qf.Sigma_yz[idxp] = sigma.yz;

        // Antisymmetric (torque-carrying) part, stored separately so the
        // divergence in SetActiveStressAndComputeBodyForce can apply
        // A_beta,alpha = -A_alpha,beta
        qf.Tau_xy[idxp] = tau.xy;
        qf.Tau_xz[idxp] = tau.xz;
        qf.Tau_yz[idxp] = tau.yz;

        // Now, we perform the timestep

        qf.qxx_new[idxp] = q_new.xx;
        qf.qxy_new[idxp] = q_new.xy;
        qf.qxz_new[idxp] = q_new.xz;
        qf.qyy_new[idxp] = q_new.yy;
        qf.qyz_new[idxp] = q_new.yz;

    };

    // Single parallel loop over the full domain. Wall-aware ghost values
    // (QGradientAndLaplacian) are resolved inline per point, including at
    // boundary nodes — no separate post-step anchoring pass needed; see
    // boundary_handler.h's header comment for why that used to exist and
    // why it doesn't anymore.
    #pragma omp parallel for num_threads(kNumOMPThreads) schedule(static)
    for (int z = 0; z < g.local_nz; ++z) {
        for (int y = 0; y < g.local_ny; ++y) {
            for (int x = 0; x < g.local_nx; ++x) {
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
    
    const LocalGrid& g = qf.grid;

    auto compute_cell = [&](int x, int y, int z) {
        
        const int idxp = g.halo_idx(x, y, z);

        // First, add the active force. Q's own gradient — wall-aware ghost
        // values (QGradientAndLaplacian), not the QXoff/QYoff/QZoff
        // Neumann-only clamp (still correct/used below for P, which has no
        // prescribed wall target of its own).
        const QDerivs dQxx = QGradientAndLaplacian<QComp::XX, BC>(qf.qxx.data(), x, y, z, g);
        const QDerivs dQxy = QGradientAndLaplacian<QComp::XY, BC>(qf.qxy.data(), x, y, z, g);
        const QDerivs dQxz = QGradientAndLaplacian<QComp::XZ, BC>(qf.qxz.data(), x, y, z, g);
        const QDerivs dQyy = QGradientAndLaplacian<QComp::YY, BC>(qf.qyy.data(), x, y, z, g);
        const QDerivs dQyz = QGradientAndLaplacian<QComp::YZ, BC>(qf.qyz.data(), x, y, z, g);

        const Vec3 passive_div = PassiveStressDivergence<BC>(
            qf.Sigma_xx.data(),
            qf.Sigma_xy.data(),
            qf.Sigma_xz.data(),
            qf.Sigma_yy.data(),
            qf.Sigma_yz.data(),
            qf.Tau_xy.data(),
            qf.Tau_xz.data(),
            qf.Tau_yz.data(),
            x,
            y,
            z,
            g
        );

        const Vec3 u{
            ff.ux[idxp],
            ff.uy[idxp],
            ff.uz[idxp]
        };

        Vec3 force = PointwiseSetActiveStressAndComputeBodyForce(
            dQxx,
            dQxy,
            dQxz,
            dQyy,
            dQyz,
            passive_div,
            u
        );

        ff.fx[idxp] += force.x;
        ff.fy[idxp] += force.y;
        ff.fz[idxp] += force.z;
    };

    #pragma omp parallel for default(shared) num_threads(kNumOMPThreads) schedule(static)
    for (int z = 0; z < g.local_nz; ++z) {
        for (int y = 0; y < g.local_ny; ++y) {
            for (int x = 0; x < g.local_nx; ++x) {
                compute_cell(x, y, z);
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
// Loop 2 writes ff.fx/fy/fz; Loop 3 reads them. Both loops read qf.qxx/Sigma_xx at
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
