#ifndef LBM_AN_MODEL_H
#define LBM_AN_MODEL_H

#include <params.h>
// For kQAdvection. This is the one place a physics header reaches up to
// sim_config.h: the advection scheme is selected there, next to the boundary
// conditions, because both are discretisation choices. The BC config still
// arrives by template parameter, not by inclusion.
#include <sim_config.h>
#include "qtensor_types.h"
#include "physics_helpers.h"
#include "boundary_handler.h"

#include <cmath>

#ifndef CUDA_HOST_DEVICE
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif
#endif

using namespace Params;

// Equilibrium scalar order parameter of the Landau-de Gennes bulk free energy:
// the S at which the bulk part of the molecular field H vanishes on the uniaxial
// family Q = S (nn - I/3).
//
// Substituting Q = diag(2S/3, -S/3, -S/3) into the bulk terms of H below,
// -(A + C TrQ^2) Q - B [Q^2]_traceless, gives TrQ^2 = (2/3)S^2 and
// [Q^2]_traceless,xx = (2/9)S^2, so
//
//     H_xx = -(4/9) C S^3 - (2/9) B S^2 = 0   ->   2 C S^2 + B S + 3 A = 0
//
// whose ordered (+) root is the value below. With A = 0 it reduces to -B/(2C).
//
// Host-only: this is initialisation/analysis, never called from a kernel.
inline double EquilibriumScalarOrder() {
    return (-B + std::sqrt(B * B - 24.0 * A * C)) / (4.0 * C);
}

// One axis's contribution to u.grad(Q), i.e. u_a * dQ/dx_a.
//
//   Advection::Centred:  u * dQ
//   Advection::Upwind:   u * dQ - (|u|/2) * d2Q
//
// The upwind form follows from Q(a) - Q(a-1) = dQ - d2Q/2 for u > 0 and
// Q(a+1) - Q(a) = dQ + d2Q/2 for u < 0; combining the two cases gives the
// single branch-free expression above.
//
// Written this way the correction is manifestly a diffusion term of
// coefficient |u|/2, i.e. upwinding adds a numerical Q diffusivity of |u|*DX/2.
// That is the price of the scheme and it is not small — see the envelope
// measured next to kQAdvection in sim_config.h.
//
// Prefer AdvectiveDerivative below at call sites: this takes three
// interchangeable doubles, so a transposed (velocity, derivative) pair would
// compile silently.
template<Advection Scheme>
inline CUDA_HOST_DEVICE double AdvectiveAxisTerm(double u, double dQ, double d2Q) {
    if constexpr (Scheme == Advection::Upwind) {
        const double abs_u = u < 0.0 ? -u : u;
        return u * dQ - 0.5 * abs_u * d2Q;
    } else {
        return u * dQ;
    }
}

// u.grad(Q) for one Q component, summed over all three axes.
//
// Taking the whole velocity and the whole QDerivs makes the axis pairing
// structural: there is no way for a caller to hand the x-velocity an
// x-derivative from the wrong component, which the three-double form above
// permits. The Beris-Edwards right-hand side carries -u.grad(Q), so call sites
// negate.
template<Advection Scheme = kQAdvection>
inline CUDA_HOST_DEVICE double AdvectiveDerivative(const Vec3& u, const QDerivs& d) {
    return AdvectiveAxisTerm<Scheme>(u.x, d.dx, d.d2x)
         + AdvectiveAxisTerm<Scheme>(u.y, d.dy, d.d2y)
         + AdvectiveAxisTerm<Scheme>(u.z, d.dz, d.d2z);
}

// In model.h — CUDA_HOST_DEVICE throughout
struct QStencil {
    // Central point values
    SymTrLessTensor5 Q;
    Vec3 u;
    // Pre-fetched gradients/Laplacians for each component
    QDerivs dQxx, dQxy, dQxz, dQyy, dQyz;
    GradTensor gradu;
};

// The nematic stress Pi = Sigma + Tau is returned in two pieces:
//   sigma — symmetric traceless, Sigma = -(2/3) lambda H - lambda (QH + HQ)
//   tau   — antisymmetric, Tau = QH - HQ, upper triangle only (no diagonal)
//
// Separate because Tau_beta,alpha = -Tau_alpha,beta, which a symmetric 5-slot
// container cannot represent. The divergence consumes both under the row
// convention f_alpha = d_beta Pi_alpha,beta (see PassiveStressDivergence in
// boundary_handler.h).
inline CUDA_HOST_DEVICE void PointwiseStepAndSetupBodyForce(
    const QStencil& qs,
    SymTrLessTensor5& q_new,
    SymTrLessTensor5& sigma,
    AntiSymTensor3& tau,
    Vec3& ericksen_force
) {

    // Fields

    const double Qxx = qs.Q.xx;
    const double Qxy = qs.Q.xy;
    const double Qxz = qs.Q.xz;
    const double Qyy = qs.Q.yy;
    const double Qyz = qs.Q.yz;


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
    // component at SpecularReflection walls

    // Velocity gradient tensor: vA_B = ∂(u_A)/∂B
    auto [ux_x, ux_y, ux_z, uy_x, uy_y, uy_z, uz_x, uz_y] = qs.gradu;

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

    const double Exx = ux_x;
    const double Exy = 0.5 * (ux_y + uy_x);
    const double Exz = 0.5 * (ux_z + uz_x);
    const double Eyy = uy_y;
    const double Eyz = 0.5 * (uy_z + uz_y);

    /*
    Omegaxy = 1/2(ux_y - uy_x)
    */
    const double Wxy = 0.5 * (ux_y - uy_x);
    const double Wxz = 0.5 * (ux_z - uz_x);
    const double Wyz = 0.5 * (uy_z - uz_y);


    // Q-tensor gradient + Laplacian (central/7-point stencil, wall-aware
    // ghost values — see boundary_handler.h's QGradientAndLaplacian; NOT
    // the QXoff/QYoff/QZoff Neumann-only clamp, which is wrong for
    // Anchoring walls)

    // Named by member rather than destructured: QDerivs also carries the
    // per-axis second differences (consumed by AdvectiveDerivative below), and a
    // structured binding would have to list every member.
    const double Qxxx = qs.dQxx.dx, Qxxy = qs.dQxx.dy, Qxxz = qs.dQxx.dz;
    const double Qxyx = qs.dQxy.dx, Qxyy = qs.dQxy.dy, Qxyz = qs.dQxy.dz;
    const double Qxzx = qs.dQxz.dx, Qxzy = qs.dQxz.dy, Qxzz = qs.dQxz.dz;
    const double Qyyx = qs.dQyy.dx, Qyyy = qs.dQyy.dy, Qyyz = qs.dQyy.dz;
    const double Qyzx = qs.dQyz.dx, Qyzy = qs.dQyz.dy, Qyzz = qs.dQyz.dz;

    const double lap_Qxx = qs.dQxx.lap, lap_Qxy = qs.dQxy.lap, lap_Qxz = qs.dQxz.lap;
    const double lap_Qyy = qs.dQyy.lap, lap_Qyz = qs.dQyz.lap;

    // Advection: -u · ∇Q. The centred/upwind choice lives entirely inside
    // AdvectiveDerivative (see kQAdvection in sim_config.h).
    const double adv_xx = -AdvectiveDerivative(qs.u, qs.dQxx);
    const double adv_xy = -AdvectiveDerivative(qs.u, qs.dQxy);
    const double adv_xz = -AdvectiveDerivative(qs.u, qs.dQxz);
    const double adv_yy = -AdvectiveDerivative(qs.u, qs.dQyy);
    const double adv_yz = -AdvectiveDerivative(qs.u, qs.dQyz);
    
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
    const double ld = A + C * TrQ2;
    const double Hxx = L * lap_Qxx - ld * Qxx - B * Q2_xx;
    const double Hxy = L * lap_Qxy - ld * Qxy - B * Q2_xy;
    const double Hxz = L * lap_Qxz - ld * Qxz - B * Q2_xz;
    const double Hyy = L * lap_Qyy - ld * Qyy - B * Q2_yy;
    const double Hyz = L * lap_Qyz - ld * Qyz - B * Q2_yz;
    
    // The Ericksen (distortion) force, f_alpha = -H_ij d_alpha Q_ij. This is the
    // half of the backflow coupling that is NOT a stress divergence; the other
    // half, div(Sigma + Tau), is added by SetActiveStressAndComputeBodyForce.
    // Together they reproduce d_beta Pi_alpha,beta.
    //
    // Despite carrying \nabla Q it is not an advective term — there is no
    // velocity in it, hence no upstream direction, so kQAdvection does NOT apply
    // here and these gradients stay centred. Biasing them would also break the
    // cancellation against div(Sigma + Tau) that makes the total force integrate
    // to zero over a periodic domain, injecting spurious momentum.
    //
    // H:\nabla_k Q sums over all nine index pairs. With H_zz = -(H_xx + H_yy)
    // and \partial_k Q_zz = -(\partial_k Q_xx + \partial_k Q_yy), the zz pair
    // expands to H_xx dQ_xx + H_xx dQ_yy + H_yy dQ_xx + H_yy dQ_yy, so
    //   H:\nabla_k Q = 2(H_xx dQ_xx + H_yy dQ_yy + H_xy dQ_xy + H_xz dQ_xz + H_yz dQ_yz)
    //                  + H_xx dQ_yy + H_yy dQ_xx
    // Same contraction pattern as Q:H below and Q:\nabla^2 Q in analysis_fields.cc's
    // TotalNematicFreeEnergy. Guarded by tests/unit/test_backflow_contraction.cc.

    ericksen_force.x = -2.0 * (Hxx*Qxxx + Hxy*Qxyx + Hxz*Qxzx + Hyy*Qyyx + Hyz*Qyzx) - Hxx*Qyyx - Hyy*Qxxx;
    ericksen_force.y = -2.0 * (Hxx*Qxxy + Hxy*Qxyy + Hxz*Qxzy + Hyy*Qyyy + Hyz*Qyzy) - Hxx*Qyyy - Hyy*Qxxy;
    ericksen_force.z = -2.0 * (Hxx*Qxxz + Hxy*Qxyz + Hxz*Qxzz + Hyy*Qyyz + Hyz*Qyzz) - Hxx*Qyyz - Hyy*Qxxz;

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
    
    // Antisymmetric part Tau = QH - HQ. Only the upper triangle is independent:
    // the diagonal vanishes identically and Tau_yx = -Tau_xy, etc. These are
    // returned separately from the symmetric part rather than added into it,
    // since the divergence needs the lower-triangle signs.
    const double Tau_xy = (Hxy*Qxx + Hyy*Qxy + Hyz*Qxz) - (Qxy*Hxx + Qyy*Hxy + Qyz*Hxz);
    const double Tau_xz = (Hxz*Qxx + Hyz*Qxy + (-Hxx - Hyy)*Qxz) - (Qxz*Hxx + Qyz*Hxy + (-Qxx - Qyy)*Hxz);
    const double Tau_yz = (Hxz*Qxy + Hyz*Qyy + (-Hxx - Hyy)*Qyz) - (Qxz*Hxy + Qyz*Hyy + (-Qxx - Qyy)*Hyz);

    // Symmetric-traceless part of the nematic stress
    sigma.xx = -ktwo_thirds * LAMBDA * Hxx - LAMBDA * QHxx;
    sigma.xy = -ktwo_thirds * LAMBDA * Hxy - LAMBDA * QHxy;
    sigma.xz = -ktwo_thirds * LAMBDA * Hxz - LAMBDA * QHxz;
    sigma.yy = -ktwo_thirds * LAMBDA * Hyy - LAMBDA * QHyy;
    sigma.yz = -ktwo_thirds * LAMBDA * Hyz - LAMBDA * QHyz;

    // Antisymmetric part, kept separate
    tau.xy = Tau_xy;
    tau.xz = Tau_xz;
    tau.yz = Tau_yz;

    // Now, we perform the timestep
    q_new.xx = Qxx + DT*(adv_xx + cor_xx + LAMBDA * (ktwo_thirds * Exx + aln2_xx) + GAMMA * Hxx);
    q_new.xy = Qxy + DT*(adv_xy + cor_xy + LAMBDA * (ktwo_thirds * Exy + aln2_xy) + GAMMA * Hxy);
    q_new.xz = Qxz + DT*(adv_xz + cor_xz + LAMBDA * (ktwo_thirds * Exz + aln2_xz) + GAMMA * Hxz);
    q_new.yy = Qyy + DT*(adv_yy + cor_yy + LAMBDA * (ktwo_thirds * Eyy + aln2_yy) + GAMMA * Hyy);
    q_new.yz = Qyz + DT*(adv_yz + cor_yz + LAMBDA * (ktwo_thirds * Eyz + aln2_yz) + GAMMA * Hyz);

};

inline CUDA_HOST_DEVICE Vec3 PointwiseSetActiveStressAndComputeBodyForce(
    const QDerivs& dQxx,    
    const QDerivs& dQxy,    
    const QDerivs& dQxz,    
    const QDerivs& dQyy,    
    const QDerivs& dQyz,
    const Vec3& passive_div,
    const Vec3& u
) {
    
    // First, add the active force. Q's own gradient

    double fx = -ALPHA * (dQxx.dx + dQxy.dy + dQxz.dz) + passive_div.x - MU * u.x;;
    double fy = -ALPHA * (dQxy.dx + dQyy.dy + dQyz.dz) + passive_div.y - MU * u.y;
    double fz = -ALPHA * (dQxz.dx + dQyz.dy - dQxx.dz - dQyy.dz) + passive_div.z - MU * u.z; // Since Pzz = -(Sigma_xx + Sigma_yy)

    return {fx, fy, fz};
};

#endif // LBM_AN_MODEL_H
