#ifndef LBM_AN_MODEL_H
#define LBM_AN_MODEL_H

#include "params.h"
#include "qtensor_types.h"
#include "physics_helpers.h"
#include "boundary_handler.h"

#ifndef CUDA_HOST_DEVICE
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif
#endif

using namespace Params;

// In model.h — CUDA_HOST_DEVICE throughout
struct QStencil {
    // Central point values
    SymTrLessTensor5 Q;
    Vec3 u;
    // Pre-fetched gradients/Laplacians for each component
    QDerivs dQxx, dQxy, dQxz, dQyy, dQyz;
    GradTensor gradu;
};

inline CUDA_HOST_DEVICE void PointwiseStepAndSetupBodyForce(
    const QStencil& qs,
    SymTrLessTensor5& q_new,
    SymTrLessTensor5& passive_stress,
    Vec3& advective_backflow
) {

    // Fields

    const double Qxx = qs.Q.xx;
    const double Qxy = qs.Q.xy;
    const double Qxz = qs.Q.xz;
    const double Qyy = qs.Q.yy;
    const double Qyz = qs.Q.yz;

    const double ux = qs.u.x;
    const double uy = qs.u.y;
    const double uz = qs.u.z;

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

    const auto [Qxxx, Qxxy, Qxxz, lap_Qxx] = qs.dQxx;
    const auto [Qxyx, Qxyy, Qxyz, lap_Qxy] = qs.dQxy;
    const auto [Qxzx, Qxzy, Qxzz, lap_Qxz] = qs.dQxz;
    const auto [Qyyx, Qyyy, Qyyz, lap_Qyy] = qs.dQyy;
    const auto [Qyzx, Qyzy, Qyzz, lap_Qyz] = qs.dQyz;

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
    const double ld = A + C * TrQ2;
    const double Hxx = L * lap_Qxx - ld * Qxx - B * Q2_xx;
    const double Hxy = L * lap_Qxy - ld * Qxy - B * Q2_xy;
    const double Hxz = L * lap_Qxz - ld * Qxz - B * Q2_xz;
    const double Hyy = L * lap_Qyy - ld * Qyy - B * Q2_yy;
    const double Hyz = L * lap_Qyz - ld * Qyz - B * Q2_yz;
    
    // Add the advective counter part of the back-flow to the body force, H:\nabla Q
    // since this does not come from the divergence of the stress tensor.
    // The backflow from the divergence will be added to this by SetActiveStressAndComputeBodyForce

    advective_backflow.x = -2.0 * (Hxx*Qxxx + Hxy*Qxyx + Hxz*Qxzx + Hyy*Qyyx + Hyz*Qyzx) + Hxx*Qyyx + Hyy*Qxxx;
    advective_backflow.y = -2.0 * (Hxx*Qxxy + Hxy*Qxyy + Hxz*Qxzy + Hyy*Qyyy + Hyz*Qyzy) + Hxx*Qyyy + Hyy*Qxxy;
    advective_backflow.z = -2.0 * (Hxx*Qxxz + Hxy*Qxyz + Hxz*Qxzz + Hyy*Qyyz + Hyz*Qyzz) + Hxx*Qyyz + Hyy*Qxxz;

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
    
    const double Taux_x = 0.0; // Diagonal component of antisymmetric tensor
    const double Taux_y = (Hxy*Qxx + Hyy*Qxy + Hyz*Qxz) - (Qxy*Hxx + Qyy*Hxy + Qyz*Hxz);
    const double Taux_z = (Hxz*Qxx + Hyz*Qxy + (-Hxx - Hyy)*Qxz) - (Qxz*Hxx + Qyz*Hxy + (-Qxx - Qyy)*Hxz);
    const double Tauy_y = 0.0; // Diagonal component of antisymmetric tensor
    const double Tauy_z = (Hxz*Qxy + Hyz*Qyy + (-Hxx - Hyy)*Qyz) - (Qxz*Hxy + Qyz*Hyy + (-Qxx - Qyy)*Hyz);

    // Update nematic stress (passive + active)
    passive_stress.xx = -ktwo_thirds * LAMBDA * Hxx - LAMBDA * QHxx + Taux_x;
    passive_stress.xy = -ktwo_thirds * LAMBDA * Hxy - LAMBDA * QHxy + Taux_y;
    passive_stress.xz = -ktwo_thirds * LAMBDA * Hxz - LAMBDA * QHxz + Taux_z;
    passive_stress.yy = -ktwo_thirds * LAMBDA * Hyy - LAMBDA * QHyy + Tauy_y;
    passive_stress.yz = -ktwo_thirds * LAMBDA * Hyz - LAMBDA * QHyz + Tauy_z;

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
    double fz = -ALPHA * (dQxz.dx + dQyz.dy - dQxx.dz - dQyy.dz) + passive_div.z - MU * u.z; // Since Pzz = -(Pxx + Pyy)

    return {fx, fy, fz};
};

#endif // LBM_AN_MODEL_H
