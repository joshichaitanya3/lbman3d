#include <iostream>
#include <stdexcept>
#include <string>
#include <format>
#include <params.h>
#include "device_fields.h"
#include "qtensor_types.h"
#include "physics_helpers.h"
#include "lattice_stencil.h"
#include "boundary_handler.h"
#include "model.h"

using namespace Params;

constexpr int kBlockX = 32;
constexpr int kBlockY = 4;
constexpr int kBlockZ = 4;

// Indexing (idx/InDomain) lives on LocalGrid — passed by value into every
// kernel below and forwarded to the shared CUDA_HOST_DEVICE helpers as a
// const& (see src/mpi/CLAUDE.md, "LocalGrid must be a by-value kernel
// argument"). This is a single-rank build: kHaloMPI==0, so halo_idx collapses
// to the flat i-slowest device layout the old free idx(x,y,z[,i]) produced.


// D3Q15 stencil, copied once from Lattice:: (lattice_stencil.h) into
// __constant__ memory in DeviceFields::Initialize(). CUDA C++ does not allow
// a raw array of scalar type to be used inside a kernel unless it's used
// inside a constexpr __device__ or __host__ __device__ function, so a
// runtime-indexed device array needs its own device-resident storage.
__constant__ int d_ex[Lattice::ndir];
__constant__ int d_ey[Lattice::ndir];
__constant__ int d_ez[Lattice::ndir];
__constant__ double d_w[Lattice::ndir];
__constant__ int d_opp[Lattice::ndir];
__constant__ int d_specX[Lattice::ndir];
__constant__ int d_specY[Lattice::ndir];
__constant__ int d_specZ[Lattice::ndir];

template<typename BC>
__global__ void GpuCollideAndStream(
    double* f,
    double* f_new,
    double* force_x,
    double* force_y,
    double* force_z,
    double* rho,
    double* ux,
    double* uy,
    double* uz,
    LocalGrid g
) {

    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (!g.InDomain(x, y, z)) return;

    const int gid = g.halo_idx(x, y, z);
    Vec3 force{force_x[gid], force_y[gid], force_z[gid]};

    Moments m = ComputeMoments(
        f,
        {x, y, z},
        force,
        d_ex,
        d_ey,
        d_ez,
        g
    );
    rho[gid] = m.rho;
    ux[gid]  = m.u.x;
    uy[gid]  = m.u.y;
    uz[gid]  = m.u.z;

    double uF = m.u.Dot(force);
    double u2 = m.u.Dot(m.u);

    // ── Multi-rank split detection (loop-invariant) ─────────────────────────
    // local_n < global_n when MPI cuts that axis. On single-rank or unsplit
    // axes local_n == global_n, so the seam checks below are false and the
    // existing StreamXoff path fires unchanged.
    const bool split_x = (g.local_nx != nx);
    const bool split_y = (g.local_ny != ny);
    const bool split_z = (g.local_nz != nz);

    // Compile-time periodicity flags — both faces must be Periodic.
    constexpr bool x_per = std::is_same_v<typename BC::XLo::UBC, Periodic>
                         && std::is_same_v<typename BC::XHi::UBC, Periodic>;
    constexpr bool y_per = std::is_same_v<typename BC::YLo::UBC, Periodic>
                         && std::is_same_v<typename BC::YHi::UBC, Periodic>;
    constexpr bool z_per = std::is_same_v<typename BC::ZLo::UBC, Periodic>
                         && std::is_same_v<typename BC::ZHi::UBC, Periodic>;

    for (int i = 0; i < Lattice::ndir; ++i) {
        Vec3 e_i{
            static_cast<double>(d_ex[i]),
            static_cast<double>(d_ey[i]),
            static_cast<double>(d_ez[i])
        };

        auto [feq, forcing_term] = ComputeFeqAndForcing(m, u2, uF, force, e_i, d_w[i]);
        double f_star = PointwiseBGKCollide(f[g.halo_idx(x, y, z, i)], feq, forcing_term);
        // ── Stream + Apply Boundary Conditions ───────────────────

        const int raw_dx = x + d_ex[i];
        const int raw_dy = y + d_ey[i];
        const int raw_dz = z + d_ez[i];

        // Wall crossing check in GLOBAL coords. Mirrors CPU-MPI's
        // x_crosses_wall gate (src/lbm_solver.tpp:154): a pop is at a
        // physical wall only if its destination leaves the *global* domain
        // on a non-periodic axis. Using local coords misidentifies a middle
        // seam on a split non-periodic axis (Poiseuille Y under dims={1,2,1})
        // as a wall and falsely bounces pops that should ship to the neighbour.
        const int raw_gx = g.offset_x + raw_dx;
        const int raw_gy = g.offset_y + raw_dy;
        const int raw_gz = g.offset_z + raw_dz;
        const bool x_wall = !x_per && (raw_gx < 0 || raw_gx >= nx);
        const bool y_wall = !y_per && (raw_gy < 0 || raw_gy >= ny);
        const bool z_wall = !z_per && (raw_gz < 0 || raw_gz >= nz);

        if (!x_wall && !y_wall && !z_wall) {
            // Unified write: in-domain, seam ghost (split axis), or unsplit-
            // periodic wrap. Split axes keep raw local coord → ghost slot.
            // Unsplit periodic axes wrap locally so the write lands in
            // [0, local_n). Non-periodic middle seams (raw_gy inside global
            // range but raw_dy outside local range) fall through to the
            // raw local coord, which is the correct ghost slot for the
            // halo exchange to pack.
            int gx = raw_dx, gy = raw_dy, gz = raw_dz;
            if constexpr (x_per) {
                if (!split_x && (raw_dx < 0 || raw_dx >= g.local_nx))
                    gx = (raw_dx + g.local_nx) % g.local_nx;
            }
            if constexpr (y_per) {
                if (!split_y && (raw_dy < 0 || raw_dy >= g.local_ny))
                    gy = (raw_dy + g.local_ny) % g.local_ny;
            }
            if constexpr (z_per) {
                if (!split_z && (raw_dz < 0 || raw_dz >= g.local_nz))
                    gz = (raw_dz + g.local_nz) % g.local_nz;
            }
            f_new[g.halo_idx(gx, gy, gz, i)] = f_star;
        } else {
            // Wall bounce on at least one axis. Dispatch per-axis, mirroring
            // CPU-MPI: each wall crossing applies its own HandleBoundaryPoint
            // locally (bounce-back writes to the source cell with opp[i]),
            // so composed wall + orthogonal-seam corners work without any
            // extra ghost routing — the bounced pop stays in owned and
            // streams normally next step.
            if (x_wall) {
                if (raw_gx < 0)
                    HandleBoundaryPoint<typename BC::XLo>(x, y, z, i, d_specX[i], f_star, m.rho, f_new, d_ex, d_ey, d_ez, d_w, d_opp, g);
                else
                    HandleBoundaryPoint<typename BC::XHi>(x, y, z, i, d_specX[i], f_star, m.rho, f_new, d_ex, d_ey, d_ez, d_w, d_opp, g);
            }
            if (y_wall) {
                if (raw_gy < 0)
                    HandleBoundaryPoint<typename BC::YLo>(x, y, z, i, d_specY[i], f_star, m.rho, f_new, d_ex, d_ey, d_ez, d_w, d_opp, g);
                else
                    HandleBoundaryPoint<typename BC::YHi>(x, y, z, i, d_specY[i], f_star, m.rho, f_new, d_ex, d_ey, d_ez, d_w, d_opp, g);
            }
            if (z_wall) {
                if (raw_gz < 0)
                    HandleBoundaryPoint<typename BC::ZLo>(x, y, z, i, d_specZ[i], f_star, m.rho, f_new, d_ex, d_ey, d_ez, d_w, d_opp, g);
                else
                    HandleBoundaryPoint<typename BC::ZHi>(x, y, z, i, d_specZ[i], f_star, m.rho, f_new, d_ex, d_ey, d_ez, d_w, d_opp, g);
            }
        }
    }
}


template<typename BC>
__global__ void GpuQTensorStep(
    double* qxx,
    double* qxy,
    double* qxz,
    double* qyy,
    double* qyz,
    double* qxx_new,
    double* qxy_new,
    double* qxz_new,
    double* qyy_new,
    double* qyz_new,
    double* ux,
    double* uy,
    double* uz,
    double* force_x,
    double* force_y,
    double* force_z,
    double* Sigma_xx,
    double* Sigma_xy,
    double* Sigma_xz,
    double* Sigma_yy,
    double* Sigma_yz,
    double* Tau_xy,
    double* Tau_xz,
    double* Tau_yz,
    LocalGrid g
) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (!g.InDomain(x, y, z)) return;

    const int gid = g.halo_idx(x, y, z);

    const SymTrLessTensor5 Q{
        qxx[gid],
        qxy[gid],
        qxz[gid],
        qyy[gid],
        qyz[gid]
    };

    const Vec3 u{
        ux[gid],
        uy[gid],
        uz[gid]
    };

    const QDerivs dQxx = QGradientAndLaplacian<QComp::XX, BC>(qxx, x, y, z, g);
    const QDerivs dQxy = QGradientAndLaplacian<QComp::XY, BC>(qxy, x, y, z, g);
    const QDerivs dQxz = QGradientAndLaplacian<QComp::XZ, BC>(qxz, x, y, z, g);
    const QDerivs dQyy = QGradientAndLaplacian<QComp::YY, BC>(qyy, x, y, z, g);
    const QDerivs dQyz = QGradientAndLaplacian<QComp::YZ, BC>(qyz, x, y, z, g);

    // Velocity gradient tensor: vA_B = ∂(u_A)/∂B
    const GradTensor nabla_u = VelocityGradientTensor<BC>(ux, uy, uz, x, y, z, g);
    
    const QStencil qs{
            Q, u, dQxx, dQxy, dQxz, dQyy, dQyz, nabla_u
        };
        
    SymTrLessTensor5 q_new, sigma;
    AntiSymTensor3 tau;
    Vec3 ericksen_force;

    PointwiseStepAndSetupBodyForce(
        qs,
        q_new,
        sigma,
        tau,
        ericksen_force
    );


    qxx_new[gid] = q_new.xx;
    qxy_new[gid] = q_new.xy;
    qxz_new[gid] = q_new.xz;
    qyy_new[gid] = q_new.yy;
    qyz_new[gid] = q_new.yz;

    Sigma_xx[gid] = sigma.xx;
    Sigma_xy[gid] = sigma.xy;
    Sigma_xz[gid] = sigma.xz;
    Sigma_yy[gid] = sigma.yy;
    Sigma_yz[gid] = sigma.yz;

    // Antisymmetric (torque-carrying) part, kept separate so the divergence in
    // GpuComputeBodyForce can apply A_beta,alpha = -A_alpha,beta
    Tau_xy[gid] = tau.xy;
    Tau_xz[gid] = tau.xz;
    Tau_yz[gid] = tau.yz;

    // Seed the body force with the Ericksen (distortion) force; GpuComputeBodyForce
    // adds div(Sigma + Tau), the active stress and friction.
    force_x[gid] = ericksen_force.x;
    force_y[gid] = ericksen_force.y;
    force_z[gid] = ericksen_force.z;
}


template<typename BC>
__global__ void GpuComputeBodyForce(
    double* qxx,
    double* qxy,
    double* qxz,
    double* qyy,
    double* qyz,
    double* ux,
    double* uy,
    double* uz,
    double* force_x,
    double* force_y,
    double* force_z,
    double* Sigma_xx,
    double* Sigma_xy,
    double* Sigma_xz,
    double* Sigma_yy,
    double* Sigma_yz,
    double* Tau_xy,
    double* Tau_xz,
    double* Tau_yz,
    LocalGrid g
) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (!g.InDomain(x, y, z)) return;

    const int gid = g.halo_idx(x, y, z);

    const QDerivs dQxx = QGradientAndLaplacian<QComp::XX, BC>(qxx, x, y, z, g);
    const QDerivs dQxy = QGradientAndLaplacian<QComp::XY, BC>(qxy, x, y, z, g);
    const QDerivs dQxz = QGradientAndLaplacian<QComp::XZ, BC>(qxz, x, y, z, g);
    const QDerivs dQyy = QGradientAndLaplacian<QComp::YY, BC>(qyy, x, y, z, g);
    const QDerivs dQyz = QGradientAndLaplacian<QComp::YZ, BC>(qyz, x, y, z, g);

    const Vec3 passive_div = PassiveStressDivergence<BC>(
        Sigma_xx,
        Sigma_xy,
        Sigma_xz,
        Sigma_yy,
        Sigma_yz,
        Tau_xy,
        Tau_xz,
        Tau_yz,
        x,
        y,
        z,
        g
    );

    const Vec3 u{
        ux[gid],
        uy[gid],
        uz[gid]
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

    force_x[gid] += force.x;
    force_y[gid] += force.y;
    force_z[gid] += force.z;
}
