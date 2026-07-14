#include <iostream>
#include <stdexcept>
#include <string>
#include <format>
#include "params.h"
#include "device_fields.h"
#include "qtensor_types.h"
#include "physics_helpers.h"
#include "lattice_stencil.h"
#include "boundary_handler.h"
#include "model.h"

using namespace Params;

constexpr int kHalo = 1;
constexpr int kBlockX = 32;
constexpr int kBlockY = 4;
constexpr int kBlockZ = 4;

static dim3 block_{kBlockX, kBlockY, kBlockZ};
static dim3 grid_{(nx + kBlockX - 1) / kBlockX, (ny + kBlockY - 1) / kBlockY, (nz + kBlockZ - 1) / kBlockZ};

// idx(x,y,z[,i]) now comes from physics_helpers.h, shared with host code.
__device__ inline int wrap(int i, int n) { return (i + n) % n; }


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
    double* uz
) {

    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx || y >= ny || z >= nz) return;  // bounds guard

    const int gid = idx(x, y, z);
    Vec3 force{force_x[gid], force_y[gid], force_z[gid]};

    Moments m = ComputeMoments(
        f,
        {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)},
        force,
        d_ex,
        d_ey,
        d_ez
    );
    rho[idx(x, y, z)] = m.rho;
    ux[idx(x, y, z)]  = m.u.x;
    uy[idx(x, y, z)]  = m.u.y;
    uz[idx(x, y, z)]  = m.u.z;

    double uF = m.u.Dot(force);
    double u2 = m.u.Dot(m.u);

    for (int i = 0; i < Lattice::ndir; ++i) {
        Vec3 e_i{
            static_cast<double>(d_ex[i]),
            static_cast<double>(d_ey[i]),
            static_cast<double>(d_ez[i])
        };

        auto [feq, forcing_term] = ComputeFeqAndForcing(m, u2, uF, force, e_i, d_w[i]);
        double f_star = PointwiseBGKCollide(f[idx(x, y, z, i)], feq, forcing_term);
        // ── Stream + Apply Boundary Conditions ───────────────────
        const int dx = StreamXoff<BC>(x, d_ex[i]);
        const int dy = StreamYoff<BC>(y, d_ey[i]);
        const int dz = StreamZoff<BC>(z, d_ez[i]);   
        if (InDomain(dx, dy, dz)) {
            f_new[idx(dx, dy, dz, i)] = f_star;
        } else {
            if (dx < 0) {
                HandleBoundaryPoint<typename BC::XLo>(x, y, z, i, d_specX[i], f_star, m.rho, f_new, d_ex, d_ey, d_ez, d_w, d_opp);
            }
            else if (dx >= nx) {
                HandleBoundaryPoint<typename BC::XHi>(x, y, z, i, d_specX[i], f_star, m.rho, f_new, d_ex, d_ey, d_ez, d_w, d_opp);
            }
            if (dy < 0) {
                HandleBoundaryPoint<typename BC::YLo>(x, y, z, i, d_specY[i], f_star, m.rho, f_new, d_ex, d_ey, d_ez, d_w, d_opp);
            }
            else if (dy >= ny) {
                HandleBoundaryPoint<typename BC::YHi>(x, y, z, i, d_specY[i], f_star, m.rho, f_new, d_ex, d_ey, d_ez, d_w, d_opp);
            }
            if (dz < 0) {
                HandleBoundaryPoint<typename BC::ZLo>(x, y, z, i, d_specZ[i], f_star, m.rho, f_new, d_ex, d_ey, d_ez, d_w, d_opp);
            }
            else if (dz >= nz) {
                HandleBoundaryPoint<typename BC::ZHi>(x, y, z, i, d_specZ[i], f_star, m.rho, f_new, d_ex, d_ey, d_ez, d_w, d_opp);
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
    double* Pxx,
    double* Pxy,
    double* Pxz,
    double* Pyy,
    double* Pyz
    
) {
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

    unsigned int z = blockIdx.z * blockDim.z + tz;
    unsigned int y = blockIdx.y * blockDim.y + ty;
    unsigned int x = blockIdx.x * blockDim.x + tx;
    const bool in_domain = (x < nx) && (y < ny) && (z < nz);

    if (!in_domain) return;

    const int gid = idx(x, y, z);
    
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
    
    const QDerivs dQxx = QGradientAndLaplacian<QComp::XX, BC>(qxx, x, y, z);
    const QDerivs dQxy = QGradientAndLaplacian<QComp::XY, BC>(qxy, x, y, z);
    const QDerivs dQxz = QGradientAndLaplacian<QComp::XZ, BC>(qxz, x, y, z);
    const QDerivs dQyy = QGradientAndLaplacian<QComp::YY, BC>(qyy, x, y, z);
    const QDerivs dQyz = QGradientAndLaplacian<QComp::YZ, BC>(qyz, x, y, z);

    // Velocity gradient tensor: vA_B = ∂(u_A)/∂B
    const GradTensor nabla_u = VelocityGradientTensor<BC>(ux, uy, uz, x, y, z);
    
    const QStencil qs{
            Q, u, dQxx, dQxy, dQxz, dQyy, dQyz, nabla_u
        };
        
    SymTrLessTensor5 q_new, passive_stress;
    Vec3 advective_backflow;
    
    PointwiseStepAndSetupBodyForce(
        qs,
        q_new,
        passive_stress,
        advective_backflow
    );
    

    qxx_new[gid] = q_new.xx;
    qxy_new[gid] = q_new.xy;
    qxz_new[gid] = q_new.xz;
    qyy_new[gid] = q_new.yy;
    qyz_new[gid] = q_new.yz;

    Pxx[gid] = passive_stress.xx;
    Pxy[gid] = passive_stress.xy;
    Pxz[gid] = passive_stress.xz;
    Pyy[gid] = passive_stress.yy;
    Pyz[gid] = passive_stress.yz;

    force_x[gid] = advective_backflow.x;
    force_y[gid] = advective_backflow.y;
    force_z[gid] = advective_backflow.z;
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
    double* Pxx,
    double* Pxy,
    double* Pxz,
    double* Pyy,
    double* Pyz
) {
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

    unsigned int z = blockIdx.z * blockDim.z + tz;
    unsigned int y = blockIdx.y * blockDim.y + ty;
    unsigned int x = blockIdx.x * blockDim.x + tx;
    const bool in_domain = (x < nx) && (y < ny) && (z < nz);

    if (!in_domain) return;

    const int gid = idx(x, y, z);

    const QDerivs dQxx = QGradientAndLaplacian<QComp::XX, BC>(qxx, x, y, z);
    const QDerivs dQxy = QGradientAndLaplacian<QComp::XY, BC>(qxy, x, y, z);
    const QDerivs dQxz = QGradientAndLaplacian<QComp::XZ, BC>(qxz, x, y, z);
    const QDerivs dQyy = QGradientAndLaplacian<QComp::YY, BC>(qyy, x, y, z);
    const QDerivs dQyz = QGradientAndLaplacian<QComp::YZ, BC>(qyz, x, y, z);

    const Vec3 passive_div = PassiveStressDivergence<BC>(
        Pxx,
        Pxy,
        Pxz,
        Pyy,
        Pyz,
        x,
        y,
        z
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
