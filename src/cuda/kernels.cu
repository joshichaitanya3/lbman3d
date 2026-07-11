#include <iostream>
#include <stdexcept>
#include <string>
#include <format>
#include "params.h"
#include "device_fields.h"
#include "physics_helpers.h"
#include "lattice_stencil.h"

using namespace Params;

constexpr int kHalo = 1;
constexpr int kBlockX = 32;
constexpr int kBlockY = 4;
constexpr int kBlockZ = 4;

static constexpr size_t kQstepSmem =
        8 * (kBlockZ+2*kHalo) * (kBlockY+2*kHalo) * (kBlockX+2*kHalo) * sizeof(double);
dim3 block_{kBlockX, kBlockY, kBlockZ};
dim3 grid_{(nx + kBlockX - 1) / kBlockX, (ny + kBlockY - 1) / kBlockY, (nz + kBlockZ - 1) / kBlockZ};

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

__global__ void GpuInitialize(
    double* f, 
    double* rho,
    double* ux,
    double* uy,
    double* uz) {
    
    double rhop, uxp, uyp, uzp;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx || y >= ny || z >= nz) return;  // bounds guard

    rhop = rho[idx(x, y, z)]; // rho at current point
    uxp =   ux[idx(x, y, z)]; // ux at current point
    uyp =   uy[idx(x, y, z)]; // uy at current point
    uzp =   uz[idx(x, y, z)]; // uz at current point
    Vec3 up{uxp, uyp, uzp};
    double u2 = up.Dot(up);	//Velocity squared
    for (int i = 0; i < Lattice::ndir; ++i) {
        Vec3 e{
            static_cast<double>(d_ex[i]),
            static_cast<double>(d_ey[i]),
            static_cast<double>(d_ez[i])
        };
        // f[idx(x, y, z, i)] = Feq({rhop, up}, u2, i);
        f[idx(x, y, z, i)] = Feq({rhop, up}, e, u2, d_w[i]);
    }
}

// ---- Future work: compile-time-switchable boundary conditions ----------------
// To support multiple BC types without runtime branching, template this kernel
// on top/bottom wall BC types using an enum and if constexpr:
//
//   enum class WallBC { BounceBack, SpecularReflect, MovingWall };
//
//   template<WallBC BC>
//   __device__ void ApplyWallBC(double* f_new, int x, int y, int i,
//                                double f_star, double rho) {
//       if constexpr (BC == WallBC::BounceBack) {
//           f_new[idx(x, y, d_opp[i])] = f_star;
//       } else if constexpr (BC == WallBC::SpecularReflect) {
//           f_new[idx(x, y, d_spec[i])] = f_star;
//       } else if constexpr (BC == WallBC::MovingWall) {
//           // Ladd correction: f_opp = f*_i - 6 * w_i * rho * (e_i . u_wall)
//           // For a wall moving at kWallVelocity in x:
//           double correction = 6.0 * d_w[i] * rho * d_ex[i] * kWallVelocity;
//           f_new[idx(x, y, d_opp[i])] = f_star - correction;
//       }
//   }
//
//   template<WallBC TopBC, WallBC BottomBC>
//   __global__ void GpuCollideAndStream(...) {
//       ...
//       } else if (yd >= ny) {
//           ApplyWallBC<TopBC>(f_new, x, y, i, f_star, rhop);
//       } else {  // yd < 0
//           ApplyWallBC<BottomBC>(f_new, x, y, i, f_star, rhop);
//       }
//   }
//
// Instantiate whichever combo is needed, e.g.:
//   GpuCollideAndStream<WallBC::BounceBack,  WallBC::BounceBack><<<...>>>();  // Poiseuille
//   GpuCollideAndStream<WallBC::MovingWall,  WallBC::BounceBack><<<...>>>();  // Couette
//   GpuCollideAndStream<WallBC::SpecularReflect, WallBC::BounceBack><<<...>>>();  // free-slip top
// ------------------------------------------------------------------------------
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
        Vec3 e{
            static_cast<double>(d_ex[i]),
            static_cast<double>(d_ey[i]),
            static_cast<double>(d_ez[i])
        };

        auto [feq, forcing_term] = ComputeFeqAndForcing(m, u2, uF, force, e, d_w[i]);
        double f_star = omega * f[idx(x, y, z, i)] + omega_prime * feq + DT * forcing_term;
        int xd = wrap(static_cast<int>(x) + d_ex[i], nx);
        int yd = wrap(static_cast<int>(y) + d_ey[i], ny);
        int zd = wrap(static_cast<int>(z) + d_ez[i], nz);
        if (InDomain(xd, yd, zd)) {
            f_new[idx(xd, yd, zd, i)] = f_star;
        } else {
            f_new[idx(x, y, z, d_opp[i])] = f_star;
        }
    }
}

// Fills face halos in a 3D shared tile. Fully periodic: the GPU Q-tensor step
// doesn't have per-axis BC-aware neighbor offsets yet (unlike the host's
// QXoff/QYoff/QZoff), so wrap() is applied explicitly here rather than relying
// on idx()'s own bounds — idx() now asserts its (x,y,z) is already in-domain.
__device__ void set_halo(
    double* d_arr,
    double s_arr[][kBlockY+2*kHalo][kBlockX+2*kHalo],
    int tx, int ty, int tz,
    int sx, int sy, int sz,
    int gx, int gy, int gz)
{
    if (tx < kHalo)
        s_arr[sz][sy][sx - kHalo] = d_arr[idx(wrap(gx - kHalo, nx), gy, gz)];
    if (tx >= (kBlockX - kHalo))
        s_arr[sz][sy][sx + kHalo] = d_arr[idx(wrap(gx + kHalo, nx), gy, gz)];
    if (ty < kHalo)
        s_arr[sz][sy - kHalo][sx] = d_arr[idx(gx, wrap(gy - kHalo, ny), gz)];
    if (ty >= (kBlockY - kHalo))
        s_arr[sz][sy + kHalo][sx] = d_arr[idx(gx, wrap(gy + kHalo, ny), gz)];
    if (tz < kHalo)
        s_arr[sz - kHalo][sy][sx] = d_arr[idx(gx, gy, wrap(gz - kHalo, nz))];
    if (tz >= (kBlockZ - kHalo))
        s_arr[sz + kHalo][sy][sx] = d_arr[idx(gx, gy, wrap(gz + kHalo, nz))];
}


// ------------------------------------------------------------------------------

struct GradTensor {
    double ux_x, ux_y, ux_z;
    double uy_x, uy_y, uy_z;
    double uz_x, uz_y;  // uz_z = -(ux_x + uy_y) by incompressibility
};

__device__ Vec3 Gradient(double s_arr[][kBlockY+2*kHalo][kBlockX+2*kHalo], int sx, int sy, int sz) {
    return {
        0.5*(s_arr[sz][sy][sx+1] - s_arr[sz][sy][sx-1]),
        0.5*(s_arr[sz][sy+1][sx] - s_arr[sz][sy-1][sx]),
        0.5*(s_arr[sz+1][sy][sx] - s_arr[sz-1][sy][sx])
    };
}

__device__ GradTensor VelGradient(
    double ux_arr[][kBlockY+2*kHalo][kBlockX+2*kHalo],
    double uy_arr[][kBlockY+2*kHalo][kBlockX+2*kHalo],
    double uz_arr[][kBlockY+2*kHalo][kBlockX+2*kHalo],
    int sx, int sy, int sz)
{
    return {
        0.5*(ux_arr[sz][sy][sx+1] - ux_arr[sz][sy][sx-1]),
        0.5*(ux_arr[sz][sy+1][sx] - ux_arr[sz][sy-1][sx]),
        0.5*(ux_arr[sz+1][sy][sx] - ux_arr[sz-1][sy][sx]),
        0.5*(uy_arr[sz][sy][sx+1] - uy_arr[sz][sy][sx-1]),
        0.5*(uy_arr[sz][sy+1][sx] - uy_arr[sz][sy-1][sx]),
        0.5*(uy_arr[sz+1][sy][sx] - uy_arr[sz-1][sy][sx]),
        0.5*(uz_arr[sz][sy][sx+1] - uz_arr[sz][sy][sx-1]),
        0.5*(uz_arr[sz][sy+1][sx] - uz_arr[sz][sy-1][sx])
    };
}

__device__ double Laplacian(double s_arr[][kBlockY+2*kHalo][kBlockX+2*kHalo], int sx, int sy, int sz) {
    return s_arr[sz][sy][sx+1] + s_arr[sz][sy][sx-1]
         + s_arr[sz][sy+1][sx] + s_arr[sz][sy-1][sx]
         + s_arr[sz+1][sy][sx] + s_arr[sz-1][sy][sx]
         - 6.0 * s_arr[sz][sy][sx];
}

// 3D Q-tensor step: evolves all 5 independent components qxx, qxy, qxz, qyy, qyz
// (qzz = -qxx - qyy by tracelessness)
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
    double* force_z
) {
    extern __shared__ double smem[];
    constexpr int kTile = (kBlockZ+2*kHalo) * (kBlockY+2*kHalo) * (kBlockX+2*kHalo);
    using Tile = double(*)[kBlockY+2*kHalo][kBlockX+2*kHalo];
    auto s_qxx = Tile(smem + 0*kTile);
    auto s_qxy = Tile(smem + 1*kTile);
    auto s_qxz = Tile(smem + 2*kTile);
    auto s_qyy = Tile(smem + 3*kTile);
    auto s_qyz = Tile(smem + 4*kTile);
    auto s_ux  = Tile(smem + 5*kTile);
    auto s_uy  = Tile(smem + 6*kTile);
    auto s_uz  = Tile(smem + 7*kTile);

    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

    unsigned int z = blockIdx.z * blockDim.z + tz;
    unsigned int y = blockIdx.y * blockDim.y + ty;
    unsigned int x = blockIdx.x * blockDim.x + tx;
    if (x >= nx || y >= ny || z >= nz) return;

    const int sx = tx + kHalo, sy = ty + kHalo, sz = tz + kHalo;
    const int gid = idx(x, y, z);

    s_qxx[sz][sy][sx] = qxx[gid];  s_qxy[sz][sy][sx] = qxy[gid];
    s_qxz[sz][sy][sx] = qxz[gid];  s_qyy[sz][sy][sx] = qyy[gid];
    s_qyz[sz][sy][sx] = qyz[gid];
    s_ux[sz][sy][sx]  = ux[gid];   s_uy[sz][sy][sx]  = uy[gid];
    s_uz[sz][sy][sx]  = uz[gid];

    __syncthreads();

    set_halo(qxx, s_qxx, tx, ty, tz, sx, sy, sz, x, y, z);
    set_halo(qxy, s_qxy, tx, ty, tz, sx, sy, sz, x, y, z);
    set_halo(qxz, s_qxz, tx, ty, tz, sx, sy, sz, x, y, z);
    set_halo(qyy, s_qyy, tx, ty, tz, sx, sy, sz, x, y, z);
    set_halo(qyz, s_qyz, tx, ty, tz, sx, sy, sz, x, y, z);
    set_halo(ux,  s_ux,  tx, ty, tz, sx, sy, sz, x, y, z);
    set_halo(uy,  s_uy,  tx, ty, tz, sx, sy, sz, x, y, z);
    set_halo(uz,  s_uz,  tx, ty, tz, sx, sy, sz, x, y, z);

    __syncthreads();

    const double qxxp = s_qxx[sz][sy][sx], qxyp = s_qxy[sz][sy][sx];
    const double qxzp = s_qxz[sz][sy][sx], qyyp = s_qyy[sz][sy][sx];
    const double qyzp = s_qyz[sz][sy][sx];
    // qzzp = -qxxp - qyyp

    // tr(Q²) = 2*(qxx² + qyy² + qxx·qyy + qxy² + qxz² + qyz²)
    const double trq2 = 2.0*(qxxp*qxxp + qyyp*qyyp + qxxp*qyyp
                            + qxyp*qxyp + qxzp*qxzp + qyzp*qyzp);

    const double kone_thirds = 1.0/3.0;
    const double Q2_xx = qxxp*qxxp + qxyp*qxyp + qxzp*qxzp - kone_thirds * trq2;
    const double Q2_xy = qxxp*qxyp + qxyp*qyyp + qxzp*qyzp;
    const double Q2_xz = qxyp*qyzp - qxzp*qyyp;
    const double Q2_yy = qxyp*qxyp + qyyp*qyyp + qyzp*qyzp - kone_thirds * trq2;
    const double Q2_yz = qxyp*qxzp - qyzp*qxxp;

    const double ld = A + C*trq2;

    // Molecular field H = L·∇²Q - ld·Q
    const double H_xx = L*Laplacian(s_qxx, sx, sy, sz) - ld*qxxp - B * Q2_xx;
    const double H_xy = L*Laplacian(s_qxy, sx, sy, sz) - ld*qxyp - B * Q2_xy;
    const double H_xz = L*Laplacian(s_qxz, sx, sy, sz) - ld*qxzp - B * Q2_xz;
    const double H_yy = L*Laplacian(s_qyy, sx, sy, sz) - ld*qyyp - B * Q2_yy;
    const double H_yz = L*Laplacian(s_qyz, sx, sy, sz) - ld*qyzp - B * Q2_yz;

    // Q gradients (for advection and active force)
    const Vec3 gqxx = Gradient(s_qxx, sx, sy, sz);
    const Vec3 gqxy = Gradient(s_qxy, sx, sy, sz);
    const Vec3 gqxz = Gradient(s_qxz, sx, sy, sz);
    const Vec3 gqyy = Gradient(s_qyy, sx, sy, sz);
    const Vec3 gqyz = Gradient(s_qyz, sx, sy, sz);

    // Velocity gradient tensor: vA_B = ∂(u_A)/∂B
    auto [ux_x, ux_y, ux_z, uy_x, uy_y, uy_z, uz_x, uz_y] =
        VelGradient(s_ux, s_uy, s_uz, sx, sy, sz);

    const double W_xy = 0.5*(ux_y - uy_x);
    const double W_xz = 0.5*(ux_z - uz_x);
    const double W_yz = 0.5*(uy_z - uz_y);

    // Symmetric strain D_AB = (∂u_A/∂B + ∂u_B/∂A) / 2
    const double D_xx = ux_x;
    const double D_xy = 0.5*(ux_y + uy_x);
    const double D_xz = 0.5*(ux_z + uz_x);
    const double D_yy = uy_y;
    const double D_yz = 0.5*(uy_z + uz_y);

    // Co-rotation S = (W·Q - Q·W)
    const double S_xx =  2.0*(W_xy*qxyp + W_xz*qxzp);
    const double S_xy =  W_xy*(qyyp - qxxp) + W_xz*qyzp + W_yz*qxzp;
    const double S_xz =  W_xy*qyzp - W_xz*(2.0*qxxp + qyyp) - W_yz*qxyp;
    const double S_yy = -2.0*W_xy*qxyp + 2.0*W_yz*qyzp;
    const double S_yz = -W_xy*qxzp - W_yz*(qxxp + 2.0*qyyp) - W_xz*qxyp;

        
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
    
    const double tr_QE = (2.0 * D_xx + D_yy) * qxxp + (2.0 * D_xy) * qxyp + (2.0 * D_xz) * qxzp + (D_xx + 2.0 * D_yy) * qyyp + 2.0 * D_yz * qyzp;

    const double aln2_xx = 2.0 * (D_xx * qxxp + D_xy * qxyp + D_xz * qxzp) - ktwo_thirds * tr_QE;
    const double aln2_xy = D_xy * qxxp + D_yy * qxyp + D_yz * qxzp
                            + qxyp * D_xx + qyyp * D_xy + qyzp * D_xz;
    const double aln2_xz = D_xz * qxxp + D_yz * qxyp + (-D_xx - D_yy) * qxzp
                            + qxzp * D_xx + qyzp * D_xy + (-qxxp - qyyp) * D_xz;
    
    const double aln2_yy = 2.0 * (D_xy * qxyp + D_yy * qyyp + D_yz * qyzp) - ktwo_thirds * tr_QE;
    const double aln2_yz = D_xz * qxyp + D_yz * qyyp + (-D_xx - D_yy) * qyzp
                            + qxzp * D_xy + qyzp * D_yy + (-qxxp - qyyp) * D_yz;

    const Vec3 u{s_ux[sz][sy][sx], s_uy[sz][sy][sx], s_uz[sz][sy][sx]};

    qxx_new[gid] = qxxp + DT*(GAMMA*H_xx + S_xx + LAMBDA * (ktwo_thirds * D_xx + aln2_xx) - u.Dot(gqxx));
    qxy_new[gid] = qxyp + DT*(GAMMA*H_xy + S_xy + LAMBDA * (ktwo_thirds * D_xy + aln2_xy) - u.Dot(gqxy));
    qxz_new[gid] = qxzp + DT*(GAMMA*H_xz + S_xz + LAMBDA * (ktwo_thirds * D_xz + aln2_xz) - u.Dot(gqxz));
    qyy_new[gid] = qyyp + DT*(GAMMA*H_yy + S_yy + LAMBDA * (ktwo_thirds * D_yy + aln2_yy) - u.Dot(gqyy));
    qyz_new[gid] = qyzp + DT*(GAMMA*H_yz + S_yz + LAMBDA * (ktwo_thirds * D_yz + aln2_yz) - u.Dot(gqyz));

    // Active force: f_a = -ALPHA*(div Q)_a - MU*u_a
    // qzz = -qxx - qyy, so ∂z(qzz) = -gqxx.z - gqyy.z
    force_x[gid] = -ALPHA*(gqxx.x + gqxy.y + gqxz.z) - MU*u.x;
    force_y[gid] = -ALPHA*(gqxy.x + gqyy.y + gqyz.z) - MU*u.y;
    force_z[gid] = -ALPHA*(gqxz.x + gqyz.y - gqxx.z - gqyy.z) - MU*u.z;
}
