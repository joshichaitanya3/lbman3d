// Transport-agnostic GPU pack/unpack kernels for halo exchange.
//
// No NVSHMEM or MPI dependencies. The __global__ kernels live in an anonymous
// namespace (internal linkage); callers use the Launch* wrappers declared in
// halo_pack_kernels.h. Does not require -rdc=true.

#include "halo_pack_kernels.h"
#include "cuda_utils.h"

namespace {

// Launch geometry — 16×4 per block, field index on gridDim.z.
constexpr int kBlockA = 16;
constexpr int kBlockB = 4;

dim3 PackBlock() { return dim3{kBlockA, kBlockB, 1}; }

dim3 PackGrid(int extentA, int extentB, int nfields) {
    return dim3{
        static_cast<unsigned>((extentA + kBlockA - 1) / kBlockA),
        static_cast<unsigned>((extentB + kBlockB - 1) / kBlockB),
        static_cast<unsigned>(nfields)
    };
}

// -------- X axis (YZ face, transverse: y, z) --------
__global__ void PackAxisX(HaloFieldPtrsConst fields, int nfields,
                          double* out_lo, double* out_hi,
                          LocalGrid g, size_t face_stride) {
    const int y  = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int z  = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int fi = static_cast<int>(blockIdx.z);
    if (y >= g.local_ny || z >= g.local_nz || fi >= nfields) return;
    const int packed  = z * g.local_ny + y;
    const int lo_flat = g.halo_idx(0,              y, z);
    const int hi_flat = g.halo_idx(g.local_nx - 1, y, z);
    out_lo[fi * face_stride + packed] = fields.p[fi][lo_flat];
    out_hi[fi * face_stride + packed] = fields.p[fi][hi_flat];
}

__global__ void UnpackAxisX(HaloFieldPtrsMut fields, int nfields,
                            const double* in_lo, const double* in_hi,
                            LocalGrid g, size_t face_stride) {
    const int y  = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int z  = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int fi = static_cast<int>(blockIdx.z);
    if (y >= g.local_ny || z >= g.local_nz || fi >= nfields) return;
    const int packed  = z * g.local_ny + y;
    const int lo_flat = g.halo_idx(-1,          y, z);
    const int hi_flat = g.halo_idx(g.local_nx,  y, z);
    fields.p[fi][lo_flat] = in_lo[fi * face_stride + packed];
    fields.p[fi][hi_flat] = in_hi[fi * face_stride + packed];
}

// -------- Y axis (XZ face, transverse: x, z) --------
__global__ void PackAxisY(HaloFieldPtrsConst fields, int nfields,
                          double* out_lo, double* out_hi,
                          LocalGrid g, size_t face_stride) {
    const int x  = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int z  = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int fi = static_cast<int>(blockIdx.z);
    if (x >= g.local_nx || z >= g.local_nz || fi >= nfields) return;
    const int packed  = z * g.local_nx + x;
    const int lo_flat = g.halo_idx(x, 0,              z);
    const int hi_flat = g.halo_idx(x, g.local_ny - 1, z);
    out_lo[fi * face_stride + packed] = fields.p[fi][lo_flat];
    out_hi[fi * face_stride + packed] = fields.p[fi][hi_flat];
}

__global__ void UnpackAxisY(HaloFieldPtrsMut fields, int nfields,
                            const double* in_lo, const double* in_hi,
                            LocalGrid g, size_t face_stride) {
    const int x  = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int z  = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int fi = static_cast<int>(blockIdx.z);
    if (x >= g.local_nx || z >= g.local_nz || fi >= nfields) return;
    const int packed  = z * g.local_nx + x;
    const int lo_flat = g.halo_idx(x, -1,         z);
    const int hi_flat = g.halo_idx(x, g.local_ny, z);
    fields.p[fi][lo_flat] = in_lo[fi * face_stride + packed];
    fields.p[fi][hi_flat] = in_hi[fi * face_stride + packed];
}

// -------- Z axis (XY face, transverse: x, y) --------
__global__ void PackAxisZ(HaloFieldPtrsConst fields, int nfields,
                          double* out_lo, double* out_hi,
                          LocalGrid g, size_t face_stride) {
    const int x  = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y  = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int fi = static_cast<int>(blockIdx.z);
    if (x >= g.local_nx || y >= g.local_ny || fi >= nfields) return;
    const int packed  = y * g.local_nx + x;
    const int lo_flat = g.halo_idx(x, y, 0);
    const int hi_flat = g.halo_idx(x, y, g.local_nz - 1);
    out_lo[fi * face_stride + packed] = fields.p[fi][lo_flat];
    out_hi[fi * face_stride + packed] = fields.p[fi][hi_flat];
}

__global__ void UnpackAxisZ(HaloFieldPtrsMut fields, int nfields,
                            const double* in_lo, const double* in_hi,
                            LocalGrid g, size_t face_stride) {
    const int x  = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y  = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int fi = static_cast<int>(blockIdx.z);
    if (x >= g.local_nx || y >= g.local_ny || fi >= nfields) return;
    const int packed  = y * g.local_nx + x;
    const int lo_flat = g.halo_idx(x, y, -1);
    const int hi_flat = g.halo_idx(x, y, g.local_nz);
    fields.p[fi][lo_flat] = in_lo[fi * face_stride + packed];
    fields.p[fi][hi_flat] = in_hi[fi * face_stride + packed];
}

}  // namespace

// ---- External wrappers ----

void LaunchPackAxisX(HaloFieldPtrsConst fields, int nfields,
                     double* out_lo, double* out_hi,
                     const LocalGrid& g, std::size_t face_stride,
                     cudaStream_t stream) {
    PackAxisX<<<PackGrid(g.local_ny, g.local_nz, nfields), PackBlock(), 0, stream>>>(
        fields, nfields, out_lo, out_hi, g, face_stride);
    checkCudaErrors(cudaGetLastError());
}

void LaunchUnpackAxisX(HaloFieldPtrsMut fields, int nfields,
                       const double* in_lo, const double* in_hi,
                       const LocalGrid& g, std::size_t face_stride,
                       cudaStream_t stream) {
    UnpackAxisX<<<PackGrid(g.local_ny, g.local_nz, nfields), PackBlock(), 0, stream>>>(
        fields, nfields, in_lo, in_hi, g, face_stride);
    checkCudaErrors(cudaGetLastError());
}

void LaunchPackAxisY(HaloFieldPtrsConst fields, int nfields,
                     double* out_lo, double* out_hi,
                     const LocalGrid& g, std::size_t face_stride,
                     cudaStream_t stream) {
    PackAxisY<<<PackGrid(g.local_nx, g.local_nz, nfields), PackBlock(), 0, stream>>>(
        fields, nfields, out_lo, out_hi, g, face_stride);
    checkCudaErrors(cudaGetLastError());
}

void LaunchUnpackAxisY(HaloFieldPtrsMut fields, int nfields,
                       const double* in_lo, const double* in_hi,
                       const LocalGrid& g, std::size_t face_stride,
                       cudaStream_t stream) {
    UnpackAxisY<<<PackGrid(g.local_nx, g.local_nz, nfields), PackBlock(), 0, stream>>>(
        fields, nfields, in_lo, in_hi, g, face_stride);
    checkCudaErrors(cudaGetLastError());
}

void LaunchPackAxisZ(HaloFieldPtrsConst fields, int nfields,
                     double* out_lo, double* out_hi,
                     const LocalGrid& g, std::size_t face_stride,
                     cudaStream_t stream) {
    PackAxisZ<<<PackGrid(g.local_nx, g.local_ny, nfields), PackBlock(), 0, stream>>>(
        fields, nfields, out_lo, out_hi, g, face_stride);
    checkCudaErrors(cudaGetLastError());
}

void LaunchUnpackAxisZ(HaloFieldPtrsMut fields, int nfields,
                       const double* in_lo, const double* in_hi,
                       const LocalGrid& g, std::size_t face_stride,
                       cudaStream_t stream) {
    UnpackAxisZ<<<PackGrid(g.local_nx, g.local_ny, nfields), PackBlock(), 0, stream>>>(
        fields, nfields, in_lo, in_hi, g, face_stride);
    checkCudaErrors(cudaGetLastError());
}

// ============================================================================
// LBM face pack/unpack (PR VII-g, face-only)
// ============================================================================

namespace {

// Grid geometry helper for LBM pack/unpack. Same block shape as the Q pack,
// but the gridDim.z is fixed at 5 (crossing dirs per face) not nfields.
dim3 LbmPackGrid(int extentA, int extentB) {
    return dim3{
        static_cast<unsigned>((extentA + kBlockA - 1) / kBlockA),
        static_cast<unsigned>((extentB + kBlockB - 1) / kBlockB),
        5u   // one block-row per crossing dir
    };
}

// Wall-skip predicate. Returns true if this cell's crossing dir would flow
// into an orthogonal physical wall — in which case local `HandleBoundaryPoint`
// has already written the correct bounce value, and unpack must not clobber
// it.
__device__ inline bool ShouldSkipUnpack(int e_a, int e_b,
                                        int coord_a_global, int coord_b_global,
                                        LbmFaceSkipCtx skip) {
    if (skip.wall_a_lo && coord_a_global == 0                  && e_a > 0) return true;
    if (skip.wall_a_hi && coord_a_global == skip.global_n_a - 1 && e_a < 0) return true;
    if (skip.wall_b_lo && coord_b_global == 0                  && e_b > 0) return true;
    if (skip.wall_b_hi && coord_b_global == skip.global_n_b - 1 && e_b < 0) return true;
    return false;
}

// -------- X axis (YZ face, transverse a=Y, b=Z) --------

__global__ void PackLbmAxisX(const double* d_f,
                             double* send_lo, double* send_hi,
                             LocalGrid g,
                             LbmFaceCrossings lo_face, LbmFaceCrossings hi_face) {
    const int y = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int z = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int k = static_cast<int>(blockIdx.z);
    if (y >= g.local_ny || z >= g.local_nz || k >= 5) return;
    const int face_area = g.local_ny * g.local_nz;
    const int packed    = k * face_area + z * g.local_ny + y;

    // Pack -X ghost (dirs missingXHi, ex<0) from cell (-1, y, z).
    send_lo[packed] = d_f[g.halo_idx(-1,           y, z, lo_face.dir[k])];
    // Pack +X ghost (dirs missingXLo, ex>0) from cell (local_nx, y, z).
    send_hi[packed] = d_f[g.halo_idx(g.local_nx,   y, z, hi_face.dir[k])];
}

__global__ void UnpackLbmAxisX(double* d_f,
                               const double* recv_lo, const double* recv_hi,
                               LocalGrid g,
                               LbmFaceCrossings lo_face, LbmFaceCrossings hi_face,
                               LbmFaceSkipCtx skip) {
    const int y = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int z = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int k = static_cast<int>(blockIdx.z);
    if (y >= g.local_ny || z >= g.local_nz || k >= 5) return;
    const int face_area = g.local_ny * g.local_nz;
    const int packed    = k * face_area + z * g.local_ny + y;
    const int y_g = skip.offset_a + y;
    const int z_g = skip.offset_b + z;

    // Unpack -X owned boundary (dirs missingXLo — dirs the -X neighbour's +X
    // ghost pack sent to us). Dest cell (0, y, z).
    if (!ShouldSkipUnpack(lo_face.e_trans_a[k], lo_face.e_trans_b[k],
                          y_g, z_g, skip)) {
        d_f[g.halo_idx(0, y, z, lo_face.dir[k])] = recv_lo[packed];
    }
    // Unpack +X owned boundary (dirs missingXHi — from +X neighbour's -X
    // ghost pack). Dest cell (local_nx-1, y, z).
    if (!ShouldSkipUnpack(hi_face.e_trans_a[k], hi_face.e_trans_b[k],
                          y_g, z_g, skip)) {
        d_f[g.halo_idx(g.local_nx - 1, y, z, hi_face.dir[k])] = recv_hi[packed];
    }
}

// -------- Y axis (XZ face, transverse a=X, b=Z) --------

__global__ void PackLbmAxisY(const double* d_f,
                             double* send_lo, double* send_hi,
                             LocalGrid g,
                             LbmFaceCrossings lo_face, LbmFaceCrossings hi_face) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int z = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int k = static_cast<int>(blockIdx.z);
    if (x >= g.local_nx || z >= g.local_nz || k >= 5) return;
    const int face_area = g.local_nx * g.local_nz;
    const int packed    = k * face_area + z * g.local_nx + x;

    send_lo[packed] = d_f[g.halo_idx(x, -1,         z, lo_face.dir[k])];
    send_hi[packed] = d_f[g.halo_idx(x, g.local_ny, z, hi_face.dir[k])];
}

__global__ void UnpackLbmAxisY(double* d_f,
                               const double* recv_lo, const double* recv_hi,
                               LocalGrid g,
                               LbmFaceCrossings lo_face, LbmFaceCrossings hi_face,
                               LbmFaceSkipCtx skip) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int z = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int k = static_cast<int>(blockIdx.z);
    if (x >= g.local_nx || z >= g.local_nz || k >= 5) return;
    const int face_area = g.local_nx * g.local_nz;
    const int packed    = k * face_area + z * g.local_nx + x;
    const int x_g = skip.offset_a + x;
    const int z_g = skip.offset_b + z;

    if (!ShouldSkipUnpack(lo_face.e_trans_a[k], lo_face.e_trans_b[k],
                          x_g, z_g, skip)) {
        d_f[g.halo_idx(x, 0, z, lo_face.dir[k])] = recv_lo[packed];
    }
    if (!ShouldSkipUnpack(hi_face.e_trans_a[k], hi_face.e_trans_b[k],
                          x_g, z_g, skip)) {
        d_f[g.halo_idx(x, g.local_ny - 1, z, hi_face.dir[k])] = recv_hi[packed];
    }
}

// -------- Z axis (XY face, transverse a=X, b=Y) --------

__global__ void PackLbmAxisZ(const double* d_f,
                             double* send_lo, double* send_hi,
                             LocalGrid g,
                             LbmFaceCrossings lo_face, LbmFaceCrossings hi_face) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int k = static_cast<int>(blockIdx.z);
    if (x >= g.local_nx || y >= g.local_ny || k >= 5) return;
    const int face_area = g.local_nx * g.local_ny;
    const int packed    = k * face_area + y * g.local_nx + x;

    send_lo[packed] = d_f[g.halo_idx(x, y, -1,         lo_face.dir[k])];
    send_hi[packed] = d_f[g.halo_idx(x, y, g.local_nz, hi_face.dir[k])];
}

__global__ void UnpackLbmAxisZ(double* d_f,
                               const double* recv_lo, const double* recv_hi,
                               LocalGrid g,
                               LbmFaceCrossings lo_face, LbmFaceCrossings hi_face,
                               LbmFaceSkipCtx skip) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int k = static_cast<int>(blockIdx.z);
    if (x >= g.local_nx || y >= g.local_ny || k >= 5) return;
    const int face_area = g.local_nx * g.local_ny;
    const int packed    = k * face_area + y * g.local_nx + x;
    const int x_g = skip.offset_a + x;
    const int y_g = skip.offset_b + y;

    if (!ShouldSkipUnpack(lo_face.e_trans_a[k], lo_face.e_trans_b[k],
                          x_g, y_g, skip)) {
        d_f[g.halo_idx(x, y, 0, lo_face.dir[k])] = recv_lo[packed];
    }
    if (!ShouldSkipUnpack(hi_face.e_trans_a[k], hi_face.e_trans_b[k],
                          x_g, y_g, skip)) {
        d_f[g.halo_idx(x, y, g.local_nz - 1, hi_face.dir[k])] = recv_hi[packed];
    }
}

}  // namespace

// ---- LBM external wrappers ----

void LaunchPackLbmAxisX(const double* d_f,
                        double* send_lo, double* send_hi,
                        LbmFaceCrossings lo_face, LbmFaceCrossings hi_face,
                        const LocalGrid& g, cudaStream_t stream) {
    PackLbmAxisX<<<LbmPackGrid(g.local_ny, g.local_nz), PackBlock(), 0, stream>>>(
        d_f, send_lo, send_hi, g, lo_face, hi_face);
    checkCudaErrors(cudaGetLastError());
}

void LaunchUnpackLbmAxisX(double* d_f,
                          const double* recv_lo, const double* recv_hi,
                          LbmFaceCrossings lo_face, LbmFaceCrossings hi_face,
                          LbmFaceSkipCtx skip,
                          const LocalGrid& g, cudaStream_t stream) {
    UnpackLbmAxisX<<<LbmPackGrid(g.local_ny, g.local_nz), PackBlock(), 0, stream>>>(
        d_f, recv_lo, recv_hi, g, lo_face, hi_face, skip);
    checkCudaErrors(cudaGetLastError());
}

void LaunchPackLbmAxisY(const double* d_f,
                        double* send_lo, double* send_hi,
                        LbmFaceCrossings lo_face, LbmFaceCrossings hi_face,
                        const LocalGrid& g, cudaStream_t stream) {
    PackLbmAxisY<<<LbmPackGrid(g.local_nx, g.local_nz), PackBlock(), 0, stream>>>(
        d_f, send_lo, send_hi, g, lo_face, hi_face);
    checkCudaErrors(cudaGetLastError());
}

void LaunchUnpackLbmAxisY(double* d_f,
                          const double* recv_lo, const double* recv_hi,
                          LbmFaceCrossings lo_face, LbmFaceCrossings hi_face,
                          LbmFaceSkipCtx skip,
                          const LocalGrid& g, cudaStream_t stream) {
    UnpackLbmAxisY<<<LbmPackGrid(g.local_nx, g.local_nz), PackBlock(), 0, stream>>>(
        d_f, recv_lo, recv_hi, g, lo_face, hi_face, skip);
    checkCudaErrors(cudaGetLastError());
}

void LaunchPackLbmAxisZ(const double* d_f,
                        double* send_lo, double* send_hi,
                        LbmFaceCrossings lo_face, LbmFaceCrossings hi_face,
                        const LocalGrid& g, cudaStream_t stream) {
    PackLbmAxisZ<<<LbmPackGrid(g.local_nx, g.local_ny), PackBlock(), 0, stream>>>(
        d_f, send_lo, send_hi, g, lo_face, hi_face);
    checkCudaErrors(cudaGetLastError());
}

void LaunchUnpackLbmAxisZ(double* d_f,
                          const double* recv_lo, const double* recv_hi,
                          LbmFaceCrossings lo_face, LbmFaceCrossings hi_face,
                          LbmFaceSkipCtx skip,
                          const LocalGrid& g, cudaStream_t stream) {
    UnpackLbmAxisZ<<<LbmPackGrid(g.local_nx, g.local_ny), PackBlock(), 0, stream>>>(
        d_f, recv_lo, recv_hi, g, lo_face, hi_face, skip);
    checkCudaErrors(cudaGetLastError());
}
