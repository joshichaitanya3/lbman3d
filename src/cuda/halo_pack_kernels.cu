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
