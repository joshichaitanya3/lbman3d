// PR VII-e — Q-tensor + velocity halo exchange on NVSHMEM.
//
// This TU replaces the VII-a link-only stub with the real ExchangeQTensor
// implementation. See src/cuda/CLAUDE.md → "NVSHMEM multi-GPU roadmap" for the
// full plan, and halo_exchange_qtensor_nvshmem.h for the class-level rationale.
//
// Physics (unchanged from the CPU HaloExchangeQTensor::ExchangeQTensor):
//   1. Pack owned-boundary planes of 5 Q + 3 velocity fields into send buffers.
//   2. One-sided put each face's send buffer into the neighbour's paired recv
//      buffer, on the same CUDA stream as the physics kernels.
//   3. Barrier so every PE's incoming puts have landed.
//   4. Unpack recv buffers into this rank's ghost cells.
//
// Portability seam: pack/unpack kernels are transport-agnostic — they take a
// plain double* buffer whose location the caller chose. Only ExchangeQTensor
// itself calls NVSHMEM. Swapping in CUDA-aware MPI later is a change to
// ExchangeQTensor's body, not to the kernels.

#include <nvshmem.h>
#include <nvshmemx.h>

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdexcept>

#include "cuda_utils.h"
#include "device_fields.h"
#include "halo_exchange_qtensor_nvshmem.h"
#include "local_grid.h"
#include "mpi/mpi_context.h"
#include <params.h>

namespace {

// Packed pointer bundle so a single kernel launch can pack all 8 fields for one
// axis. Passed by value (~64 B) into the kernel — well under the 4 KB param
// limit, and every thread reads the pointers from registers instead of global
// memory.
struct FieldPtrsConst {
    const double* p[HaloExchangeQTensorNvshmem::kMaxFields];
};
struct FieldPtrsMut {
    double* p[HaloExchangeQTensorNvshmem::kMaxFields];
};

// Pack/unpack kernels — one per axis. `nfields <= kMaxFields` lets the pack
// unit test target a single field without paying for eight kernel writes.
//
// Slot layout matches the CPU pack: for an X-axis face,
//     send_buf[fi * max_yz + z * local_ny + y] = field[halo_idx(x_bdy, y, z)]
// with `x_bdy = 0` for lo, `local_nx-1` for hi. Y and Z axes rotate the
// same rule.

// -------- X axis (packs lo-x and hi-x faces, transverse coords (y, z)) --------
__global__ void PackAxisX(FieldPtrsConst fields, int nfields,
                          double* out_lo, double* out_hi,
                          LocalGrid g, size_t face_stride) {
    const int y  = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int z  = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int fi = static_cast<int>(blockIdx.z);
    if (y >= g.local_ny || z >= g.local_nz || fi >= nfields) return;
    const int packed  = z * g.local_ny + y;
    const int lo_flat = g.halo_idx(0,               y, z);
    const int hi_flat = g.halo_idx(g.local_nx - 1,  y, z);
    out_lo[fi * face_stride + packed] = fields.p[fi][lo_flat];
    out_hi[fi * face_stride + packed] = fields.p[fi][hi_flat];
}

__global__ void UnpackAxisX(FieldPtrsMut fields, int nfields,
                            const double* in_lo, const double* in_hi,
                            LocalGrid g, size_t face_stride) {
    const int y  = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int z  = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int fi = static_cast<int>(blockIdx.z);
    if (y >= g.local_ny || z >= g.local_nz || fi >= nfields) return;
    const int packed  = z * g.local_ny + y;
    const int lo_flat = g.halo_idx(-1,           y, z);
    const int hi_flat = g.halo_idx(g.local_nx,   y, z);
    fields.p[fi][lo_flat] = in_lo[fi * face_stride + packed];
    fields.p[fi][hi_flat] = in_hi[fi * face_stride + packed];
}

// -------- Y axis (packs lo-y and hi-y faces, transverse coords (x, z)) --------
__global__ void PackAxisY(FieldPtrsConst fields, int nfields,
                          double* out_lo, double* out_hi,
                          LocalGrid g, size_t face_stride) {
    const int x  = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int z  = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int fi = static_cast<int>(blockIdx.z);
    if (x >= g.local_nx || z >= g.local_nz || fi >= nfields) return;
    const int packed  = z * g.local_nx + x;
    const int lo_flat = g.halo_idx(x, 0,               z);
    const int hi_flat = g.halo_idx(x, g.local_ny - 1,  z);
    out_lo[fi * face_stride + packed] = fields.p[fi][lo_flat];
    out_hi[fi * face_stride + packed] = fields.p[fi][hi_flat];
}

__global__ void UnpackAxisY(FieldPtrsMut fields, int nfields,
                            const double* in_lo, const double* in_hi,
                            LocalGrid g, size_t face_stride) {
    const int x  = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int z  = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int fi = static_cast<int>(blockIdx.z);
    if (x >= g.local_nx || z >= g.local_nz || fi >= nfields) return;
    const int packed  = z * g.local_nx + x;
    const int lo_flat = g.halo_idx(x, -1,          z);
    const int hi_flat = g.halo_idx(x, g.local_ny,  z);
    fields.p[fi][lo_flat] = in_lo[fi * face_stride + packed];
    fields.p[fi][hi_flat] = in_hi[fi * face_stride + packed];
}

// -------- Z axis (packs lo-z and hi-z faces, transverse coords (x, y)) --------
__global__ void PackAxisZ(FieldPtrsConst fields, int nfields,
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

__global__ void UnpackAxisZ(FieldPtrsMut fields, int nfields,
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

// Launch geometry — small transverse blocks with the field axis on gridDim.z.
// 16×4 is well under 512 threads/block, and the axis loops are typically
// small enough (64 × 128 on the benchmark shape) that occupancy is bounded by
// the launch grid, not by resource pressure.
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

// Build the 8-field pointer bundle in the exact order the CPU exchange uses,
// so a future side-by-side correctness check between CPU and GPU halos can
// compare field-by-field without re-mapping.
FieldPtrsConst BuildQTensorFieldPtrs(const DeviceFields& df) {
    FieldPtrsConst ptrs{};
    ptrs.p[0] = df.d_qxx;
    ptrs.p[1] = df.d_qxy;
    ptrs.p[2] = df.d_qxz;
    ptrs.p[3] = df.d_qyy;
    ptrs.p[4] = df.d_qyz;
    ptrs.p[5] = df.d_ux.data().get();
    ptrs.p[6] = df.d_uy.data().get();
    ptrs.p[7] = df.d_uz.data().get();
    return ptrs;
}

FieldPtrsMut BuildQTensorFieldPtrsMut(DeviceFields& df) {
    FieldPtrsMut ptrs{};
    ptrs.p[0] = df.d_qxx;
    ptrs.p[1] = df.d_qxy;
    ptrs.p[2] = df.d_qxz;
    ptrs.p[3] = df.d_qyy;
    ptrs.p[4] = df.d_qyz;
    ptrs.p[5] = df.d_ux.data().get();
    ptrs.p[6] = df.d_uy.data().get();
    ptrs.p[7] = df.d_uz.data().get();
    return ptrs;
}

}  // namespace

HaloExchangeQTensorNvshmem::HaloExchangeQTensorNvshmem(
    const LocalGrid& grid, const MPIContext& mpi)
    : grid_(grid),
      world_size_(mpi.world_size),
      max_yz_(0),
      max_xz_(0),
      max_xy_(0)
{
    for (int f = 0; f < 6; ++f) {
        send_buf_[f] = nullptr;
        recv_buf_[f] = nullptr;
    }
    for (int d = 0; d < 3; ++d) {
        neighbor_lo_[d] = MPI_PROC_NULL;
        neighbor_hi_[d] = MPI_PROC_NULL;
        MPI_Cart_shift(mpi.cart_comm, d, 1, &neighbor_lo_[d], &neighbor_hi_[d]);
    }

    // Max face area over all ranks — deterministic from global dims and rank
    // count, so every PE computes the same value without a collective. Matches
    // the CPU HaloExchangeQTensor sizing exactly.
    auto ceil_div = [](int global, int n) { return (global + n - 1) / n; };
    max_yz_ = static_cast<size_t>(ceil_div(Params::ny, mpi.dims[1]))
            * static_cast<size_t>(ceil_div(Params::nz, mpi.dims[2]));
    max_xz_ = static_cast<size_t>(ceil_div(Params::nx, mpi.dims[0]))
            * static_cast<size_t>(ceil_div(Params::nz, mpi.dims[2]));
    max_xy_ = static_cast<size_t>(ceil_div(Params::nx, mpi.dims[0]))
            * static_cast<size_t>(ceil_div(Params::ny, mpi.dims[1]));

    // Symmetric-heap allocation. Allocated in a fixed order on every PE so
    // offsets align — NVSHMEM's "symmetric" guarantee is that the k-th
    // nvshmem_malloc on any PE lands at the same virtual offset (as long as
    // every PE's k-th call is the same size), and one-sided puts use exactly
    // that offset. Passing 0 bytes here on ranks where a face happens to be
    // unused would BREAK the symmetric property, so we always allocate even
    // when the face is at a physical wall.
    const size_t bytes_yz = sizeof(double) * kMaxFields * max_yz_;
    const size_t bytes_xz = sizeof(double) * kMaxFields * max_xz_;
    const size_t bytes_xy = sizeof(double) * kMaxFields * max_xy_;
    const size_t face_bytes[6] = { bytes_yz, bytes_yz,
                                   bytes_xz, bytes_xz,
                                   bytes_xy, bytes_xy };

    for (int f = 0; f < 6; ++f) {
        send_buf_[f] = static_cast<double*>(nvshmem_malloc(face_bytes[f]));
        recv_buf_[f] = static_cast<double*>(nvshmem_malloc(face_bytes[f]));
        if (!send_buf_[f] || !recv_buf_[f]) {
            throw std::runtime_error(
                "HaloExchangeQTensorNvshmem: nvshmem_malloc failed for face buffer");
        }
        // Zero recv so an untouched slot (physical-wall face where no put lands)
        // does not surface last run's leftover into ghost cells. Pack overwrites
        // send every step, so it does not need zeroing here.
        checkCudaErrors(cudaMemset(recv_buf_[f], 0, face_bytes[f]));
    }
}

HaloExchangeQTensorNvshmem::~HaloExchangeQTensorNvshmem() {
    for (int f = 0; f < 6; ++f) {
        if (send_buf_[f]) nvshmem_free(send_buf_[f]);
        if (recv_buf_[f]) nvshmem_free(recv_buf_[f]);
    }
}

std::size_t HaloExchangeQTensorNvshmem::face_area(int axis) const {
    switch (axis) {
        case 0: return max_yz_;
        case 1: return max_xz_;
        case 2: return max_xy_;
        default: return 0;
    }
}

void HaloExchangeQTensorNvshmem::ExchangeQTensor(DeviceFields& df, cudaStream_t stream) {
    // Single-rank fast path. Skipping the barrier is essential — a global
    // barrier with one PE would still work, but the guard also matches the CPU
    // exchange's early-return and lets local nranks=1 regression keep the
    // exchange out of the timeline entirely.
    if (world_size_ == 1) return;

    FieldPtrsConst src_ptrs = BuildQTensorFieldPtrs(df);
    FieldPtrsMut   dst_ptrs = BuildQTensorFieldPtrsMut(df);
    const int nfields = static_cast<int>(kMaxFields);

    // ---- 1. Pack all six faces on the stream. ----
    PackAxisX<<<PackGrid(grid_.local_ny, grid_.local_nz, nfields),
                PackBlock(), 0, stream>>>(
        src_ptrs, nfields, send_buf_[0], send_buf_[1], grid_, max_yz_);
    PackAxisY<<<PackGrid(grid_.local_nx, grid_.local_nz, nfields),
                PackBlock(), 0, stream>>>(
        src_ptrs, nfields, send_buf_[2], send_buf_[3], grid_, max_xz_);
    PackAxisZ<<<PackGrid(grid_.local_nx, grid_.local_ny, nfields),
                PackBlock(), 0, stream>>>(
        src_ptrs, nfields, send_buf_[4], send_buf_[5], grid_, max_xy_);
    checkCudaErrors(cudaGetLastError());

    // ---- 2. One-sided puts. ----
    // Rank r's lo face is neighbor_lo's hi ghost (and vice versa). The
    // destination pointer is expressed as a symmetric-heap address on THIS
    // PE — NVSHMEM translates to the same-offset allocation on `peer_pe`.
    // Skips on MPI_PROC_NULL faces (physical wall): the ghost value there is
    // resolved by boundary_handler at read time, not by exchange, and NVSHMEM
    // rejects PE = -2.
    const size_t n_yz = kMaxFields * max_yz_;
    const size_t n_xz = kMaxFields * max_xz_;
    const size_t n_xy = kMaxFields * max_xy_;

    if (neighbor_lo_[0] != MPI_PROC_NULL) {
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[1], send_buf_[0], n_yz, neighbor_lo_[0], stream);
    }
    if (neighbor_hi_[0] != MPI_PROC_NULL) {
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[0], send_buf_[1], n_yz, neighbor_hi_[0], stream);
    }
    if (neighbor_lo_[1] != MPI_PROC_NULL) {
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[3], send_buf_[2], n_xz, neighbor_lo_[1], stream);
    }
    if (neighbor_hi_[1] != MPI_PROC_NULL) {
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[2], send_buf_[3], n_xz, neighbor_hi_[1], stream);
    }
    if (neighbor_lo_[2] != MPI_PROC_NULL) {
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[5], send_buf_[4], n_xy, neighbor_lo_[2], stream);
    }
    if (neighbor_hi_[2] != MPI_PROC_NULL) {
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[4], send_buf_[5], n_xy, neighbor_hi_[2], stream);
    }

    // ---- 3. Barrier: puts sent to me have completed and are visible. ----
    // Global barrier is fine — every PE participates in ExchangeQTensor at
    // every step (world_size == 1 already returned above), so no deadlock.
    nvshmemx_barrier_all_on_stream(stream);

    // ---- 4. Unpack all six faces on the stream. ----
    UnpackAxisX<<<PackGrid(grid_.local_ny, grid_.local_nz, nfields),
                  PackBlock(), 0, stream>>>(
        dst_ptrs, nfields, recv_buf_[0], recv_buf_[1], grid_, max_yz_);
    UnpackAxisY<<<PackGrid(grid_.local_nx, grid_.local_nz, nfields),
                  PackBlock(), 0, stream>>>(
        dst_ptrs, nfields, recv_buf_[2], recv_buf_[3], grid_, max_xz_);
    UnpackAxisZ<<<PackGrid(grid_.local_nx, grid_.local_ny, nfields),
                  PackBlock(), 0, stream>>>(
        dst_ptrs, nfields, recv_buf_[4], recv_buf_[5], grid_, max_xy_);
    checkCudaErrors(cudaGetLastError());
}

void HaloExchangeQTensorNvshmem::PackSingleFieldForTest(
    const double* d_field, std::size_t field_idx, int axis, cudaStream_t stream) {
    // Test path — pack one field into the axis's send buffers at slot
    // field_idx, no puts, no barrier. Lets the pack unit test verify slot
    // layout at nranks = 1 without needing a real neighbour.
    FieldPtrsConst src{};
    for (std::size_t i = 0; i < kMaxFields; ++i) src.p[i] = d_field;
    const int nfields = static_cast<int>(field_idx) + 1;

    switch (axis) {
        case 0:
            PackAxisX<<<PackGrid(grid_.local_ny, grid_.local_nz, nfields),
                        PackBlock(), 0, stream>>>(
                src, nfields, send_buf_[0], send_buf_[1], grid_, max_yz_);
            break;
        case 1:
            PackAxisY<<<PackGrid(grid_.local_nx, grid_.local_nz, nfields),
                        PackBlock(), 0, stream>>>(
                src, nfields, send_buf_[2], send_buf_[3], grid_, max_xz_);
            break;
        case 2:
            PackAxisZ<<<PackGrid(grid_.local_nx, grid_.local_ny, nfields),
                        PackBlock(), 0, stream>>>(
                src, nfields, send_buf_[4], send_buf_[5], grid_, max_xy_);
            break;
        default:
            throw std::runtime_error("PackSingleFieldForTest: axis out of range");
    }
    checkCudaErrors(cudaGetLastError());
}

// Keep the VII-a link-only stubs so anything already referencing them (build
// diagnostics, prior benchmarks) still links.
namespace lbm::nvshmem_stub {

__global__ void PingPeKernel(int* out) {
    if (out != nullptr && threadIdx.x == 0 && blockIdx.x == 0) {
        *out = nvshmem_my_pe();
    }
}

extern "C" int LbmNvshmemStubHostRef() {
    return nvshmem_my_pe();
}

}  // namespace lbm::nvshmem_stub
