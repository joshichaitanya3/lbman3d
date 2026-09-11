// PR VII-e — NVSHMEM transport layer for Q-tensor + velocity halo exchange.
//
// This TU is the only one that calls NVSHMEM. Pack/unpack kernels live in
// halo_pack_kernels.cu (no NVSHMEM dependency); this file calls their
// Launch* wrappers. Swapping in a CUDA-aware MPI backend means replacing
// only this file — the pack kernels and the exchange interface are unchanged.
//
// Physics (unchanged from the CPU HaloExchangeQTensor::ExchangeQTensor):
//   1. Pack owned-boundary planes of 5 Q + 3 velocity fields into send bufs.
//   2. One-sided put each face's send buf into the neighbour's recv buf,
//      enqueued on the same CUDA stream as the physics kernels.
//   3. Barrier so every PE's incoming puts have landed before unpack fires.
//   4. Unpack recv bufs into this rank's ghost cells.

#include <nvshmem.h>
#include <nvshmemx.h>

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdexcept>

#include "cuda_utils.h"
#include "device_fields.h"
#include "halo_exchange_lbm_nvshmem.h"
#include "halo_exchange_passive_stresses_nvshmem.h"
#include "halo_exchange_qtensor_nvshmem.h"
#include "halo_pack_kernels.h"
#include "lattice_stencil.h"
#include "local_grid.h"
#include "mpi/mpi_context.h"
#include <params.h>

namespace {

// Build the 8-field pointer bundle in the order the CPU exchange uses, so a
// future side-by-side correctness check can compare field-by-field without
// re-mapping. Q-tensor-specific; depends on DeviceFields so it stays here,
// not in the transport-agnostic halo_pack_kernels.cu.
HaloFieldPtrsConst BuildQTensorFieldPtrs(const DeviceFields& df) {
    HaloFieldPtrsConst ptrs{};
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

HaloFieldPtrsMut BuildQTensorFieldPtrsMut(DeviceFields& df) {
    HaloFieldPtrsMut ptrs{};
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

// Field order matches HaloExchangeQTensor::ExchangePassiveStresses on the CPU
// path so a side-by-side pack-buffer diff between backends stays trivial:
// 5 Q + 5 Σ + 3 τ. Q lives at slots 0-4 because both phase-1 (writes Q_new
// pointwise) and phase-2 (reads Q at neighbours) need fresh Q ghosts here;
// see halo_exchange_passive_stresses_nvshmem.h.
HaloFieldPtrsConst BuildPassiveStressFieldPtrs(const DeviceFields& df) {
    HaloFieldPtrsConst ptrs{};
    ptrs.p[0]  = df.d_qxx;
    ptrs.p[1]  = df.d_qxy;
    ptrs.p[2]  = df.d_qxz;
    ptrs.p[3]  = df.d_qyy;
    ptrs.p[4]  = df.d_qyz;
    ptrs.p[5]  = df.d_Sigma_xx;
    ptrs.p[6]  = df.d_Sigma_xy;
    ptrs.p[7]  = df.d_Sigma_xz;
    ptrs.p[8]  = df.d_Sigma_yy;
    ptrs.p[9]  = df.d_Sigma_yz;
    ptrs.p[10] = df.d_Tau_xy;
    ptrs.p[11] = df.d_Tau_xz;
    ptrs.p[12] = df.d_Tau_yz;
    return ptrs;
}

HaloFieldPtrsMut BuildPassiveStressFieldPtrsMut(DeviceFields& df) {
    HaloFieldPtrsMut ptrs{};
    ptrs.p[0]  = df.d_qxx;
    ptrs.p[1]  = df.d_qxy;
    ptrs.p[2]  = df.d_qxz;
    ptrs.p[3]  = df.d_qyy;
    ptrs.p[4]  = df.d_qyz;
    ptrs.p[5]  = df.d_Sigma_xx;
    ptrs.p[6]  = df.d_Sigma_xy;
    ptrs.p[7]  = df.d_Sigma_xz;
    ptrs.p[8]  = df.d_Sigma_yy;
    ptrs.p[9]  = df.d_Sigma_yz;
    ptrs.p[10] = df.d_Tau_xy;
    ptrs.p[11] = df.d_Tau_xz;
    ptrs.p[12] = df.d_Tau_yz;
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
    // every PE's k-th call requests the same size), and one-sided puts use
    // exactly that offset. Always allocating — even on physical-wall faces
    // where no put lands — preserves that symmetry.
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
        // does not surface last run's leftover into ghost cells.
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
    // Single-rank fast path — matches the CPU exchange's early-return and
    // keeps the exchange out of the timeline entirely at nranks = 1.
    if (world_size_ == 1) return;

    HaloFieldPtrsConst src_ptrs = BuildQTensorFieldPtrs(df);
    HaloFieldPtrsMut   dst_ptrs = BuildQTensorFieldPtrsMut(df);
    const int nfields = static_cast<int>(kMaxFields);

    // ---- 1. Pack all six faces on the stream. ----
    LaunchPackAxisX(src_ptrs, nfields, send_buf_[0], send_buf_[1], grid_, max_yz_, stream);
    LaunchPackAxisY(src_ptrs, nfields, send_buf_[2], send_buf_[3], grid_, max_xz_, stream);
    LaunchPackAxisZ(src_ptrs, nfields, send_buf_[4], send_buf_[5], grid_, max_xy_, stream);

    // ---- 2. One-sided puts. ----
    // Rank r's lo face is neighbor_lo's hi ghost (and vice versa). The
    // destination pointer is expressed as a symmetric-heap address on THIS
    // PE — NVSHMEM translates to the same-offset allocation on `peer_pe`.
    // Skips on MPI_PROC_NULL faces (physical wall): the ghost value is
    // resolved by boundary_handler at read time, and NVSHMEM rejects PE = -2.
    const size_t n_yz = kMaxFields * max_yz_;
    const size_t n_xz = kMaxFields * max_xz_;
    const size_t n_xy = kMaxFields * max_xy_;

    if (neighbor_lo_[0] != MPI_PROC_NULL)
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[1], send_buf_[0], n_yz, neighbor_lo_[0], stream);
    if (neighbor_hi_[0] != MPI_PROC_NULL)
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[0], send_buf_[1], n_yz, neighbor_hi_[0], stream);
    if (neighbor_lo_[1] != MPI_PROC_NULL)
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[3], send_buf_[2], n_xz, neighbor_lo_[1], stream);
    if (neighbor_hi_[1] != MPI_PROC_NULL)
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[2], send_buf_[3], n_xz, neighbor_hi_[1], stream);
    if (neighbor_lo_[2] != MPI_PROC_NULL)
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[5], send_buf_[4], n_xy, neighbor_lo_[2], stream);
    if (neighbor_hi_[2] != MPI_PROC_NULL)
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[4], send_buf_[5], n_xy, neighbor_hi_[2], stream);

    // ---- 3. Barrier: all incoming puts have completed and are visible. ----
    // Global barrier is safe — every PE participates (world_size == 1 already
    // returned above), so no deadlock.
    nvshmemx_barrier_all_on_stream(stream);

    // ---- 4. Unpack all six faces on the stream. ----
    LaunchUnpackAxisX(dst_ptrs, nfields, recv_buf_[0], recv_buf_[1], grid_, max_yz_, stream);
    LaunchUnpackAxisY(dst_ptrs, nfields, recv_buf_[2], recv_buf_[3], grid_, max_xz_, stream);
    LaunchUnpackAxisZ(dst_ptrs, nfields, recv_buf_[4], recv_buf_[5], grid_, max_xy_, stream);
}

void HaloExchangeQTensorNvshmem::PackSingleFieldForTest(
    const double* d_field, std::size_t field_idx, int axis, cudaStream_t stream) {
    HaloFieldPtrsConst src{};
    for (std::size_t i = 0; i < kMaxFields; ++i) src.p[i] = d_field;
    const int nfields = static_cast<int>(field_idx) + 1;

    switch (axis) {
        case 0: LaunchPackAxisX(src, nfields, send_buf_[0], send_buf_[1], grid_, max_yz_, stream); break;
        case 1: LaunchPackAxisY(src, nfields, send_buf_[2], send_buf_[3], grid_, max_xz_, stream); break;
        case 2: LaunchPackAxisZ(src, nfields, send_buf_[4], send_buf_[5], grid_, max_xy_, stream); break;
        default: throw std::runtime_error("PackSingleFieldForTest: axis out of range");
    }
}

// ===========================================================================
// VII-f — HaloExchangePassiveStressesNvshmem
// ===========================================================================

HaloExchangePassiveStressesNvshmem::HaloExchangePassiveStressesNvshmem(
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
    // HaloExchangeQTensorNvshmem sizing exactly (the two exchanges share
    // face geometry; only the field count differs).
    auto ceil_div = [](int global, int n) { return (global + n - 1) / n; };
    max_yz_ = static_cast<size_t>(ceil_div(Params::ny, mpi.dims[1]))
            * static_cast<size_t>(ceil_div(Params::nz, mpi.dims[2]));
    max_xz_ = static_cast<size_t>(ceil_div(Params::nx, mpi.dims[0]))
            * static_cast<size_t>(ceil_div(Params::nz, mpi.dims[2]));
    max_xy_ = static_cast<size_t>(ceil_div(Params::nx, mpi.dims[0]))
            * static_cast<size_t>(ceil_div(Params::ny, mpi.dims[1]));

    // Symmetric-heap allocation. Every PE allocates the same number of bytes
    // in the same order — NVSHMEM's symmetric-offset guarantee is that the
    // k-th nvshmem_malloc lands at the same virtual offset on every PE, and
    // the one-sided put uses exactly that offset. The QTensor halo class
    // allocates its buffers first (constructor ordering in ActiveNematicSim);
    // this class's k-th malloc is stable across PEs because every PE runs the
    // constructors in the same order.
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
                "HaloExchangePassiveStressesNvshmem: nvshmem_malloc failed for face buffer");
        }
        // Zero recv so an untouched slot (physical-wall face where no put
        // lands) does not surface last run's leftover into ghost cells.
        checkCudaErrors(cudaMemset(recv_buf_[f], 0, face_bytes[f]));
    }
}

HaloExchangePassiveStressesNvshmem::~HaloExchangePassiveStressesNvshmem() {
    for (int f = 0; f < 6; ++f) {
        if (send_buf_[f]) nvshmem_free(send_buf_[f]);
        if (recv_buf_[f]) nvshmem_free(recv_buf_[f]);
    }
}

std::size_t HaloExchangePassiveStressesNvshmem::face_area(int axis) const {
    switch (axis) {
        case 0: return max_yz_;
        case 1: return max_xz_;
        case 2: return max_xy_;
        default: return 0;
    }
}

void HaloExchangePassiveStressesNvshmem::ExchangePassiveStresses(
    DeviceFields& df, cudaStream_t stream) {
    // Single-rank fast path — matches ExchangeQTensor's early-return and keeps
    // the exchange out of the timeline entirely at nranks = 1.
    if (world_size_ == 1) return;

    HaloFieldPtrsConst src_ptrs = BuildPassiveStressFieldPtrs(df);
    HaloFieldPtrsMut   dst_ptrs = BuildPassiveStressFieldPtrsMut(df);
    const int nfields = static_cast<int>(kMaxFields);

    // ---- 1. Pack all six faces on the stream. ----
    LaunchPackAxisX(src_ptrs, nfields, send_buf_[0], send_buf_[1], grid_, max_yz_, stream);
    LaunchPackAxisY(src_ptrs, nfields, send_buf_[2], send_buf_[3], grid_, max_xz_, stream);
    LaunchPackAxisZ(src_ptrs, nfields, send_buf_[4], send_buf_[5], grid_, max_xy_, stream);

    // ---- 2. One-sided puts. ----
    // Same routing as ExchangeQTensor: rank r's lo face is neighbor_lo's hi
    // ghost (and vice versa). Skip on MPI_PROC_NULL (physical wall).
    const size_t n_yz = kMaxFields * max_yz_;
    const size_t n_xz = kMaxFields * max_xz_;
    const size_t n_xy = kMaxFields * max_xy_;

    if (neighbor_lo_[0] != MPI_PROC_NULL)
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[1], send_buf_[0], n_yz, neighbor_lo_[0], stream);
    if (neighbor_hi_[0] != MPI_PROC_NULL)
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[0], send_buf_[1], n_yz, neighbor_hi_[0], stream);
    if (neighbor_lo_[1] != MPI_PROC_NULL)
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[3], send_buf_[2], n_xz, neighbor_lo_[1], stream);
    if (neighbor_hi_[1] != MPI_PROC_NULL)
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[2], send_buf_[3], n_xz, neighbor_hi_[1], stream);
    if (neighbor_lo_[2] != MPI_PROC_NULL)
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[5], send_buf_[4], n_xy, neighbor_lo_[2], stream);
    if (neighbor_hi_[2] != MPI_PROC_NULL)
        nvshmemx_double_put_nbi_on_stream(
            recv_buf_[4], send_buf_[5], n_xy, neighbor_hi_[2], stream);

    // ---- 3. Barrier: all incoming puts have completed and are visible. ----
    nvshmemx_barrier_all_on_stream(stream);

    // ---- 4. Unpack all six faces on the stream. ----
    LaunchUnpackAxisX(dst_ptrs, nfields, recv_buf_[0], recv_buf_[1], grid_, max_yz_, stream);
    LaunchUnpackAxisY(dst_ptrs, nfields, recv_buf_[2], recv_buf_[3], grid_, max_xz_, stream);
    LaunchUnpackAxisZ(dst_ptrs, nfields, recv_buf_[4], recv_buf_[5], grid_, max_xy_, stream);
}

void HaloExchangePassiveStressesNvshmem::PackSingleFieldForTest(
    const double* d_field, std::size_t field_idx, int axis, cudaStream_t stream) {
    HaloFieldPtrsConst src{};
    for (std::size_t i = 0; i < kMaxFields; ++i) src.p[i] = d_field;
    const int nfields = static_cast<int>(field_idx) + 1;

    switch (axis) {
        case 0: LaunchPackAxisX(src, nfields, send_buf_[0], send_buf_[1], grid_, max_yz_, stream); break;
        case 1: LaunchPackAxisY(src, nfields, send_buf_[2], send_buf_[3], grid_, max_xz_, stream); break;
        case 2: LaunchPackAxisZ(src, nfields, send_buf_[4], send_buf_[5], grid_, max_xy_, stream); break;
        default: throw std::runtime_error("PackSingleFieldForTest: axis out of range");
    }
}

// ===========================================================================
// VII-g — HaloExchangeLbmNvshmem (face-only stage)
// ===========================================================================
//
// Design notes:
//   - Ghost -> owned direction (opposite of Q-tensor halo).
//   - Only the 5-direction crossing subset per face (Lattice::missing*).
//   - Skip-unpack at physical walls preserves local bounce values.
//   - `split_[d]==false` on unsplit axes turns pack/put/unpack into a no-op
//     (Plan A wraps unsplit periodic axes at streaming time, leaving the
//     ghost untouched; unpacking would overwrite valid owned values with
//     stale zeros).
//   - Edge and corner puts (needed for 2-D and 3-D splits) are a follow-up.
//     Face-only is correct for the 2-GPU {2,1,1} Poiseuille case.

namespace {

// Build a LbmFaceCrossings struct from a Lattice::missing* array and the
// two transverse velocity component arrays. Runs on the host at ctor time;
// the result is passed by value into pack/unpack kernels.
LbmFaceCrossings MakeFaceCrossings(const int (&dirs)[5],
                                   const int (&e_a)[Lattice::ndir],
                                   const int (&e_b)[Lattice::ndir]) {
    LbmFaceCrossings c{};
    for (int k = 0; k < 5; ++k) {
        c.dir[k]         = dirs[k];
        c.e_trans_a[k]   = e_a[dirs[k]];
        c.e_trans_b[k]   = e_b[dirs[k]];
    }
    return c;
}

// Build a per-axis skip context from is_wall[6] and grid geometry.
// For the X-face: transverse axes are Y (a) and Z (b).
LbmFaceSkipCtx MakeSkipCtxX(const std::array<bool, 6>& w, const LocalGrid& g) {
    return LbmFaceSkipCtx{
        /*wall_a_lo=*/w[2], /*wall_a_hi=*/w[3],
        /*wall_b_lo=*/w[4], /*wall_b_hi=*/w[5],
        g.offset_y, g.offset_z,
        Params::ny, Params::nz
    };
}
LbmFaceSkipCtx MakeSkipCtxY(const std::array<bool, 6>& w, const LocalGrid& g) {
    return LbmFaceSkipCtx{
        w[0], w[1],
        w[4], w[5],
        g.offset_x, g.offset_z,
        Params::nx, Params::nz
    };
}
LbmFaceSkipCtx MakeSkipCtxZ(const std::array<bool, 6>& w, const LocalGrid& g) {
    return LbmFaceSkipCtx{
        w[0], w[1],
        w[2], w[3],
        g.offset_x, g.offset_y,
        Params::nx, Params::ny
    };
}

}  // namespace

HaloExchangeLbmNvshmem::HaloExchangeLbmNvshmem(
    const LocalGrid& grid, const MPIContext& mpi,
    std::array<bool, 6> is_wall)
    : grid_(grid),
      world_size_(mpi.world_size),
      is_wall_(is_wall),
      max_yz_(0),
      max_xz_(0),
      max_xy_(0)
{
    for (int f = 0; f < 6; ++f) {
        send_buf_[f] = nullptr;
        recv_buf_[f] = nullptr;
    }

    // Cart neighbours along each axis; MPI_PROC_NULL where the axis is a
    // physical wall or unsplit non-periodic. MPI_Cart_shift handles both.
    for (int d = 0; d < 3; ++d) {
        int lo, hi;
        MPI_Cart_shift(mpi.cart_comm, d, 1, &lo, &hi);
        neighbor_[2 * d]     = lo;
        neighbor_[2 * d + 1] = hi;
        split_[d] = (mpi.dims[d] > 1);
    }

    // Max face area over all ranks. Deterministic from Params::n* and MPI
    // dims, so every PE computes the same value → symmetric heap alignment
    // is preserved when we allocate below.
    auto ceil_div = [](int global, int n) { return (global + n - 1) / n; };
    max_yz_ = static_cast<size_t>(ceil_div(Params::ny, mpi.dims[1]))
            * static_cast<size_t>(ceil_div(Params::nz, mpi.dims[2]));
    max_xz_ = static_cast<size_t>(ceil_div(Params::nx, mpi.dims[0]))
            * static_cast<size_t>(ceil_div(Params::nz, mpi.dims[2]));
    max_xy_ = static_cast<size_t>(ceil_div(Params::nx, mpi.dims[0]))
            * static_cast<size_t>(ceil_div(Params::ny, mpi.dims[1]));

    // Symmetric-heap allocation. Every PE allocates 6 face buffers of
    // identical size in identical order → the k-th nvshmem_malloc lands at
    // the same virtual offset on every PE, which is what makes
    // `recv_buf_[f]` valid as both a local pointer AND a remote destination
    // handle in the one-sided put below.
    const size_t bytes_yz = sizeof(double) * kCrossingDirs * max_yz_;
    const size_t bytes_xz = sizeof(double) * kCrossingDirs * max_xz_;
    const size_t bytes_xy = sizeof(double) * kCrossingDirs * max_xy_;
    const size_t face_bytes[6] = { bytes_yz, bytes_yz,
                                   bytes_xz, bytes_xz,
                                   bytes_xy, bytes_xy };

    for (int f = 0; f < 6; ++f) {
        send_buf_[f] = static_cast<double*>(nvshmem_malloc(face_bytes[f]));
        recv_buf_[f] = static_cast<double*>(nvshmem_malloc(face_bytes[f]));
        if (!send_buf_[f] || !recv_buf_[f]) {
            throw std::runtime_error(
                "HaloExchangeLbmNvshmem: nvshmem_malloc failed for face buffer");
        }
        checkCudaErrors(cudaMemset(recv_buf_[f], 0, face_bytes[f]));
    }
}

HaloExchangeLbmNvshmem::~HaloExchangeLbmNvshmem() {
    for (int f = 0; f < 6; ++f) {
        if (send_buf_[f]) nvshmem_free(send_buf_[f]);
        if (recv_buf_[f]) nvshmem_free(recv_buf_[f]);
    }
}

std::size_t HaloExchangeLbmNvshmem::face_area(int axis) const {
    switch (axis) {
        case 0: return max_yz_;
        case 1: return max_xz_;
        case 2: return max_xy_;
        default: return 0;
    }
}

void HaloExchangeLbmNvshmem::ExchangeLBM(DeviceFields& df, cudaStream_t stream) {
    // Single-rank fast path.
    if (world_size_ == 1) return;

    // Precompute the six face-crossings tables and three per-axis skip
    // contexts once per call. These are small PODs passed by value into
    // the kernels — cheap to build on the host.
    const LbmFaceCrossings lo_x = MakeFaceCrossings(Lattice::missingXHi,
                                                    Lattice::ey, Lattice::ez);
    const LbmFaceCrossings hi_x = MakeFaceCrossings(Lattice::missingXLo,
                                                    Lattice::ey, Lattice::ez);
    const LbmFaceCrossings lo_y = MakeFaceCrossings(Lattice::missingYHi,
                                                    Lattice::ex, Lattice::ez);
    const LbmFaceCrossings hi_y = MakeFaceCrossings(Lattice::missingYLo,
                                                    Lattice::ex, Lattice::ez);
    const LbmFaceCrossings lo_z = MakeFaceCrossings(Lattice::missingZHi,
                                                    Lattice::ex, Lattice::ey);
    const LbmFaceCrossings hi_z = MakeFaceCrossings(Lattice::missingZLo,
                                                    Lattice::ex, Lattice::ey);
    const LbmFaceSkipCtx skip_x = MakeSkipCtxX(is_wall_, grid_);
    const LbmFaceSkipCtx skip_y = MakeSkipCtxY(is_wall_, grid_);
    const LbmFaceSkipCtx skip_z = MakeSkipCtxZ(is_wall_, grid_);

    // ---- 1. Pack owned ghosts on split axes only. ----
    if (split_[0])
        LaunchPackLbmAxisX(df.d_f, send_buf_[0], send_buf_[1],
                           lo_x, hi_x, grid_, stream);
    if (split_[1])
        LaunchPackLbmAxisY(df.d_f, send_buf_[2], send_buf_[3],
                           lo_y, hi_y, grid_, stream);
    if (split_[2])
        LaunchPackLbmAxisZ(df.d_f, send_buf_[4], send_buf_[5],
                           lo_z, hi_z, grid_, stream);

    // ---- 2. One-sided puts. ----
    // Rank r's -X send goes to -X neighbour's +X recv slot: I packed my
    // -X ghost from cells (-1, y, z) — those pops originated at MY (0, y, z)
    // and are meant for -X neighbour's owned (local_nx-1, y, z), which is
    // its "+X owned boundary" = recv_buf_[+X] slot on the -X neighbour side.
    // Same crossed pattern on every axis (send_buf_[i] → recv_buf_[i XOR 1]).
    // MPI_PROC_NULL and unsplit axes are skipped: no put means recv_buf_
    // keeps its cudaMemset-zero, and the unpack is likewise gated below.
    const size_t n_yz = kCrossingDirs * max_yz_;
    const size_t n_xz = kCrossingDirs * max_xz_;
    const size_t n_xy = kCrossingDirs * max_xy_;

    if (split_[0]) {
        if (neighbor_[0] != MPI_PROC_NULL)
            nvshmemx_double_put_nbi_on_stream(
                recv_buf_[1], send_buf_[0], n_yz, neighbor_[0], stream);
        if (neighbor_[1] != MPI_PROC_NULL)
            nvshmemx_double_put_nbi_on_stream(
                recv_buf_[0], send_buf_[1], n_yz, neighbor_[1], stream);
    }
    if (split_[1]) {
        if (neighbor_[2] != MPI_PROC_NULL)
            nvshmemx_double_put_nbi_on_stream(
                recv_buf_[3], send_buf_[2], n_xz, neighbor_[2], stream);
        if (neighbor_[3] != MPI_PROC_NULL)
            nvshmemx_double_put_nbi_on_stream(
                recv_buf_[2], send_buf_[3], n_xz, neighbor_[3], stream);
    }
    if (split_[2]) {
        if (neighbor_[4] != MPI_PROC_NULL)
            nvshmemx_double_put_nbi_on_stream(
                recv_buf_[5], send_buf_[4], n_xy, neighbor_[4], stream);
        if (neighbor_[5] != MPI_PROC_NULL)
            nvshmemx_double_put_nbi_on_stream(
                recv_buf_[4], send_buf_[5], n_xy, neighbor_[5], stream);
    }

    // ---- 3. Barrier: every PE participates (world_size_ == 1 already
    //         returned). Guarantees all inbound puts have landed before
    //         unpack fires. ----
    nvshmemx_barrier_all_on_stream(stream);

    // ---- 4. Unpack recv buffers into owned boundary cells (with wall
    //         skip). Gated on split_[d] for the same reason as pack —
    //         Plan A means an unsplit axis's owned boundary is already
    //         correct from local streaming. ----
    if (split_[0])
        LaunchUnpackLbmAxisX(df.d_f, recv_buf_[0], recv_buf_[1],
                             MakeFaceCrossings(Lattice::missingXLo,
                                               Lattice::ey, Lattice::ez),
                             MakeFaceCrossings(Lattice::missingXHi,
                                               Lattice::ey, Lattice::ez),
                             skip_x, grid_, stream);
    if (split_[1])
        LaunchUnpackLbmAxisY(df.d_f, recv_buf_[2], recv_buf_[3],
                             MakeFaceCrossings(Lattice::missingYLo,
                                               Lattice::ex, Lattice::ez),
                             MakeFaceCrossings(Lattice::missingYHi,
                                               Lattice::ex, Lattice::ez),
                             skip_y, grid_, stream);
    if (split_[2])
        LaunchUnpackLbmAxisZ(df.d_f, recv_buf_[4], recv_buf_[5],
                             MakeFaceCrossings(Lattice::missingZLo,
                                               Lattice::ex, Lattice::ey),
                             MakeFaceCrossings(Lattice::missingZHi,
                                               Lattice::ex, Lattice::ey),
                             skip_z, grid_, stream);
}

// Keep the VII-a link-only stubs so anything already referencing them still links.
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
