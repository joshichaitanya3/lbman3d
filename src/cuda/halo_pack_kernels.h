#ifndef LBM_AN_CUDA_HALO_PACK_KERNELS_H_
#define LBM_AN_CUDA_HALO_PACK_KERNELS_H_

// Transport-agnostic GPU pack/unpack interface for halo exchange.
//
// Any transport backend (NVSHMEM, CUDA-aware MPI, …) includes this header
// and calls the Launch* wrappers. The __global__ kernels and launch-geometry
// helpers are implementation details of halo_pack_kernels.cu with internal
// linkage — nothing here touches NVSHMEM or MPI.
//
// Slot layout (all axes):
//   buf[fi * face_stride + transverse_packed_idx] = field[halo_idx(bdy, ...)]
// where fi is the field index and face_stride = face_area in doubles.
// Matches the CPU HaloExchangeQTensor layout for side-by-side comparison.

#include <cuda_runtime.h>
#include <cstddef>

#include "local_grid.h"

// Upper bound on fields packed in one launch. Sized to the widest exchange
// (HaloExchangePassiveStressesNvshmem's 13 = 5 Q + 5 Σ + 3 τ). Narrower
// exchanges — HaloExchangeQTensorNvshmem's 8 — pass their own smaller
// nfields; the pointer bundle wastes ~40 B in kernel-parameter space, but the
// pack kernel launches only nfields field-blocks so no work is duplicated.
inline constexpr std::size_t kHaloMaxFields = 13;

// Field pointer bundles passed by value into kernels (~64 B, fits in
// registers). Callers set nfields <= kHaloMaxFields to limit active slots.
struct HaloFieldPtrsConst { const double* p[kHaloMaxFields]; };
struct HaloFieldPtrsMut   { double*       p[kHaloMaxFields]; };

// ---- X axis (YZ face, transverse: y × z) ----
void LaunchPackAxisX  (HaloFieldPtrsConst fields, int nfields,
                       double* out_lo, double* out_hi,
                       const LocalGrid& g, std::size_t face_stride,
                       cudaStream_t stream = 0);
void LaunchUnpackAxisX(HaloFieldPtrsMut fields, int nfields,
                       const double* in_lo, const double* in_hi,
                       const LocalGrid& g, std::size_t face_stride,
                       cudaStream_t stream = 0);

// ---- Y axis (XZ face, transverse: x × z) ----
void LaunchPackAxisY  (HaloFieldPtrsConst fields, int nfields,
                       double* out_lo, double* out_hi,
                       const LocalGrid& g, std::size_t face_stride,
                       cudaStream_t stream = 0);
void LaunchUnpackAxisY(HaloFieldPtrsMut fields, int nfields,
                       const double* in_lo, const double* in_hi,
                       const LocalGrid& g, std::size_t face_stride,
                       cudaStream_t stream = 0);

// ---- Z axis (XY face, transverse: x × y) ----
void LaunchPackAxisZ  (HaloFieldPtrsConst fields, int nfields,
                       double* out_lo, double* out_hi,
                       const LocalGrid& g, std::size_t face_stride,
                       cudaStream_t stream = 0);
void LaunchUnpackAxisZ(HaloFieldPtrsMut fields, int nfields,
                       const double* in_lo, const double* in_hi,
                       const LocalGrid& g, std::size_t face_stride,
                       cudaStream_t stream = 0);

// ─────────────────────────────────────────────────────────────────────────────
// LBM face pack/unpack (PR VII-g, face-only stage)
// ─────────────────────────────────────────────────────────────────────────────
//
// Distinct from the Q-tensor pack API above because:
//   - Direction is reversed: pack reads from GHOST, unpack writes to OWNED.
//   - Only 5 crossing dirs per face (not all 15 populations).
//   - Unpack needs a per-face wall-skip predicate (invariant 3, see
//     src/mpi/CLAUDE.md "Post-stream LBM exchange").
//
// One kernel per axis handles both lo and hi faces in a single launch.

// Crossings for one face: the 5 dirs in Lattice::missing{X,Y,Z}{Lo,Hi} plus
// each dir's two transverse velocity components (needed by unpack's wall
// skip). By axis:
//   X-face   → e_trans_a = ey,  e_trans_b = ez
//   Y-face   → e_trans_a = ex,  e_trans_b = ez
//   Z-face   → e_trans_a = ex,  e_trans_b = ey
struct LbmFaceCrossings {
    int dir[5];
    int e_trans_a[5];
    int e_trans_b[5];
};

// Skip predicate context: at an owned-boundary cell (0, y, z) or
// (local_n-1, y, z) on an axis face, skip the unpack when the cell is
// on an orthogonal physical wall AND the crossing dir would flow into
// that wall (bounce already wrote it locally).
//
// For X-face: transverse axes are (Y, Z), so
//   wall_a_lo/hi = is_wall[YLo]/[YHi], global_n_a = Params::ny, etc.
struct LbmFaceSkipCtx {
    bool wall_a_lo;
    bool wall_a_hi;
    bool wall_b_lo;
    bool wall_b_hi;
    int  offset_a;
    int  offset_b;
    int  global_n_a;
    int  global_n_b;
};

// ---- X axis: pack (from ±X ghost) and unpack (into ±X owned boundary) ----
void LaunchPackLbmAxisX  (const double* d_f,
                          double* send_lo, double* send_hi,
                          LbmFaceCrossings lo_face, LbmFaceCrossings hi_face,
                          const LocalGrid& g, cudaStream_t stream = 0);
void LaunchUnpackLbmAxisX(double* d_f,
                          const double* recv_lo, const double* recv_hi,
                          LbmFaceCrossings lo_face, LbmFaceCrossings hi_face,
                          LbmFaceSkipCtx skip,
                          const LocalGrid& g, cudaStream_t stream = 0);

// ---- Y axis ----
void LaunchPackLbmAxisY  (const double* d_f,
                          double* send_lo, double* send_hi,
                          LbmFaceCrossings lo_face, LbmFaceCrossings hi_face,
                          const LocalGrid& g, cudaStream_t stream = 0);
void LaunchUnpackLbmAxisY(double* d_f,
                          const double* recv_lo, const double* recv_hi,
                          LbmFaceCrossings lo_face, LbmFaceCrossings hi_face,
                          LbmFaceSkipCtx skip,
                          const LocalGrid& g, cudaStream_t stream = 0);

// ---- Z axis ----
void LaunchPackLbmAxisZ  (const double* d_f,
                          double* send_lo, double* send_hi,
                          LbmFaceCrossings lo_face, LbmFaceCrossings hi_face,
                          const LocalGrid& g, cudaStream_t stream = 0);
void LaunchUnpackLbmAxisZ(double* d_f,
                          const double* recv_lo, const double* recv_hi,
                          LbmFaceCrossings lo_face, LbmFaceCrossings hi_face,
                          LbmFaceSkipCtx skip,
                          const LocalGrid& g, cudaStream_t stream = 0);

#endif  // LBM_AN_CUDA_HALO_PACK_KERNELS_H_
