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

#endif  // LBM_AN_CUDA_HALO_PACK_KERNELS_H_
