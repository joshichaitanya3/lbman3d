#ifndef LBM_AN_CUDA_HALO_EXCHANGE_QTENSOR_NVSHMEM_H_
#define LBM_AN_CUDA_HALO_EXCHANGE_QTENSOR_NVSHMEM_H_

#include "local_grid.h"
#include "mpi/mpi_context.h"

#ifdef LBM_ENABLE_NVSHMEM

#include <cstddef>
#include <cuda_runtime.h>

struct DeviceFields;   // fwd decl (device_fields.h is CUDA-heavy)

// Q-tensor + velocity halo exchange on NVSHMEM (PR VII-e).
//
// Direction and physics semantics match the CPU HaloExchangeQTensor::ExchangeQTensor:
// each rank packs its OWNED boundary planes for 5 Q + 3 velocity fields, one-sided
// puts them into the neighbour's ghost recv buffer, barriers, then unpacks into its
// own ghost cells. Owned -> ghost, star-stencil, face-only (no edges/corners — those
// are a D3Q15-population concern that lives in VII-g's ExchangeLBM).
//
// Buffers live on the NVSHMEM symmetric heap. Sized to the max face area across all
// ranks (identical on every PE — MPI_Dims_create + Params::nx/ny/nz are collective),
// so every PE allocates the same number of bytes and the heap stays symmetric.
//
// Portability seam: pack/unpack kernels take plain double* pointers (no NVSHMEM
// assumptions). Only ExchangeQTensor itself calls NVSHMEM. See
// src/cuda/CLAUDE.md "Portability goal: keep the transport swappable".
struct HaloExchangeQTensorNvshmem {
    LocalGrid grid_;
    int       world_size_;
    // NVSHMEM PE IDs equal cart_comm ranks (VII-c invariant); neighbour_lo/hi_[d]
    // are the direct put targets. MPI_PROC_NULL on physical-wall axes; put is
    // skipped there — the ghost value is resolved by boundary_handler at read time.
    int       neighbor_lo_[3];
    int       neighbor_hi_[3];

    // Field list = 5 Q + 3 velocity. Kept aligned with the CPU exchange's 8-field
    // pack (halo_exchange_qtensor.h::ExchangeQTensor). VII-f will introduce a
    // separate class for passive stresses; keeping this class 8-field-sized keeps
    // the symmetric-heap footprint per PR minimal and honest.
    static constexpr std::size_t kMaxFields = 8;

    // Face-area upper bounds (over all ranks) for buffer sizing. Face indices:
    //   0 = lo-x, 1 = hi-x   (buffers sized max_yz_ * kMaxFields)
    //   2 = lo-y, 3 = hi-y   (buffers sized max_xz_ * kMaxFields)
    //   4 = lo-z, 5 = hi-z   (buffers sized max_xy_ * kMaxFields)
    std::size_t max_yz_;
    std::size_t max_xz_;
    std::size_t max_xy_;

    // Symmetric-heap allocations. send_buf_[f] is packed on this PE; recv_buf_[f]
    // is written by the paired put from the neighbour PE. Neighbour picks the same
    // symmetric-heap offset (same allocation index on every PE, by NVSHMEM
    // definition), so the local pointer serves as the remote destination handle.
    double* send_buf_[6];
    double* recv_buf_[6];

    HaloExchangeQTensorNvshmem() = default;
    HaloExchangeQTensorNvshmem(const LocalGrid& grid, const MPIContext& mpi);
    ~HaloExchangeQTensorNvshmem();

    HaloExchangeQTensorNvshmem(const HaloExchangeQTensorNvshmem&)            = delete;
    HaloExchangeQTensorNvshmem& operator=(const HaloExchangeQTensorNvshmem&) = delete;

    // Fill ghost cells of d_qxx..d_qyz and d_ux/uy/uz with neighbours' owned
    // values. Enqueued on `stream`; the barrier at the end ensures all incoming
    // puts have landed before unpack fires, so the caller can immediately launch
    // the Q update kernel on the same stream.
    void ExchangeQTensor(DeviceFields& df, cudaStream_t stream = 0);

    // Test-only entry point. Packs one scalar field into send_buf_[2*d] (lo face)
    // and send_buf_[2*d+1] (hi face) at slot `field_idx`, no put, no barrier.
    // Used by the pack unit test to inspect buffer contents at nranks = 1.
    void PackSingleFieldForTest(const double* d_field, std::size_t field_idx,
                                int axis, cudaStream_t stream = 0);

    // Face area for a given axis, in doubles per field.
    std::size_t face_area(int axis) const;
};

#else  // !LBM_ENABLE_NVSHMEM

// Zero-cost stub so ActiveNematicSim can hold a member unconditionally under
// SIM_WITH_CUDA. All calls compile to no-ops; non-NVSHMEM GPU builds fall through
// to the single-rank device-solver path exactly as before.
struct DeviceFields;
struct HaloExchangeQTensorNvshmem {
    HaloExchangeQTensorNvshmem() = default;
    HaloExchangeQTensorNvshmem(const LocalGrid&, const MPIContext&) {}
    void ExchangeQTensor(DeviceFields&, int = 0) {}
};

#endif

#endif  // LBM_AN_CUDA_HALO_EXCHANGE_QTENSOR_NVSHMEM_H_
