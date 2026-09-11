#ifndef LBM_AN_CUDA_HALO_EXCHANGE_PASSIVE_STRESSES_NVSHMEM_H_
#define LBM_AN_CUDA_HALO_EXCHANGE_PASSIVE_STRESSES_NVSHMEM_H_

#include "local_grid.h"
#include "mpi/mpi_context.h"

#ifdef LBM_ENABLE_NVSHMEM

#include <cstddef>
#include <cuda_runtime.h>
#include "halo_pack_kernels.h"

struct DeviceFields;   // fwd decl (device_fields.h is CUDA-heavy)

// Passive-stress halo exchange on NVSHMEM (PR VII-f).
//
// Runs between phase 1 (GpuQTensorStep — Q update + Σ/τ write) and phase 2
// (GpuComputeBodyForce — reads Q, Σ, τ at neighbours to form div(Σ + τ) and
// the active-stress divergence). Direction and physics semantics match the
// CPU HaloExchangeQTensor::ExchangePassiveStresses: each rank packs its OWNED
// boundary planes for 5 Q + 5 symmetric-stress Σ + 3 antisymmetric-stress τ =
// 13 scalar fields, one-sided puts them into the neighbour's ghost recv
// buffer, barriers, then unpacks into its own ghost cells.
//
// Why re-exchange Q here even though ExchangeQTensor already ran at the start
// of the step: phase 1 wrote a fresh Q_new pointwise, so the ghost cells
// filled by ExchangeQTensor now hold the pre-step Q. Phase 2's Q stencils
// (used by the active-stress div(αQ) and the Ericksen distortion term
// bookkeeping) read updated Q at neighbours, so Q ghosts must be refreshed.
// τ is antisymmetric-carrying and also lives in per-cell fields, so it needs
// its own halo just like Σ.
//
// Structurally identical to HaloExchangeQTensorNvshmem — separate class only
// so the symmetric-heap allocation reports the honest per-exchange footprint
// (kMaxFields = 13 here vs. 8 there). Pack/unpack kernels are shared verbatim
// via halo_pack_kernels.h.
//
// Portability seam: pack/unpack kernels take plain double* pointers (no
// NVSHMEM assumptions). Only ExchangePassiveStresses itself calls NVSHMEM.
// See src/cuda/CLAUDE.md "Portability goal: keep the transport swappable".
struct HaloExchangePassiveStressesNvshmem {
    LocalGrid grid_;
    int       world_size_;
    // NVSHMEM PE IDs equal cart_comm ranks (VII-c invariant); neighbour_lo/hi_[d]
    // are the direct put targets. MPI_PROC_NULL on physical-wall axes; put is
    // skipped there — the ghost value is resolved by boundary_handler at read time.
    int       neighbor_lo_[3];
    int       neighbor_hi_[3];

    // 5 Q + 5 Σ + 3 τ.
    static constexpr std::size_t kMaxFields = 13;
    static_assert(kMaxFields <= kHaloMaxFields,
                  "kHaloMaxFields must cover this class's field count; bump the "
                  "constant in halo_pack_kernels.h if a wider exchange lands.");

    // Face-area upper bounds (over all ranks) for buffer sizing. Face indices:
    //   0 = lo-x, 1 = hi-x   (buffers sized max_yz_ * kMaxFields)
    //   2 = lo-y, 3 = hi-y   (buffers sized max_xz_ * kMaxFields)
    //   4 = lo-z, 5 = hi-z   (buffers sized max_xy_ * kMaxFields)
    std::size_t max_yz_;
    std::size_t max_xz_;
    std::size_t max_xy_;

    // Symmetric-heap allocations. See HaloExchangeQTensorNvshmem for the
    // symmetric-heap offset argument — allocated in a fixed order on every PE
    // so each nvshmem_malloc lands at the same offset on every PE, letting the
    // local pointer serve as the remote put destination.
    double* send_buf_[6];
    double* recv_buf_[6];

    HaloExchangePassiveStressesNvshmem() = default;
    HaloExchangePassiveStressesNvshmem(const LocalGrid& grid, const MPIContext& mpi);
    ~HaloExchangePassiveStressesNvshmem();

    HaloExchangePassiveStressesNvshmem(const HaloExchangePassiveStressesNvshmem&)            = delete;
    HaloExchangePassiveStressesNvshmem& operator=(const HaloExchangePassiveStressesNvshmem&) = delete;

    // Fill ghost cells of d_qxx..d_qyz and d_Sigma_*, d_Tau_* with the
    // neighbour's owned values. Enqueued on `stream`; the barrier at the end
    // ensures all incoming puts have landed before unpack fires, so the caller
    // can immediately launch GpuComputeBodyForce on the same stream.
    void ExchangePassiveStresses(DeviceFields& df, cudaStream_t stream = 0);

    // Test-only entry point. Packs one scalar field into send_buf_[2*d] (lo
    // face) and send_buf_[2*d+1] (hi face) at slot `field_idx`, no put, no
    // barrier. Used by the pack unit test to inspect buffer contents at
    // nranks = 1.
    void PackSingleFieldForTest(const double* d_field, std::size_t field_idx,
                                int axis, cudaStream_t stream = 0);

    // Face area for a given axis, in doubles per field.
    std::size_t face_area(int axis) const;
};

#else  // !LBM_ENABLE_NVSHMEM

// Zero-cost stub so ActiveNematicSim can hold a member unconditionally under
// SIM_WITH_CUDA. All calls compile to no-ops; non-NVSHMEM GPU builds fall
// through to the single-rank device-solver path exactly as before.
struct DeviceFields;
struct HaloExchangePassiveStressesNvshmem {
    HaloExchangePassiveStressesNvshmem() = default;
    HaloExchangePassiveStressesNvshmem(const LocalGrid&, const MPIContext&) {}
    void ExchangePassiveStresses(DeviceFields&, int = 0) {}
};

#endif

#endif  // LBM_AN_CUDA_HALO_EXCHANGE_PASSIVE_STRESSES_NVSHMEM_H_
