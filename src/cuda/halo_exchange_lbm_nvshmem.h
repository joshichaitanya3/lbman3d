#ifndef LBM_AN_CUDA_HALO_EXCHANGE_LBM_NVSHMEM_H_
#define LBM_AN_CUDA_HALO_EXCHANGE_LBM_NVSHMEM_H_

#include "local_grid.h"
#include "mpi/mpi_context.h"

#ifdef LBM_ENABLE_NVSHMEM

#include <array>
#include <cstddef>
#include <cuda_runtime.h>

struct DeviceFields;   // fwd decl (device_fields.h is CUDA-heavy)

// LBM post-stream halo exchange on NVSHMEM (PR VII-g, face-only stage).
//
// Direction (opposite of Q-tensor halo):
//   push streaming has already deposited outgoing crossings into MY ghost
//   layer. This exchange packs those ghost values, one-sided-puts them into
//   the NEIGHBOUR's owned-boundary recv buffer, barriers, then the neighbour
//   unpacks into its own owned boundary — where its next-step collision
//   reads them. Ghost -> owned, not owned -> ghost.
//
// Only the 5-direction crossing subset per face is exchanged (D3Q15), from
// Lattice::missing{X,Y,Z}{Lo,Hi}. Sending all 15 dirs would stomp owned-
// boundary slots at non-crossing dirs that local streaming filled correctly.
//
// Three tiers exist in D3Q15 because dirs 7-14 have all three components
// nonzero — a source cell at a subdomain edge or corner deposits into edge
// or corner ghosts, not just face ghosts. This face-only implementation
// handles single-axis crossings only (dirs 1-6 always, dirs 7-14 whose
// destination stays on the face plane). It is CORRECT for 1-D splits
// (e.g. dims={2,1,1} — Poiseuille) under Plan A: on unsplit periodic axes
// the streaming kernel wraps locally, so no edge/corner ghost gets written
// and there is nothing extra to route. 2-D and 3-D splits need edge and
// corner puts respectively — same pack/put/barrier/unpack pattern, just
// more geometry — added in a follow-up increment.
//
// Skip-unpack at physical walls (invariant 3 from src/mpi/CLAUDE.md):
//   at a seam-touching cell that also sits on an orthogonal physical wall,
//   the local `HandleBoundaryPoint` already wrote the correct bounce value
//   into a crossing-dir slot at the owned boundary. The neighbour's ghost
//   at that dir was never written by push streaming (its source cell sits
//   below the wall). Unpacking anyway would clobber the valid bounce with
//   zero → immediate mass loss. `is_wall` at construction gates this per
//   face; the unpack kernel skips the write when the cell hits an
//   orthogonal wall and the dir would flow into it.
//
// Buffers live on the NVSHMEM symmetric heap. Sized to max face area over
// all ranks × 5 crossing dirs. Same symmetric-offset alignment story as
// VII-e/f: identical allocation order + identical sizes on every PE → the
// k-th nvshmem_malloc lands at the same virtual offset, so `recv_buf_[f]`
// serves as both my local pointer AND the remote destination handle when
// I put to a neighbour PE.
struct HaloExchangeLbmNvshmem {
    LocalGrid grid_;
    int       world_size_;

    // Cart neighbours by face (0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z).
    // MPI_PROC_NULL on physical-wall axes at the domain boundary. Under
    // Plan A, an unsplit periodic axis (dims[d]==1) has `dims[d]==1` and
    // MPI_Cart_shift returns MPI_PROC_NULL as well for those hops (no
    // neighbour to shift to on that axis); we double-gate on `split_[d]`
    // below anyway so the exchange stays a strict no-op there.
    int  neighbor_[6];

    // Which axes are actually split (dims[d] > 1). Plan A wraps unsplit
    // periodic axes at streaming time, so their ghost layer is never
    // written and must never be unpacked — otherwise we clobber owned
    // values with stale zeros.
    bool split_[3];

    // Per-face wall flag from BC template (is_wall_by_face<BC>).
    // Used by the unpack kernel's skip predicate.
    std::array<bool, 6> is_wall_;

    // 5 crossing dirs per face.
    static constexpr int kCrossingDirs = 5;

    // Max face area over all ranks (deterministic from Params::n* and
    // MPI dims — every PE computes the same value with no collective).
    std::size_t max_yz_;
    std::size_t max_xz_;
    std::size_t max_xy_;

    // Symmetric-heap allocations. Face indices:
    //   0 = -X, 1 = +X, 2 = -Y, 3 = +Y, 4 = -Z, 5 = +Z
    // send_buf_[f] is packed by MY pack kernel; the put targets the
    // neighbour PE's recv_buf_ at the OPPOSITE face index (so my
    // send_buf_[+X] arrives in neighbour's recv_buf_[-X]).
    double* send_buf_[6];
    double* recv_buf_[6];

    HaloExchangeLbmNvshmem() = default;
    HaloExchangeLbmNvshmem(const LocalGrid& grid, const MPIContext& mpi,
                           std::array<bool, 6> is_wall = {});
    ~HaloExchangeLbmNvshmem();

    HaloExchangeLbmNvshmem(const HaloExchangeLbmNvshmem&)            = delete;
    HaloExchangeLbmNvshmem& operator=(const HaloExchangeLbmNvshmem&) = delete;

    // One-shot exchange after GpuCollideAndStream. Enqueues pack kernels,
    // one-sided puts, a stream-scoped NVSHMEM barrier, and unpack kernels
    // — all on the caller-supplied stream — so the whole thing composes
    // with the existing async overlap. On return, `df.d_f`'s owned
    // boundary cells hold the neighbours' post-stream crossings at the
    // matching dir slots.
    void ExchangeLBM(DeviceFields& df, cudaStream_t stream = 0);

    // Face area for a given axis (in doubles per crossing-dir slot).
    std::size_t face_area(int axis) const;
};

#else  // !LBM_ENABLE_NVSHMEM

// Zero-cost stub so ActiveNematicSim can hold a member unconditionally
// under SIM_WITH_CUDA. All calls compile to no-ops; non-NVSHMEM GPU
// builds fall through to the single-rank device path exactly as before.
struct DeviceFields;
struct HaloExchangeLbmNvshmem {
    HaloExchangeLbmNvshmem() = default;
    HaloExchangeLbmNvshmem(const LocalGrid&, const MPIContext&,
                           std::array<bool, 6> = {}) {}
    void ExchangeLBM(DeviceFields&, int = 0) {}
};

#endif

#endif  // LBM_AN_CUDA_HALO_EXCHANGE_LBM_NVSHMEM_H_
