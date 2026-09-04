#ifndef LBM_AN_DEVICE_FIELDS_H_
#define LBM_AN_DEVICE_FIELDS_H_

#include <string>
#include <params.h>
#include "mpi/mpi_context.h"
#include "local_grid.h"
#include "format_compat.h"

// Structured record of what backend init actually decided: device selection,
// per-node topology, NVSHMEM heap sizing, DRAM headroom. Written once by
// InitializeComputeBackend and read wherever code needs to reason about the
// backend without parsing a log string. GPU-only fields are zero-initialised
// on CPU builds; `is_gpu` and `is_nvshmem` are the gates.
struct BackendInfo {
    // MPI decomposition
    int world_size = 1;
    int dims[3]    = {1, 1, 1};

    // Per-node
    int local_rank = 0;
    int node_size  = 1;

    // Backend kind
    bool is_gpu     = false;
    bool is_nvshmem = false;

    // GPU device props (zeros on CPU builds)
    int device_id           = 0;
    int visible_gpus        = 0;
    int multiprocessors     = 0;
    int compute_major       = 0;
    int compute_minor       = 0;
    int async_engine_count  = 0;
    int can_map_host_memory = 0;
    std::size_t total_dram_bytes = 0;
    std::size_t free_dram_bytes  = 0;
    std::string device_name;

    // NVSHMEM heap accounting (zeros unless is_nvshmem)
    std::size_t symmetric_bytes = 0;
    std::size_t regular_bytes   = 0;

    // Backend allocator name: "nvshmem" (when is_nvshmem), "cuda" (GPU CPU builds),
    // or "" (CPU builds). Used by device_allocator.h to route halo-field allocations
    // through the appropriate backend.
    std::string allocator_name;
};

inline std::string FormatBackendSummary(const BackendInfo& info) {
    if (!info.is_gpu) {
        return compat::format(
            "CPU (OpenMP kNumOMPThreads = {}, MPI world_size = {}, dims = [{}, {}, {}])",
            Params::kNumOMPThreads, info.world_size,
            info.dims[0], info.dims[1], info.dims[2]);
    }
    constexpr double bytesPerMiB = 1024.0 * 1024.0;
    std::string s = compat::format(
        "GPU\n"
        "       using device: {}\n"
        "               name: {}\n"
        "    multiprocessors: {}\n"
        " compute capability: {}.{}\n"
        "      global memory: {:.1f} MiB\n"
        "        free memory: {:.1f} MiB\n"
        "   asyncEngineCount: {}\n"
        "   canMapHostMemory: {}\n"
        "    MPI world_size: {}\n"
        "         MPI dims: [{}, {}, {}]\n"
        "         node rank: {} / {}\n"
        "     visible GPUs: {}\n",
        info.device_id, info.device_name, info.multiprocessors,
        info.compute_major, info.compute_minor,
        info.total_dram_bytes / bytesPerMiB, info.free_dram_bytes / bytesPerMiB,
        info.async_engine_count, info.can_map_host_memory,
        info.world_size, info.dims[0], info.dims[1], info.dims[2],
        info.local_rank, info.node_size, info.visible_gpus);
    if (info.is_nvshmem) {
        s += compat::format(
            "  allocator: {}\n"
            "  symmetric heap: {:.1f} MiB (per PE)\n"
            "  regular device: {:.1f} MiB (this PE, halo-inclusive)\n",
            info.allocator_name,
            info.symmetric_bytes / bytesPerMiB,
            info.regular_bytes / bytesPerMiB);
    } else if (info.is_gpu) {
        s += compat::format("  allocator: {}\n", info.allocator_name);
    }
    return s;
}

#ifdef SIM_WITH_CUDA

#include <array>
#include <vector>
#include "qtensor_fields.h"
#include "fluid_fields.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

// Binds this MPI rank's CUDA device (VII-b: per-node local rank via
// MPI_COMM_TYPE_SHARED) and — under LBM_ENABLE_NVSHMEM (VII-c) — sizes the
// symmetric heap, initialises NVSHMEM from MPIContext::cart_comm, and asserts
// that nvshmem_my_pe() equals the cart_comm rank. Returns a BackendInfo
// describing the choices made; FormatBackendSummary turns that into a log line.
//
// Call once, before any CUDA allocation on this rank. In ActiveNematicSim the
// info is stored in a leading member so the initializer-list ordering forces
// backend init to run before DeviceFields' thrust allocations.
BackendInfo InitializeComputeBackend(const MPIContext& mpi, const LocalGrid& grid);

struct DeviceFields {
    // int gpu_id;
    // cudaStream_t stream;
    LocalGrid grid;
    std::size_t halo_volume;

    // Halo-exchanged fields allocated via backend allocator (VII-d):
    // nvshmem_malloc on NVSHMEM builds, cudaMalloc otherwise.
    double* d_f;
    double* d_f_new;
    double* d_qxx;
    double* d_qxy;
    double* d_qxz;
    double* d_qyy;
    double* d_qyz;
    double* d_qxx_new;
    double* d_qxy_new;
    double* d_qxz_new;
    double* d_qyy_new;
    double* d_qyz_new;
    double* d_Sigma_xx;
    double* d_Sigma_xy;
    double* d_Sigma_xz;
    double* d_Sigma_yy;
    double* d_Sigma_yz;
    double* d_Tau_xy;
    double* d_Tau_xz;
    double* d_Tau_yz;

    // Local-only fields stay on regular cudaMalloc (not exchanged across ranks).
    thrust::device_vector<double> d_rho, d_ux, d_uy, d_uz;
    thrust::device_vector<double> d_force_x, d_force_y, d_force_z;

    explicit DeviceFields(LocalGrid g = LocalGrid::SingleRank());

    ~DeviceFields();

    // ff is mutated transiently: ff.f_new is reused as scratch space for the
    // host->device layout transpose, then restored to its normal contents.
    // See device_fields.cu for why this is safe.
    void Initialize(FluidFields& ff, const QTensorFields& qf);

    void CopyToHost(FluidFields& ff, QTensorFields& qf) const;

};

#else

inline BackendInfo InitializeComputeBackend(const MPIContext& mpi, const LocalGrid&) {
    BackendInfo info;
    info.world_size = mpi.world_size;
    info.dims[0] = mpi.dims[0];
    info.dims[1] = mpi.dims[1];
    info.dims[2] = mpi.dims[2];
    info.is_gpu     = false;
    info.is_nvshmem = false;
    return info;
}

struct DeviceFields {
    explicit DeviceFields(LocalGrid) {}
};   // zero-size, optimized away entirely

#endif
#endif // LBM_AN_DEVICE_FIELDS_H_
