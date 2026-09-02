#include "device_fields.h"
#include <params.h>
#include "cuda_utils.h"
#include "lattice_stencil.h"
#include "physics_helpers.h"
#include <algorithm>
#include <cstdlib>
#include "local_grid.h"

#ifdef LBM_ENABLE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

BackendInfo InitializeComputeBackend(const MPIContext& mpi, const LocalGrid& grid) {
    BackendInfo info;
    info.is_gpu     = true;
    info.world_size = mpi.world_size;
    info.dims[0] = mpi.dims[0];
    info.dims[1] = mpi.dims[1];
    info.dims[2] = mpi.dims[2];

    // Per-node local rank derivation. NVSHMEM binds to whichever device is
    // current at nvshmemx_init time — hardcoding device 0 pins every rank on
    // a node to the same GPU and produces no error message, just catastrophic
    // contention. Splitting MPI_COMM_WORLD by MPI_COMM_TYPE_SHARED yields a
    // communicator per shared-memory domain (one per node in typical
    // launches); rank inside it is the intra-node ordinal we bind to.
#ifdef LBM_ENABLE_MPI
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi.world_rank,
                        MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &info.local_rank);
    MPI_Comm_size(node_comm, &info.node_size);
    MPI_Comm_free(&node_comm);
#endif

    checkCudaErrors(cudaGetDeviceCount(&info.visible_gpus));
    if (info.visible_gpus <= 0) {
        throw std::runtime_error("No CUDA devices visible to this rank");
    }
    // Modulo maps two common launch patterns cleanly:
    //   - launcher exposes all node GPUs to every rank (visible_gpus == GPUs
    //     per node): each local_rank picks a distinct device up to visible_gpus,
    //     then wraps (oversubscription — visible in the log, not silent).
    //   - launcher sets CUDA_VISIBLE_DEVICES per rank (visible_gpus == 1):
    //     every rank maps to device 0, which is the only device it sees.
    const int assigned_device = info.local_rank % info.visible_gpus;
    checkCudaErrors(cudaSetDevice(assigned_device));
    checkCudaErrors(cudaGetDevice(&info.device_id));

    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, info.device_id));
    info.device_name         = props.name;
    info.multiprocessors     = props.multiProcessorCount;
    info.compute_major       = props.major;
    info.compute_minor       = props.minor;
    info.async_engine_count  = props.asyncEngineCount;
    info.can_map_host_memory = props.canMapHostMemory;
    info.total_dram_bytes    = props.totalGlobalMem;

    size_t free_mem = 0, total_mem = 0;
    checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem));
    info.free_dram_bytes = free_mem;

    // NVSHMEM bootstrap.
    //
    // Runs strictly after cudaSetDevice: NVSHMEM binds to the currently-
    // selected device at init time. Runs strictly before any halo-exchanged
    // device allocation (see ActiveNematicSim member ordering): NVSHMEM reserves
    // the symmetric heap up front, so VII-d's nvshmem_malloc calls come out of
    // that reservation rather than growing DRAM behind our back.
    //
    // Sizes the symmetric heap for the eventual VII-d field split (halo-
    // exchanged fields = 43 doubles/cell: 2 × 15 populations + 5 Q + 5 Σ + 3 τ,
    // per the table in src/cuda/CLAUDE.md). Even though VII-c does not yet
    // route those allocations through nvshmem_malloc, the heap must be sized
    // now — a mid-run resize is not an option under NVSHMEM.
#ifdef LBM_ENABLE_NVSHMEM
    info.is_nvshmem = true;
    constexpr int kSymmetricDoublesPerCell = 2 * Lattice::ndir + 5 + 5 + 3;
    constexpr int kRegularDoublesPerCell   = 4 + 3 + 5;  // ρ, u{x,y,z}, force, Q_new
    // NVSHMEM requires the symmetric heap to be the same size on every PE.
    // Uneven splits give rank (0,0,0) the ceil, but Allreduce-max is the
    // canonical way to make every PE agree without reasoning about which
    // coord got the ceil.
    long local_halo = grid.HaloVolume();
    long max_halo = local_halo;
    MPI_Allreduce(&local_halo, &max_halo, 1, MPI_LONG, MPI_MAX, mpi.cart_comm);
    constexpr size_t kSymmetricSlackBytes = 64ULL << 20;  // pack buffers + headroom
    info.symmetric_bytes =
        static_cast<size_t>(max_halo) * kSymmetricDoublesPerCell * sizeof(double)
        + kSymmetricSlackBytes;
    info.regular_bytes =
        static_cast<size_t>(local_halo) * kRegularDoublesPerCell * sizeof(double);
    constexpr double bytesPerMiB = 1024.0 * 1024.0;
    if (info.symmetric_bytes + info.regular_bytes >= free_mem) {
        // Min-rank guidance: symmetric grows with max local volume, so more
        // ranks along the largest global axis shrinks it linearly.
        throw std::runtime_error(std::format(
            "GPU DRAM check failed on device {}: need {:.1f} MiB "
            "(symmetric heap {:.1f} + regular {:.1f}) but only {:.1f} MiB free. "
            "Reduce grid size or add ranks along the largest axis.",
            info.device_id,
            (info.symmetric_bytes + info.regular_bytes) / bytesPerMiB,
            info.symmetric_bytes / bytesPerMiB,
            info.regular_bytes / bytesPerMiB,
            free_mem / bytesPerMiB));
    }
    // NVSHMEM reads NVSHMEM_SYMMETRIC_SIZE from the environment at init time.
    // Setting it here means the launcher does not have to know per-run how big
    // the local subdomain is — the code that already knows sets it.
    {
        std::string sym_size_str = std::to_string(info.symmetric_bytes);
        setenv("NVSHMEM_SYMMETRIC_SIZE", sym_size_str.c_str(), /*overwrite=*/1);
    }

    // NVSHMEMX_INIT_WITH_MPI_COMM inherits the MPI cart topology exactly:
    // NVSHMEM PE ids equal cart_comm ranks. This is a design invariant
    // (src/cuda/CLAUDE.md → "Design invariants") — MPI_Cart_shift addresses
    // are the addresses NVSHMEM puts target, no parallel PE↔rank map.
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    MPI_Comm comm = mpi.cart_comm;
    nvshmemx_set_attr_mpi_comm_args(&comm, &attr);
    if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr) != 0) {
        throw std::runtime_error("nvshmemx_init_attr failed");
    }

    // Sanity check: if the invariant above breaks, halo puts silently target
    // the wrong PE and simulation results are garbage. Catch it here, before
    // any exchange runs.
    const int my_pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();
    if (my_pe != mpi.world_rank || n_pes != mpi.world_size) {
        throw std::runtime_error(std::format(
            "NVSHMEM bootstrap invariant broken: PE {}/{} != cart rank {}/{}",
            my_pe, n_pes, mpi.world_rank, mpi.world_size));
    }
#else
    (void)grid;  // symmetric-heap sizing needs the grid; non-NVSHMEM does not
#endif

    return info;
}

// HaloVolume() (not Volume()) so device buffers include the ghost layer that
// PR VII's cross-rank exchange will read/write. Identical to Volume() at
// single rank (kHaloMPI == 0), where every kernel index g.halo_idx(...) still
// lands inside [0, HaloVolume) — the change is a no-op today and correct
// for the MPI+GPU path without further edits.
DeviceFields::DeviceFields(LocalGrid g) :
    grid      (g),
    d_f       (g.HaloVolume() * Lattice::ndir, 0.0),
    d_f_new   (g.HaloVolume() * Lattice::ndir, 0.0),
    d_rho     (g.HaloVolume(), Params::kDensity),
    d_ux      (g.HaloVolume(), 0.0),
    d_uy      (g.HaloVolume(), 0.0),
    d_uz      (g.HaloVolume(), 0.0),
    d_force_x (g.HaloVolume(), 0.0),
    d_force_y (g.HaloVolume(), 0.0),
    d_force_z (g.HaloVolume(), 0.0),
    d_qxx     (g.HaloVolume(), 0.0),
    d_qxx_new (g.HaloVolume(), 0.0),
    d_qxy     (g.HaloVolume(), 0.0),
    d_qxy_new (g.HaloVolume(), 0.0),
    d_qxz     (g.HaloVolume(), 0.0),
    d_qxz_new (g.HaloVolume(), 0.0),
    d_qyy     (g.HaloVolume(), 0.0),
    d_qyy_new (g.HaloVolume(), 0.0),
    d_qyz     (g.HaloVolume(), 0.0),
    d_qyz_new (g.HaloVolume(), 0.0),
    d_Sigma_xx     (g.HaloVolume(), 0.0),
    d_Sigma_xy     (g.HaloVolume(), 0.0),
    d_Sigma_xz     (g.HaloVolume(), 0.0),
    d_Sigma_yy     (g.HaloVolume(), 0.0),
    d_Sigma_yz     (g.HaloVolume(), 0.0),
    d_Tau_xy     (g.HaloVolume(), 0.0),
    d_Tau_xz     (g.HaloVolume(), 0.0),
    d_Tau_yz     (g.HaloVolume(), 0.0)

{}

void DeviceFields::Initialize(FluidFields& ff, const QTensorFields& qf) {

    // Device idx(x,y,z,i) has i slowest-varying (host has i fastest), so
    // ff.f can't be copied to d_f directly — transpose it first. ff.f_new is
    // reused as scratch for the transposed layout rather than allocating a
    // new buffer: it's only ever meaningful as LbmSolver::LatticeBoltzmannStep's
    // double-buffer swap target, and that never runs under SIM_WITH_CUDA
    // (Step() calls d_solver_ instead of lbm_) — so it's safe to clobber
    // here regardless of whether ff.f itself is ever synced from d_f for
    // debugging later. Restored to ff.f's contents afterward purely so it
    // doesn't look like corrupted data to anything that inspects it later.
    const LocalGrid& g = ff.grid;
    // n is the per-direction stride in the device (i-slowest) layout — must
    // match d_f's allocation size, so HaloVolume, not Volume.
    const int n = g.HaloVolume();
    for (int z = 0; z < g.local_nz; ++z)
        for (int y = 0; y < g.local_ny; ++y)
            for (int x = 0; x < g.local_nx; ++x)
                for (int i = 0; i < Lattice::ndir; ++i)
                    ff.f_new[i * n + g.halo_idx(x, y, z)] = ff.f[g.halo_idx(x, y, z, i)];

    thrust::copy(ff.f_new.begin(), ff.f_new.end(), d_f.begin());
    std::copy(ff.f.begin(), ff.f.end(), ff.f_new.begin());

    // Copy Force fields initialized on the host to the device
    thrust::copy(ff.fx.begin(),  ff.fx.end(),  d_force_x.begin());
    thrust::copy(ff.fy.begin(),  ff.fy.end(),  d_force_y.begin());
    thrust::copy(ff.fz.begin(),  ff.fz.end(),  d_force_z.begin());

    // Copy QTensor fields initialized on the host to the device
    thrust::copy(qf.qxx.begin(),  qf.qxx.end(),  d_qxx.begin());
    thrust::copy(qf.qxy.begin(),  qf.qxy.end(),  d_qxy.begin());
    thrust::copy(qf.qxz.begin(),  qf.qxz.end(),  d_qxz.begin());
    thrust::copy(qf.qyy.begin(),  qf.qyy.end(),  d_qyy.begin());
    thrust::copy(qf.qyz.begin(),  qf.qyz.end(),  d_qyz.begin());

}

void DeviceFields::CopyToHost(FluidFields& ff, QTensorFields& qf) const {
    thrust::copy(d_rho.begin(), d_rho.end(), ff.rho.begin());
    thrust::copy(d_ux.begin(),  d_ux.end(),  ff.ux.begin());
    thrust::copy(d_uy.begin(),  d_uy.end(),  ff.uy.begin());
    thrust::copy(d_uz.begin(),  d_uz.end(),  ff.uz.begin());
    thrust::copy(d_qxx.begin(),  d_qxx.end(),  qf.qxx.begin());
    thrust::copy(d_qxy.begin(),  d_qxy.end(),  qf.qxy.begin());
    thrust::copy(d_qxz.begin(),  d_qxz.end(),  qf.qxz.begin());
    thrust::copy(d_qyy.begin(),  d_qyy.end(),  qf.qyy.begin());
    thrust::copy(d_qyz.begin(),  d_qyz.end(),  qf.qyz.begin());
}
