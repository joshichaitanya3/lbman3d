#include "device_fields.h"
#include <params.h>
#include "cuda_utils.h"
#include "lattice_stencil.h"
#include "physics_helpers.h"
#include <algorithm>
#include "local_grid.h"

std::string InitializeComputeBackend(const MPIContext& mpi) {
    // Per-node local rank derivation. NVSHMEM (VII-c) will bind to whichever
    // device is current at nvshmemx_init time — hardcoding device 0 pins every
    // rank on a node to the same GPU and produces no error message, just
    // catastrophic contention. Splitting MPI_COMM_WORLD by MPI_COMM_TYPE_SHARED
    // yields a communicator per shared-memory domain (one per node in typical
    // launches); rank inside it is the intra-node ordinal we bind to.
    int local_rank = 0;
    int local_size = 1;
#ifdef LBM_ENABLE_MPI
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi.world_rank,
                        MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);
    MPI_Comm_free(&node_comm);
#endif

    int device_count = 0;
    checkCudaErrors(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA devices visible to this rank");
    }
    // Modulo maps two common launch patterns cleanly:
    //   - launcher exposes all node GPUs to every rank (device_count == GPUs
    //     per node): each local_rank picks a distinct device up to device_count,
    //     then wraps (oversubscription — visible in the log, not silent).
    //   - launcher sets CUDA_VISIBLE_DEVICES per rank (device_count == 1):
    //     every rank maps to device 0, which is the only device it sees.
    const int assigned_device = local_rank % device_count;
    checkCudaErrors(cudaSetDevice(assigned_device));

    int device_id = 0;
    checkCudaErrors(cudaGetDevice(&device_id));

    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, device_id));

    size_t free_mem, total_mem;
    checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem));

    constexpr double bytesPerMiB = 1024.0 * 1024.0;

    return std::format(
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
        device_id, props.name, props.multiProcessorCount,
        props.major, props.minor,
        props.totalGlobalMem / bytesPerMiB, free_mem / bytesPerMiB,
        props.asyncEngineCount, props.canMapHostMemory,
        mpi.world_size, mpi.dims[0], mpi.dims[1], mpi.dims[2],
        local_rank, local_size, device_count);
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
