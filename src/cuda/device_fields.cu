#include "device_fields.h"
#include <params.h>
#include "cuda_utils.h"
#include "lattice_stencil.h"
#include "physics_helpers.h"
#include <algorithm>
#include "local_grid.h"

std::string InitializeComputeBackend(const MPIContext& mpi) {
    checkCudaErrors(cudaSetDevice(0));

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
        "         MPI dims: [{}, {}, {}]\n",
        device_id, props.name, props.multiProcessorCount,
        props.major, props.minor,
        props.totalGlobalMem / bytesPerMiB, free_mem / bytesPerMiB,
        props.asyncEngineCount, props.canMapHostMemory,
        mpi.world_size, mpi.dims[0], mpi.dims[1], mpi.dims[2]);
}

DeviceFields::DeviceFields(LocalGrid g) :
    d_f       (g.Volume() * Lattice::ndir, 0.0),
    d_f_new   (g.Volume() * Lattice::ndir, 0.0),
    d_rho     (g.Volume(), Params::kDensity),
    d_ux      (g.Volume(), 0.0),
    d_uy      (g.Volume(), 0.0),
    d_uz      (g.Volume(), 0.0),
    d_force_x (g.Volume(), 0.0),
    d_force_y (g.Volume(), 0.0),
    d_force_z (g.Volume(), 0.0),
    d_qxx     (g.Volume(), 0.0),
    d_qxx_new (g.Volume(), 0.0),
    d_qxy     (g.Volume(), 0.0),
    d_qxy_new (g.Volume(), 0.0),
    d_qxz     (g.Volume(), 0.0),
    d_qxz_new (g.Volume(), 0.0),
    d_qyy     (g.Volume(), 0.0),
    d_qyy_new (g.Volume(), 0.0),
    d_qyz     (g.Volume(), 0.0),
    d_qyz_new (g.Volume(), 0.0),
    d_Pxx     (g.Volume(), 0.0),
    d_Pxy     (g.Volume(), 0.0),
    d_Pxz     (g.Volume(), 0.0),
    d_Pyy     (g.Volume(), 0.0),
    d_Pyz     (g.Volume(), 0.0)

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
    const int n = Params::nx * Params::ny * Params::nz;
    for (int z = 0; z < Params::nz; ++z)
        for (int y = 0; y < Params::ny; ++y)
            for (int x = 0; x < Params::nx; ++x)
                for (int i = 0; i < Lattice::ndir; ++i)
                    ff.f_new[i * n + idx(x, y, z)] = ff.f[idx(x, y, z, i)];

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
