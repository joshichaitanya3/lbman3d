#include "device_fields.h"
#include "params.h"
#include "cuda_utils.h"
#include "lattice_stencil.h"
#include "kernels.cu"

std::string InitializeComputeBackend() {
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
        "Warning: The GPU version is currently only proof-of-concept:\n"
        "It implements *only* fully periodic boundary conditions regardless of the sim_config, "
        "and does not implement passive stresses. For a production run, please recompile with "
        "-DLBM_FORCE_CPU=ON flag in CMake.\n",
        device_id, props.name, props.multiProcessorCount,
        props.major, props.minor,
        props.totalGlobalMem / bytesPerMiB, free_mem / bytesPerMiB,
        props.asyncEngineCount, props.canMapHostMemory);
}

DeviceFields::DeviceFields() :
    d_f       (Params::nx * Params::ny * Params::nz * Lattice::ndir, 0.0),
    d_f_new   (Params::nx * Params::ny * Params::nz * Lattice::ndir, 0.0),
    d_rho     (Params::nx * Params::ny * Params::nz, Params::kDensity),
    d_ux      (Params::nx * Params::ny * Params::nz, 0.0),
    d_uy      (Params::nx * Params::ny * Params::nz, 0.0),
    d_uz      (Params::nx * Params::ny * Params::nz, 0.0),
    d_force_x      (Params::nx * Params::ny * Params::nz, 0.0),
    d_force_y      (Params::nx * Params::ny * Params::nz, 0.0),
    d_force_z      (Params::nx * Params::ny * Params::nz, 0.0),
    d_qxx     (Params::nx * Params::ny * Params::nz, 0.0),
    d_qxx_new (Params::nx * Params::ny * Params::nz, 0.0),
    d_qxy     (Params::nx * Params::ny * Params::nz, 0.0),
    d_qxy_new (Params::nx * Params::ny * Params::nz, 0.0),
    d_qxz     (Params::nx * Params::ny * Params::nz, 0.0),
    d_qxz_new (Params::nx * Params::ny * Params::nz, 0.0),
    d_qyy     (Params::nx * Params::ny * Params::nz, 0.0),
    d_qyy_new (Params::nx * Params::ny * Params::nz, 0.0),
    d_qyz     (Params::nx * Params::ny * Params::nz, 0.0),
    d_qyz_new (Params::nx * Params::ny * Params::nz, 0.0)
{}

void DeviceFields::Initialize(const QTensorFields& qf) {

    checkCudaErrors(cudaMemcpyToSymbol(d_ex, Lattice::ex, sizeof(Lattice::ex)));
    checkCudaErrors(cudaMemcpyToSymbol(d_ey, Lattice::ey, sizeof(Lattice::ey)));
    checkCudaErrors(cudaMemcpyToSymbol(d_ez, Lattice::ez, sizeof(Lattice::ez)));
    checkCudaErrors(cudaMemcpyToSymbol(d_w, Lattice::w, sizeof(Lattice::w)));
    checkCudaErrors(cudaMemcpyToSymbol(d_opp, Lattice::opp, sizeof(Lattice::opp)));
    checkCudaErrors(cudaMemcpyToSymbol(d_specX, Lattice::specX, sizeof(Lattice::specX)));
    checkCudaErrors(cudaMemcpyToSymbol(d_specY, Lattice::specY, sizeof(Lattice::specY)));
    checkCudaErrors(cudaMemcpyToSymbol(d_specZ, Lattice::specZ, sizeof(Lattice::specZ)));

    checkCudaErrors(cudaFuncSetAttribute(
        GpuQTensorStep,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kQstepSmem
    ));
    std::cout << std::format("Requested {} bytes of shared memory\n", kQstepSmem) << std::endl; 
    GpuInitialize<<<grid_, block_>>>(
        d_f.data().get(),
        d_rho.data().get(),
        d_ux.data().get(),
        d_uy.data().get(),
        d_uz.data().get()
    );
    checkCudaErrors(cudaGetLastError());

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

void DeviceFields::QTensorStep() {

    GpuQTensorStep<<<grid_, block_, kQstepSmem>>>(
        d_qxx.data().get(),
        d_qxy.data().get(),
        d_qxz.data().get(),
        d_qyy.data().get(),
        d_qyz.data().get(),
        d_qxx_new.data().get(),
        d_qxy_new.data().get(),
        d_qxz_new.data().get(),
        d_qyy_new.data().get(),
        d_qyz_new.data().get(),
        d_ux.data().get(),
        d_uy.data().get(),
        d_uz.data().get(),
        d_force_x.data().get(),
        d_force_y.data().get(),
        d_force_z.data().get()
    );
    checkCudaErrors(cudaGetLastError());

    d_qxx.swap(d_qxx_new);
    d_qxy.swap(d_qxy_new);
    d_qxz.swap(d_qxz_new);
    d_qyy.swap(d_qyy_new);
    d_qyz.swap(d_qyz_new);
}

void DeviceFields::LBMStep() {
    GpuCollideAndStream<<<grid_, block_>>>(
        d_f.data().get(),
        d_f_new.data().get(),
        d_force_x.data().get(),
        d_force_y.data().get(),
        d_force_z.data().get(),
        d_rho.data().get(),
        d_ux.data().get(),
        d_uy.data().get(),
        d_uz.data().get()
    );
    checkCudaErrors(cudaGetLastError());

    d_f.swap(d_f_new);
}
