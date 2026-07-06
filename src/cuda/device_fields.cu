#include "device_fields.h"
#include "params.h"
#include "cuda_utils.h"
#include "kernels.cu"

DeviceFields::DeviceFields() :
    d_f       (Params::nx * Params::ny * Params::nz * Params::ndir, 0.0),
    d_f_new   (Params::nx * Params::ny * Params::nz * Params::ndir, 0.0),
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

    checkCudaErrors(cudaMemcpyToSymbol(a2, &Params::A, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(a3, &Params::B, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(a4, &Params::C, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(K, &Params::L, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(LAMBDA, &Params::LAMBDA, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(GAMMA, &Params::GAMMA, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(ALPHA, &Params::ALPHA, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(MU, &Params::MU, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(dt, &Params::DT, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(tau, &Params::TAUF, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(omega, &Params::omega, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(omega_prime, &Params::omega_prime, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(omega_forcing, &Params::omega_forcing, sizeof(double)));
    checkCudaErrors(cudaDeviceSynchronize()); // flush any queued work
    
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

    thrust::copy(qf.qxx_data.begin(),  qf.qxx_data.end(),  d_qxx.begin());
    thrust::copy(qf.qxy_data.begin(),  qf.qxy_data.end(),  d_qxy.begin());
    thrust::copy(qf.qxz_data.begin(),  qf.qxz_data.end(),  d_qxz.begin());
    thrust::copy(qf.qyy_data.begin(),  qf.qyy_data.end(),  d_qyy.begin());
    thrust::copy(qf.qyz_data.begin(),  qf.qyz_data.end(),  d_qyz.begin());

}

void DeviceFields::CopyToHost(FluidFields& ff, QTensorFields& qf) const {
    thrust::copy(d_rho.begin(), d_rho.end(), ff.rho_data.begin());
    thrust::copy(d_ux.begin(),  d_ux.end(),  ff.ux_data.begin());
    thrust::copy(d_uy.begin(),  d_uy.end(),  ff.uy_data.begin());
    thrust::copy(d_uz.begin(),  d_uz.end(),  ff.uz_data.begin());
    thrust::copy(d_qxx.begin(),  d_qxx.end(),  qf.qxx_data.begin());
    thrust::copy(d_qxy.begin(),  d_qxy.end(),  qf.qxy_data.begin());
    thrust::copy(d_qxz.begin(),  d_qxz.end(),  qf.qxz_data.begin());
    thrust::copy(d_qyy.begin(),  d_qyy.end(),  qf.qyy_data.begin());
    thrust::copy(d_qyz.begin(),  d_qyz.end(),  qf.qyz_data.begin());
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
