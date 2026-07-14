#include "device_fields.h"
#include "device_solver.h"
#include "params.h"
#include "cuda_utils.h"
#include "lattice_stencil.h"
#include "kernels.cu"

template<typename BC>
void DeviceSolver<BC>::Initialize(DeviceFields& df) {

    checkCudaErrors(cudaMemcpyToSymbol(d_ex, Lattice::ex, sizeof(Lattice::ex)));
    checkCudaErrors(cudaMemcpyToSymbol(d_ey, Lattice::ey, sizeof(Lattice::ey)));
    checkCudaErrors(cudaMemcpyToSymbol(d_ez, Lattice::ez, sizeof(Lattice::ez)));
    checkCudaErrors(cudaMemcpyToSymbol(d_w, Lattice::w, sizeof(Lattice::w)));
    checkCudaErrors(cudaMemcpyToSymbol(d_opp, Lattice::opp, sizeof(Lattice::opp)));
    checkCudaErrors(cudaMemcpyToSymbol(d_specX, Lattice::specX, sizeof(Lattice::specX)));
    checkCudaErrors(cudaMemcpyToSymbol(d_specY, Lattice::specY, sizeof(Lattice::specY)));
    checkCudaErrors(cudaMemcpyToSymbol(d_specZ, Lattice::specZ, sizeof(Lattice::specZ)));

}

template<typename BC>
void DeviceSolver<BC>::QTensorStep(DeviceFields& df) {

    GpuQTensorStep<BC><<<grid_, block_>>>(
        df.d_qxx.data().get(),
        df.d_qxy.data().get(),
        df.d_qxz.data().get(),
        df.d_qyy.data().get(),
        df.d_qyz.data().get(),
        df.d_qxx_new.data().get(),
        df.d_qxy_new.data().get(),
        df.d_qxz_new.data().get(),
        df.d_qyy_new.data().get(),
        df.d_qyz_new.data().get(),
        df.d_ux.data().get(),
        df.d_uy.data().get(),
        df.d_uz.data().get(),
        df.d_force_x.data().get(),
        df.d_force_y.data().get(),
        df.d_force_z.data().get(),
        df.d_Pxx.data().get(),
        df.d_Pxy.data().get(),
        df.d_Pxz.data().get(),
        df.d_Pyy.data().get(),
        df.d_Pyz.data().get()
    );
    checkCudaErrors(cudaGetLastError());

    df.d_qxx.swap(df.d_qxx_new);
    df.d_qxy.swap(df.d_qxy_new);
    df.d_qxz.swap(df.d_qxz_new);
    df.d_qyy.swap(df.d_qyy_new);
    df.d_qyz.swap(df.d_qyz_new);

    GpuComputeBodyForce<BC><<<grid_, block_>>>(
        df.d_qxx.data().get(),
        df.d_qxy.data().get(),
        df.d_qxz.data().get(),
        df.d_qyy.data().get(),
        df.d_qyz.data().get(),
        df.d_ux.data().get(),
        df.d_uy.data().get(),
        df.d_uz.data().get(),
        df.d_force_x.data().get(),
        df.d_force_y.data().get(),
        df.d_force_z.data().get(),
        df.d_Pxx.data().get(),
        df.d_Pxy.data().get(),
        df.d_Pxz.data().get(),
        df.d_Pyy.data().get(),
        df.d_Pyz.data().get()
    );
}

template<typename BC>
void DeviceSolver<BC>::LBMStep(DeviceFields& df) {

    GpuCollideAndStream<BC><<<grid_, block_>>>(
        df.d_f.data().get(),
        df.d_f_new.data().get(),
        df.d_force_x.data().get(),
        df.d_force_y.data().get(),
        df.d_force_z.data().get(),
        df.d_rho.data().get(),
        df.d_ux.data().get(),
        df.d_uy.data().get(),
        df.d_uz.data().get()
    );
    checkCudaErrors(cudaGetLastError());

    df.d_f.swap(df.d_f_new);
}

#include "sim_config.h"   // for SimBC

template struct DeviceSolver<SimBC>;
