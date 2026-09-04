#include "device_fields.h"
#include "device_solver.h"
#include <params.h>
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

// Launch config sized from the field's LocalGrid, so a future MPI+GPU path
// (PR VII) can shrink to per-rank local dims without touching kernel code.
// Named kernel_grid/kernel_block (not grid/block) so the CUDA launch geometry
// stays visually distinct from LocalGrid / df.grid at every call site.
static dim3 KernelGrid(const LocalGrid& g) {
    return dim3{
        static_cast<unsigned>((g.local_nx + kBlockX - 1) / kBlockX),
        static_cast<unsigned>((g.local_ny + kBlockY - 1) / kBlockY),
        static_cast<unsigned>((g.local_nz + kBlockZ - 1) / kBlockZ)
    };
}

template<typename BC>
void DeviceSolver<BC>::QTensorStep(DeviceFields& df) {

    const dim3 kernel_block{kBlockX, kBlockY, kBlockZ};
    const dim3 kernel_grid = KernelGrid(df.grid);

    GpuQTensorStep<BC><<<kernel_grid, kernel_block>>>(
        df.d_qxx,
        df.d_qxy,
        df.d_qxz,
        df.d_qyy,
        df.d_qyz,
        df.d_qxx_new,
        df.d_qxy_new,
        df.d_qxz_new,
        df.d_qyy_new,
        df.d_qyz_new,
        df.d_ux.data().get(),
        df.d_uy.data().get(),
        df.d_uz.data().get(),
        df.d_force_x.data().get(),
        df.d_force_y.data().get(),
        df.d_force_z.data().get(),
        df.d_Sigma_xx,
        df.d_Sigma_xy,
        df.d_Sigma_xz,
        df.d_Sigma_yy,
        df.d_Sigma_yz,
        df.d_Tau_xy,
        df.d_Tau_xz,
        df.d_Tau_yz,
        df.grid
    );
    checkCudaErrors(cudaGetLastError());

    // Swap Q-tensor buffers (both are raw pointers now).
    std::swap(df.d_qxx, df.d_qxx_new);
    std::swap(df.d_qxy, df.d_qxy_new);
    std::swap(df.d_qxz, df.d_qxz_new);
    std::swap(df.d_qyy, df.d_qyy_new);
    std::swap(df.d_qyz, df.d_qyz_new);

    GpuComputeBodyForce<BC><<<kernel_grid, kernel_block>>>(
        df.d_qxx,
        df.d_qxy,
        df.d_qxz,
        df.d_qyy,
        df.d_qyz,
        df.d_ux.data().get(),
        df.d_uy.data().get(),
        df.d_uz.data().get(),
        df.d_force_x.data().get(),
        df.d_force_y.data().get(),
        df.d_force_z.data().get(),
        df.d_Sigma_xx,
        df.d_Sigma_xy,
        df.d_Sigma_xz,
        df.d_Sigma_yy,
        df.d_Sigma_yz,
        df.d_Tau_xy,
        df.d_Tau_xz,
        df.d_Tau_yz,
        df.grid
    );
    checkCudaErrors(cudaGetLastError());
}

template<typename BC>
void DeviceSolver<BC>::LBMStep(DeviceFields& df) {

    const dim3 kernel_block{kBlockX, kBlockY, kBlockZ};
    const dim3 kernel_grid = KernelGrid(df.grid);

    GpuCollideAndStream<BC><<<kernel_grid, kernel_block>>>(
        df.d_f,
        df.d_f_new,
        df.d_force_x.data().get(),
        df.d_force_y.data().get(),
        df.d_force_z.data().get(),
        df.d_rho.data().get(),
        df.d_ux.data().get(),
        df.d_uy.data().get(),
        df.d_uz.data().get(),
        df.grid
    );
    checkCudaErrors(cudaGetLastError());

    std::swap(df.d_f, df.d_f_new);
}

#include <sim_config.h>   // for SimBC

template struct DeviceSolver<SimBC>;
