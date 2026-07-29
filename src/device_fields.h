#ifndef LBM_AN_DEVICE_FIELDS_H_
#define LBM_AN_DEVICE_FIELDS_H_

#include <string>
#include <params.h>
#include "mpi/mpi_context.h"

#ifdef SIM_WITH_CUDA

#include <array>
#include <vector>
#include "qtensor_fields.h"
#include "fluid_fields.h"
#include <thrust/device_vector.h>
#include "local_grid.h"

// Selects the CUDA device to use (device 0) and returns a human-readable,
// multi-line description of it (name, compute capability, memory, ...) plus
// the MPI decomposition (world size, dims split) for logging. Call once,
// before touching any other CUDA API.
std::string InitializeComputeBackend(const MPIContext& mpi);

struct DeviceFields {
    // int gpu_id;
    // cudaStream_t stream;
    thrust::device_vector<double> d_f, d_f_new;
    thrust::device_vector<double> d_rho, d_ux, d_uy, d_uz;
    thrust::device_vector<double> d_force_x, d_force_y, d_force_z;
    thrust::device_vector<double> d_qxx, d_qxx_new;
    thrust::device_vector<double> d_qxy, d_qxy_new;
    thrust::device_vector<double> d_qxz, d_qxz_new;
    thrust::device_vector<double> d_qyy, d_qyy_new;
    thrust::device_vector<double> d_qyz, d_qyz_new;

    thrust::device_vector<double> d_Pxx;
    thrust::device_vector<double> d_Pxy;
    thrust::device_vector<double> d_Pxz;
    thrust::device_vector<double> d_Pyy;
    thrust::device_vector<double> d_Pyz;

    explicit DeviceFields(LocalGrid g = LocalGrid::SingleRank());

    // ff is mutated transiently: ff.f_new is reused as scratch space for the
    // host->device layout transpose, then restored to its normal contents.
    // See device_fields.cu for why this is safe.
    void Initialize(FluidFields& ff, const QTensorFields& qf);

    void CopyToHost(FluidFields& ff, QTensorFields& qf) const;

};

#else
#include "format_compat.h"
#include "local_grid.h"

inline std::string InitializeComputeBackend(const MPIContext& mpi) {
    return compat::format(
        "CPU (OpenMP kNumOMPThreads = {}, MPI world_size = {}, dims = [{}, {}, {}])",
        Params::kNumOMPThreads, mpi.world_size,
        mpi.dims[0], mpi.dims[1], mpi.dims[2]);
}

struct DeviceFields {
    explicit DeviceFields(LocalGrid) {}
};   // zero-size, optimized away entirely

#endif
#endif // LBM_AN_DEVICE_FIELDS_H_
