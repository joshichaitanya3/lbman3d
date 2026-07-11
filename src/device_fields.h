#ifndef LBM_AN_DEVICE_FIELDS_H_
#define LBM_AN_DEVICE_FIELDS_H_

#include <string>
#include "params.h"

#ifdef SIM_WITH_CUDA

#include <array>
#include <vector>
#include "qtensor_fields.h"
#include "fluid_fields.h"
#include <thrust/device_vector.h>

// Selects the CUDA device to use (device 0) and returns a human-readable,
// multi-line description of it (name, compute capability, memory, ...) for
// logging. Call once, before touching any other CUDA API.
std::string InitializeComputeBackend();

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

    DeviceFields();

    void Initialize(const QTensorFields& qf);

    void CopyToHost(FluidFields& ff, QTensorFields& qf) const;

    void QTensorStep();
    void LBMStep();

};

#else
#include "format_compat.h"

inline std::string InitializeComputeBackend() {
    return compat::format("CPU (OpenMP, numprocs = {})", Params::numprocs);
}

struct DeviceFields {};   // zero-size, optimized away entirely

#endif
#endif // LBM_AN_DEVICE_FIELDS_H_
