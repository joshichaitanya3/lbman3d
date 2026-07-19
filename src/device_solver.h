#ifndef LBM_AN_DEVICE_SOLVER_H_
#define LBM_AN_DEVICE_SOLVER_H_

#include <string>
#include <params.h>
#include "device_fields.h"

#ifdef SIM_WITH_CUDA
template<typename BC>
struct DeviceSolver {
    // int gpu_id;
    // cudaStream_t stream;

    void Initialize(DeviceFields& df);
    void QTensorStep(DeviceFields& df);
    void LBMStep(DeviceFields& df);

};

#else
template<typename BC>
struct DeviceSolver {};   // zero-size, optimized away entirely

#endif
#endif // LBM_AN_DEVICE_SOLVER_H_
