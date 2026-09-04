#ifndef LBM_DEVICE_ALLOCATOR_H_
#define LBM_DEVICE_ALLOCATOR_H_

#include <cstddef>
#include <stdexcept>

#ifdef SIM_WITH_CUDA

#include <cuda_runtime.h>
#include "cuda_utils.h"

#ifdef LBM_ENABLE_NVSHMEM
#include <nvshmem.h>
#endif

// Backend-agnostic allocator for halo-exchanged device fields.
//
// Halo-exchanged fields (d_f, d_f_new, d_q**, d_Sigma_*, d_Tau_*) must live on
// the symmetric heap when using NVSHMEM, or on regular device memory when using
// CUDA-only or CUDA-aware MPI. This header provides a compile-time dispatch to
// the correct allocator.
//
// Local-only fields (d_rho, d_ux/uy/uz, d_force_*, d_q**_new) use regular
// cudaMalloc and do not go through this interface.

inline double* AllocateHaloField(std::size_t nelems) {
#ifdef LBM_ENABLE_NVSHMEM
    double* ptr = static_cast<double*>(nvshmem_malloc(nelems * sizeof(double)));
    if (!ptr) {
        throw std::runtime_error("nvshmem_malloc failed");
    }
    return ptr;
#else
    double* ptr;
    checkCudaErrors(cudaMalloc(&ptr, nelems * sizeof(double)));
    return ptr;
#endif
}

inline void DeallocateHaloField(double* ptr) {
#ifdef LBM_ENABLE_NVSHMEM
    nvshmem_free(ptr);
#else
    checkCudaErrors(cudaFree(ptr));
#endif
}

#endif  // SIM_WITH_CUDA

#endif  // LBM_DEVICE_ALLOCATOR_H_
