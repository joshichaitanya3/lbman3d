#ifndef LBM_AN_PHYSICS_HELPERS_H_
#define LBM_AN_PHYSICS_HELPERS_H_

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

#include "params.h"

using namespace Params;

// Flat, periodic (x,y,z) -> offset for grid-sized fields; same layout on
// host and device since there's no direction index to complicate things.
inline CUDA_HOST_DEVICE int idx(int x, int y, int z) {
    return ((z + nz) % nz) * ny * nx + ((y + ny) % ny) * nx + ((x + nx) % nx);
}

// idx(x,y,z,i) has a DIFFERENT layout on host vs device:
//   - host:   i fastest-varying — all directions for one grid point are
//     contiguous, matching the CPU's per-point loop over ndir.
//   - device: i slowest-varying — all grid points for one direction are
//     contiguous, since a kernel step processes one direction across many
//     threads/blocks at once.
// __CUDA_ARCH__ is only defined during nvcc's device-code compilation pass
// of a __host__ __device__ function (undefined during its host pass), so a
// single function body branches on it rather than needing two overloads —
// nvcc treats __host__-tagged and __device__-tagged free functions with the
// same name/signature as a redefinition, not as distinct overloads.
inline CUDA_HOST_DEVICE int idx(int x, int y, int z, int i) {
#ifdef __CUDA_ARCH__
    return i * nz * ny * nx + ((z + nz) % nz) * ny * nx + ((y + ny) % ny) * nx + ((x + nx) % nx);
#else
    return (((z + nz) % nz) * ny * nx + ((y + ny) % ny) * nx + ((x + nx) % nx)) * ndir + i;
#endif
}

#endif // LBM_AN_PHYSICS_HELPERS_H_
