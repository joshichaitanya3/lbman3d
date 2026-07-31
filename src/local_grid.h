#pragma once
#include <cassert>
#include <params.h>
#include "lattice_stencil.h"

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

// Rank-local view of the domain.
// For a single-rank run, all local_* equal Params::nx/ny/nz and offsets are 0.
// For MPI, each rank constructs this from global dimensions + rank topology.
struct LocalGrid {
    int local_nx, local_ny, local_nz;
    int offset_x,  offset_y,  offset_z;   // position in global domain
    int kHaloMPI;
    static LocalGrid SingleRank() {
        return {Params::nx, Params::ny, Params::nz, 0, 0, 0, 0};
    }


    inline CUDA_HOST_DEVICE int idx(int x, int y, int z) const {
        return (z * local_ny * local_nx + y * local_nx + x);
    }
    inline CUDA_HOST_DEVICE int idx(int x, int y, int z, int i) const {
        return (z * local_ny * local_nx + y * local_nx + x)*3 + i;
    }
    CUDA_HOST_DEVICE int Volume() const { return local_nx * local_ny * local_nz; }

    // Index into a halo-padded buffer (1 ghost cell on every face).
    // x, y, z are still the *interior* logical coordinates (0..local_n{x,y,z}-1);
    // the +1 shift and the (local_n{x,y}+2) strides account for the ghost layer.
    inline CUDA_HOST_DEVICE int halo_idx(int x, int y, int z) const {
        return ((z + kHaloMPI) * (local_ny + 2*kHaloMPI) * (local_nx + 2*kHaloMPI)
                 + (y + kHaloMPI) * (local_nx + 2*kHaloMPI)
                 + (x + kHaloMPI));
    }
    inline CUDA_HOST_DEVICE int halo_dirIdx(int x, int y, int z, int i) const {
        return ((z + kHaloMPI) * (local_ny + 2*kHaloMPI) * (local_nx + 2*kHaloMPI)
                 + (y + kHaloMPI) * (local_nx + 2*kHaloMPI)
                 + (x + kHaloMPI))*3 + i;
    }

    inline CUDA_HOST_DEVICE bool InDomain(int x, int y, int z) const {
        return (x >= 0) && (x < local_nx) && (y >= 0) && (y < local_ny) && (z >= 0) && (z < local_nz);
    }

    // Halo-inclusive bounds: owned cells plus the kHaloMPI-deep ghost layer on
    // every face. This is the valid range for halo_idx, since pack/unpack and
    // cross-rank streaming legitimately address ghost cells (InDomain, the
    // owned-only check, would reject them).
    inline CUDA_HOST_DEVICE bool InHaloDomain(int x, int y, int z) const {
        return (x >= -kHaloMPI) && (x < local_nx + kHaloMPI)
            && (y >= -kHaloMPI) && (y < local_ny + kHaloMPI)
            && (z >= -kHaloMPI) && (z < local_nz + kHaloMPI);
    }

    inline CUDA_HOST_DEVICE int halo_idx(int x, int y, int z, int i) const {
        assert(InHaloDomain(x, y, z) && "idx(x,y,z,i): coordinates out of halo-padded domain");
        assert(i >= 0 && i < Lattice::ndir && "idx(x,y,z,i): direction index out of range");
        #ifdef __CUDA_ARCH__
            return i * (local_nz+2*kHaloMPI) * (local_ny+2*kHaloMPI) * (local_nx+2*kHaloMPI) + halo_idx(x, y, z);
        #else
            return ( halo_idx(x, y, z) * Lattice::ndir + i );
        #endif
    }
    CUDA_HOST_DEVICE int HaloVolume() const {
        return (local_nx + 2*kHaloMPI) * (local_ny + 2*kHaloMPI) * (local_nz + 2*kHaloMPI);
    }
};
