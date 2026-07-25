#pragma once
#include <params.h>

// Rank-local view of the domain.
// For a single-rank run, all local_* equal Params::nx/ny/nz and offsets are 0.
// For MPI, each rank constructs this from global dimensions + rank topology.
struct LocalGrid {
    int local_nx, local_ny, local_nz;
    int offset_x,  offset_y,  offset_z;   // position in global domain
    int kHaloMPI;
    static LocalGrid SingleRank() {
        return {Params::nx, Params::ny, Params::nz, 0, 0, 0, 1};
    }


    inline int idx(int x, int y, int z) {
        return (z * local_ny * local_nx + y * local_nx + x);
    }
    inline int idx(int x, int y, int z, int i) {
        return (z * local_ny * local_nx + y * local_nx + x)*3 + i;
    }
    int Volume() const { return local_nx * local_ny * local_nz; }

    // Index into a halo-padded buffer (1 ghost cell on every face).
    // x, y, z are still the *interior* logical coordinates (0..local_n{x,y,z}-1);
    // the +1 shift and the (local_n{x,y}+2) strides account for the ghost layer.
    inline int halo_idx(int x, int y, int z) const {
        return ((z + kHaloMPI) * (local_ny + 2*kHaloMPI) * (local_nx + 2*kHaloMPI)
                 + (y + kHaloMPI) * (local_nx + 2*kHaloMPI)
                 + (x + kHaloMPI));
    }
    inline int halo_idx(int x, int y, int z, int i) const {
        return ((z + kHaloMPI) * (local_ny + 2*kHaloMPI) * (local_nx + 2*kHaloMPI)
                 + (y + kHaloMPI) * (local_nx + 2*kHaloMPI)
                 + (x + kHaloMPI))*3 + i;
    }
    int HaloVolume() const {
        return (local_nx + 2*kHaloMPI) * (local_ny + 2*kHaloMPI) * (local_nz + 2*kHaloMPI);
    }
};
