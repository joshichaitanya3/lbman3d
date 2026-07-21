#pragma once
#include <params.h>

// Rank-local view of the domain.
// For a single-rank run, all local_* equal Params::nx/ny/nz and offsets are 0.
// For MPI, each rank constructs this from global dimensions + rank topology.
struct LocalGrid {
    int local_nx, local_ny, local_nz;
    int offset_x,  offset_y,  offset_z;   // position in global domain

    static LocalGrid SingleRank() {
        return {Params::nx, Params::ny, Params::nz, 0, 0, 0};
    }

    /* !\brief Domain volume
     *
     * Currently set to `local_nx * local_ny * local_nz`.
     * We will use this volume to allocate fields for now, while we 
     * setup the MPI machinery. Eventually, the actual volume allocated
     * will be (local_nx + 2) * (local_ny + 2) * (local_nz + 2), via a 
     * separate `HaloVolume` method
     */
    int Volume() const { return local_nx * local_ny * local_nz; }
};
