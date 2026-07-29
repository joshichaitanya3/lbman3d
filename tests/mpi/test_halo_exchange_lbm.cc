#include <mpi.h>
#include <gtest/gtest.h>
#include "mpi/mpi_context.h"
#include "mpi/halo_exchange_lbm.h"
#include <array>
#include <vector>
#include "params.h"
#include "lattice_stencil.h"
#include "fluid_fields.h"

// The LBM exchange is post-push-stream: it reads from GHOST layers (where our
// own streaming deposited outgoing crossings) and writes into the neighbour's
// OWNED boundary. That is the opposite orientation from the Q-tensor halo, so
// these tests specifically target that reversed direction.

TEST(HaloExchangeLBM, PackLBMHiYZReadsHiXGhost) {
    MPIContext mpi;
    LocalGrid grid = mpi.MakeLocalGrid();
    HaloExchangeLBM halo(grid, mpi);

    // Fill the hi-x GHOST slice with position-encoded values, at dir=1.
    // Also fill the owned hi-x boundary with a distinct sentinel so we can
    // catch any accidental owned-plane reads (that was the old bug).
    std::vector<double> field(grid.HaloVolume() * Lattice::ndir, 0.0);
    for (int z = 0; z < grid.local_nz; ++z) {
        for (int y = 0; y < grid.local_ny; ++y) {
            field[grid.halo_idx(grid.local_nx,     y, z, 1)] =   1.0 + y * 10.0 + z * 100.0;   // ghost
            field[grid.halo_idx(grid.local_nx - 1, y, z, 1)] = -(1.0 + y * 10.0 + z * 100.0);  // owned
        }
    }

    halo.PackLBMHiYZ(field.data(), /*dir=*/1, /*k=*/0);

    for (int z = 0; z < grid.local_nz; ++z)
        for (int y = 0; y < grid.local_ny; ++y)
            EXPECT_DOUBLE_EQ(halo.send_buf_[1][z * grid.local_ny + y],
                             1.0 + y * 10.0 + z * 100.0);
}

TEST(HaloExchangeLBM, PackLBMLoXYReadsLoZGhost) {
    MPIContext mpi;
    LocalGrid grid = mpi.MakeLocalGrid();
    HaloExchangeLBM halo(grid, mpi);

    std::vector<double> field(grid.HaloVolume() * Lattice::ndir, 0.0);
    for (int y = 0; y < grid.local_ny; ++y)
        for (int x = 0; x < grid.local_nx; ++x)
            field[grid.halo_idx(x, y, -1, /*dir=*/6)] = 1.0 + x * 10.0 + y * 100.0;

    halo.PackLBMLoXY(field.data(), /*dir=*/6, /*k=*/2);

    for (int y = 0; y < grid.local_ny; ++y)
        for (int x = 0; x < grid.local_nx; ++x)
            EXPECT_DOUBLE_EQ(halo.send_buf_[4][2 * halo.max_xy + y * grid.local_nx + x],
                             1.0 + x * 10.0 + y * 100.0);
}

// End-to-end: seed rank 0's hi-x GHOST at dir 1 with a rank-encoded value;
// ExchangeLBM should deliver it to rank 1's OWNED lo-x boundary at dir 1
// (missingXLo[0] = 1), and leave all non-crossing dirs at that cell unchanged.
TEST(HaloExchangeLBM, ExchangeLBMDeliversGhostToOwned) {
    MPIContext mpi;
    if (mpi.world_size != 2) GTEST_SKIP() << "requires np=2";

    LocalGrid grid = mpi.MakeLocalGrid();
    HaloExchangeLBM halo(grid, mpi);   // fully periodic, no wall flags

    FluidFields ff(grid);
    // Seed a distinctive value in this rank's hi-x GHOST at dir 1 for one
    // (y, z) cell; every other f slot stays at whatever FluidFields ctor set.
    const double seed = 42.0 + mpi.world_rank;
    ff.f[grid.halo_idx(grid.local_nx, 3, 2, /*dir=*/1)] = seed;

    // Snapshot a NON-crossing dir at the receiver-side cell to detect any
    // stray writes (dir 3 has ex<0 → not in missingXLo, should be untouched).
    const int recv_from = (mpi.world_rank + 1) % 2;
    const double untouched_before = ff.f[grid.halo_idx(0, 3, 2, /*dir=*/3)];

    halo.ExchangeLBM(ff);

    // Rank r's owned lo-x boundary at (3, 2, dir=1) now holds rank recv_from's
    // seed (since dir 1 ∈ missingXLo — arrivals from -x neighbour = the other rank).
    EXPECT_DOUBLE_EQ(ff.f[grid.halo_idx(0, 3, 2, /*dir=*/1)], 42.0 + recv_from);
    // Non-crossing dir at same cell is untouched.
    EXPECT_DOUBLE_EQ(ff.f[grid.halo_idx(0, 3, 2, /*dir=*/3)], untouched_before);
}

// At a Y-wall corner (y=0 with YLo NoSlip), the local Y-bounce writes into
// dir 11 of the owned lo-x cell during the LBM step. The exchange must NOT
// overwrite that with the (empty) neighbour ghost slot. Verifies SkipUnpackYZ.
TEST(HaloExchangeLBM, ExchangeLBMSkipsWallCornerDirs) {
    MPIContext mpi;
    if (mpi.world_size != 2) GTEST_SKIP() << "requires np=2";

    LocalGrid grid = mpi.MakeLocalGrid();
    // Only Y-lo walled (index 2); other faces periodic.
    std::array<bool, 6> walls{false, false, /*YLo*/true, false, false, false};
    HaloExchangeLBM halo(grid, mpi, walls);

    FluidFields ff(grid);
    // Pretend a Y-lo bounce wrote a locally-valid value into owned (0, 0, z)
    // at dir 11 (ex=+1, ey=+1 — in missingXLo, but also a Y-lo target).
    const double bounce_value = 7.5 + mpi.world_rank;
    for (int z = 0; z < grid.local_nz; ++z)
        ff.f[grid.halo_idx(0, 0, z, /*dir=*/11)] = bounce_value;

    // Also seed dir 1 (ex=+1, ey=0 — in missingXLo but NOT Y-bounce affected)
    // so we confirm the exchange still fires for the non-wall-corner dirs.
    for (int z = 0; z < grid.local_nz; ++z)
        ff.f[grid.halo_idx(0, 0, z, /*dir=*/1)] = -1.0;   // sentinel to be overwritten

    halo.ExchangeLBM(ff);

    for (int z = 0; z < grid.local_nz; ++z) {
        // Dir 11 at Y-wall corner: preserved (skip fired).
        EXPECT_DOUBLE_EQ(ff.f[grid.halo_idx(0, 0, z, /*dir=*/11)], bounce_value);
        // Dir 1 at same cell: overwritten by exchange (no skip).
        EXPECT_NE(ff.f[grid.halo_idx(0, 0, z, /*dir=*/1)], -1.0);
    }
}
