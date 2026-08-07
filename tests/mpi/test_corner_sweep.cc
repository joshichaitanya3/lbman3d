// Failing tests for PR VI step 3b: the multi-axis corner sweep.
//
// D3Q15's 8 body-diagonal populations (dirs 7..14) route through corner-ghost
// cells that the single-axis face exchange (step 3a) does not reach — because
// each face's pack range is `[0, local_n) × [0, local_n)`, and the corner
// ghost sits at `(local_nx, local_ny, local_nz)`, one step outside every face.
// The sequential axis sweep (x -> y -> z, with widened pack extents along the
// way) is what routes those pops to the diagonal rank.
//
// Registered with `WILL_FAIL TRUE` in tests/mpi/CMakeLists.txt so ctest expects
// this executable to exit non-zero until 3b lands. When it passes, CTest will
// report "unexpected pass" — that's the signal to drop the WILL_FAIL property.

#include <mpi.h>
#include <gtest/gtest.h>
#include <array>
#include "mpi/mpi_context.h"
#include "mpi/halo_exchange_lbm.h"
#include "params.h"
#include "lattice_stencil.h"
#include "fluid_fields.h"

namespace {

// For a 2×2×2 cart, coords ⊕ (1,1,1) picks out the body-diagonal neighbour on
// each axis (the sender for our owned-origin pop in dir 7 = (+1,+1,+1), and the
// receiver for our own corner-ghost pop).
int RankAtDiagonal(const MPIContext& mpi) {
    int coords[3] = {mpi.coords[0] ^ 1, mpi.coords[1] ^ 1, mpi.coords[2] ^ 1};
    int r;
    MPI_Cart_rank(mpi.cart_comm, coords, &r);
    return r;
}

}  // namespace

// Positive assertion: seed rank r's corner GHOST at dir 7, run ExchangeLBM,
// and verify the diagonal rank received it at its owned (0,0,0) dir 7.
//
// Every rank seeds its own corner ghost with a rank-encoded value, so after
// the sweep each rank's owned (0,0,0) dir 7 should hold the value seeded by
// its body-diagonal neighbour — not its own.
TEST(CornerSweep, Dir7GhostDeliveredToDiagonalOrigin) {
    MPIContext mpi;
    if (mpi.world_size != 8) GTEST_SKIP() << "requires np=8 (2×2×2 decomposition)";

    // Sanity — MPI_Dims_create(8, 3) must factor as {2,2,2}, otherwise the
    // topology assumption below (each axis split into 2) does not hold.
    ASSERT_EQ(mpi.dims[0], 2);
    ASSERT_EQ(mpi.dims[1], 2);
    ASSERT_EQ(mpi.dims[2], 2);

    LocalGrid grid = mpi.MakeLocalGrid();
    HaloExchangeLBM halo(grid, mpi);   // fully periodic, no wall flags

    FluidFields ff(grid);
    // Seed our own corner GHOST at dir 7 with a value that encodes our rank.
    // (In a real push step this slot would be written by streaming at owned
    // (local_nx-1, local_ny-1, local_nz-1) in dir 7 — see the "seed the ghost"
    // approach in the existing ExchangeLBMDeliversGhostToOwned test.)
    const double seed = 42.0 + 100.0 * mpi.world_rank;
    ff.f[grid.halo_idx(grid.local_nx, grid.local_ny, grid.local_nz, /*dir=*/7)] = seed;

    halo.ExchangeLBM(ff);

    const int sender = RankAtDiagonal(mpi);
    const double expected = 42.0 + 100.0 * sender;
    EXPECT_DOUBLE_EQ(ff.f[grid.halo_idx(0, 0, 0, /*dir=*/7)], expected)
        << "rank " << mpi.world_rank << " expected corner-diagonal pop from rank " << sender;
}

// Negative assertion — passes-when-expected-to-fail guard.
//
// Same seed as above; all other slots stay at FluidFields's default zero.
// After ExchangeLBM, iterate every OWNED cell (x,y,z ∈ owned range) and
// every direction — only (0,0,0) at dir 7 should be non-zero. All other
// owned slots must be exactly zero. Any non-zero elsewhere means a rogue
// mis-route lands data at a cell the sweep should never have written.
//
// Note we intentionally check only owned cells and skip the ghost/halo
// layer: a correct sweep leaves legitimate transverse-ghost intermediates
// there (they are the pop-in-flight between hops, or the rank's own
// original seed). Those are not mis-routes and would false-alarm a
// whole-buffer sentinel scan.
//
// Under step 3a this passes trivially (nothing moves). Its job is to catch
// a 3b implementation that scribbles seed values into owned cells other
// than the correct diagonal-mirror.
TEST(CornerSweep, Dir7SweepTouchesNoOtherSlot) {
    MPIContext mpi;
    if (mpi.world_size != 8) GTEST_SKIP() << "requires np=8 (2×2×2 decomposition)";

    LocalGrid grid = mpi.MakeLocalGrid();
    HaloExchangeLBM halo(grid, mpi);

    FluidFields ff(grid);   // f defaults to 0 everywhere

    const double seed = 42.0 + 100.0 * mpi.world_rank;
    ff.f[grid.halo_idx(grid.local_nx, grid.local_ny, grid.local_nz, /*dir=*/7)] = seed;

    halo.ExchangeLBM(ff);

    int stray = 0;
    for (int z = 0; z < grid.local_nz; ++z)
        for (int y = 0; y < grid.local_ny; ++y)
            for (int x = 0; x < grid.local_nx; ++x)
                for (int dir = 0; dir < Lattice::ndir; ++dir) {
                    if (x == 0 && y == 0 && z == 0 && dir == 7) continue;
                    if (ff.f[grid.halo_idx(x, y, z, dir)] != 0.0) ++stray;
                }
    EXPECT_EQ(stray, 0) << "rank " << mpi.world_rank
                        << " saw " << stray
                        << " non-zero owned slots outside the diagonal-mirror";
}
