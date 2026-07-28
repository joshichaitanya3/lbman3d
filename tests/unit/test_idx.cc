#include <gtest/gtest.h>
#include "params.h"
#include "local_grid.h"
#include <set>
#include <ranges>
#include <random>
#include "lattice_stencil.h"

using namespace Params;

// LocalGrid::SingleRank() has kHaloMPI == 0, so halo_idx collapses to the plain
// compact row-major layout — these mirror the historical free-function idx()
// tests, now exercising the LocalGrid member that replaced it.

TEST(TestIdx, Uniqueness) {
    LocalGrid g = LocalGrid::SingleRank();
    std::set<int> ids;
    for (int z: std::views::iota(0, nz)) {
        for (int y: std::views::iota(0, ny)) {
            for (int x: std::views::iota(0, nx)) {
                ids.insert(g.halo_idx(x, y, z));
            }
        }
    }
    EXPECT_EQ(ids.size(), nx*ny*nz);
}

TEST(TestIdx, RowMajorFormula) {
    LocalGrid g = LocalGrid::SingleRank();
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> nx_dist(0, nx-1);
    std::uniform_int_distribution<int> ny_dist(0, ny-1);
    std::uniform_int_distribution<int> nz_dist(0, nz-1);
    for (int sample = 0; sample < 5; sample++) {
        int x = nx_dist(rng);
        int y = ny_dist(rng);
        int z = nz_dist(rng);
        EXPECT_EQ(g.halo_idx(x, y, z), z*ny*nx + y*nx + x);
    }
}

TEST(TestIdx, HostDirectionLayout) {
    LocalGrid g = LocalGrid::SingleRank();
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> nx_dist(0, nx-1);
    std::uniform_int_distribution<int> ny_dist(0, ny-1);
    std::uniform_int_distribution<int> nz_dist(0, nz-1);
    std::uniform_int_distribution<int> dir_dist(0, Lattice::ndir-1);
    for (int sample = 0; sample < 5; sample++) {
        int x = nx_dist(rng);
        int y = ny_dist(rng);
        int z = nz_dist(rng);
        int i = dir_dist(rng);
        EXPECT_EQ(g.halo_idx(x, y, z, i), Lattice::ndir*g.halo_idx(x, y, z) + i);
    }
}

TEST(TestIdx, InDomainEdgeCases) {
    LocalGrid g = LocalGrid::SingleRank();
    EXPECT_TRUE(g.InDomain(0, 0, 0));
    EXPECT_TRUE(g.InDomain(nx-1, ny-1, nz-1));
    EXPECT_FALSE(g.InDomain(-1, 0, 0));
    EXPECT_FALSE(g.InDomain(nx, 0, 0));
    ASSERT_DEATH({g.halo_idx(-1, 0, 0, 0);}, "out of domain");
}

// Halo-padded layout: with a ghost layer the owned coordinates shift by
// kHaloMPI and the per-axis strides widen by 2*kHaloMPI. Exercises the MPI
// indexing path that SingleRank (kHaloMPI == 0) can't reach.
TEST(TestIdx, HaloPaddedLayout) {
    LocalGrid g{nx, ny, nz, 0, 0, 0, /*kHaloMPI=*/1};
    const int sx = nx + 2, sy = ny + 2;
    EXPECT_EQ(g.halo_idx(0, 0, 0),  1*sy*sx + 1*sx + 1);        // owned origin sits past the ghost layer
    EXPECT_EQ(g.halo_idx(-1, 0, 0), 1*sy*sx + 1*sx + 0);        // lo-x ghost
    EXPECT_EQ(g.halo_idx(nx, 0, 0), 1*sy*sx + 1*sx + (nx+1));   // hi-x ghost
}
