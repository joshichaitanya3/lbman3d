#include <gtest/gtest.h>
#include "params.h"
#include "physics_helpers.h"
#include <set>
#include <ranges>
#include <random>
#include "lattice_stencil.h"

using namespace Params;

TEST(TestIdx, Uniqueness) {
    std::set<int> ids;
    for (int z: std::views::iota(0, nz)) {
        for (int y: std::views::iota(0, ny)) {
            for (int x: std::views::iota(0, nx)) {
                int id = idx(x, y, z);
                ids.insert(id);
            }
        }
    }
    EXPECT_EQ(ids.size(), nx*ny*nz);
}

TEST(TestIdx, RowMajorFormula) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> nx_dist(0, nx-1);
    std::uniform_int_distribution<int> ny_dist(0, ny-1);
    std::uniform_int_distribution<int> nz_dist(0, nz-1);
    for (int sample = 0; sample < 5; sample++) {
        int x = nx_dist(rng);
        int y = ny_dist(rng);
        int z = nz_dist(rng);
        EXPECT_EQ(idx(x, y, z), z*ny*nx + y*nx + x);
    }
}

TEST(TestIdx, HostDirectionLayout) {
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
        EXPECT_EQ(idx(x, y, z, i), Lattice::ndir*idx(x, y, z) + i);
    }
}

TEST(TestIdx, InDomainEdgeCases) {
    EXPECT_TRUE(InDomain(0, 0, 0));
    EXPECT_TRUE(InDomain(nx-1, ny-1, nz-1));
    EXPECT_FALSE(InDomain(-1, 0, 0));
    ASSERT_DEATH({idx(-1, 0, 0);}, "coordinates out of domain");
}
