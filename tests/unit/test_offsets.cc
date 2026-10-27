#include <gtest/gtest.h>
#include "offsets.h"
#include "boundary.h"
#include "local_grid.h"
#include <random>

/*
 * InteriorPassthrough: QAxisOffset(x, ±1, n) == x±1 for non-boundary x
 * NeumannClamp: QWallOffset<Neumann>(0, -1, n) == 0; QWallOffset<Neumann>(n-1, +1, n) == n-1
 * PeriodicWrap: QWallOffset<Periodic>(0, -1, 8) == 7; QWallOffset<Periodic>(7, +1, 8) == 0
 * AxisDispatch: For FullyPeriodicConfig, QXoff(0, -1, g) == nx-1. For ChannelConfig (Periodic Z, Neumann X/Y), QZoff(0,-1, g) == nz-1 but QYoff(0,-1, g) == 0.
 * UnsplitResolveMatchesAxisOffset: on an unsplit axis ResolveAxisOffset is QAxisOffset, for every i and s.
 * SplitPeriodicSeamsReadGhosts: on a split periodic axis, both seams (including the global wrap) land in the ghost layer.
 * SplitWalledAxis: on a split walled axis, a rank seam lands in the ghost layer; only the physical wall clamps.
 */

TEST(Offsets, InteriorPassthrough) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> nx_dist(1, nx-2); // interior point
    for (int sample = 0; sample < 5; sample++) {
        int x = nx_dist(rng);
        int offp = QAxisOffset<Neumann, Neumann>(x, 1, nx);
        int offm = QAxisOffset<Neumann, Neumann>(x, -1, nx);
        int xp = x+1;
        int xm = x-1;
        EXPECT_EQ(offp, xp);
        EXPECT_EQ(offm, xm);
    }
}

TEST(Offsets, NeumannClamp) {
    int off_lo = QWallOffset<Neumann>(0, -1, nx);
    EXPECT_EQ(off_lo, 0);
    int off_hi = QWallOffset<Neumann>(nx-1, 1, nx);
    EXPECT_EQ(off_hi, nx-1);
}

TEST(Offsets, PeriodicWrap) {
    int off_lo = QWallOffset<Periodic>(0, -1, nx);
    EXPECT_EQ(off_lo, nx-1);
    int off_hi = QWallOffset<Periodic>(nx-1, 1, nx);
    EXPECT_EQ(off_hi, 0);
}

TEST(Offsets, AxisDispatch) {
    const LocalGrid g = LocalGrid::SingleRank();

    int out;
    out = QXoff<FullyPeriodicConfig>(0, -1, g);
    EXPECT_EQ(out, nx-1);
    out = QXoff<FullyPeriodicConfig>(nx-1, 1, g);
    EXPECT_EQ(out, 0);
    out = QYoff<FullyPeriodicConfig>(0, -1, g);
    EXPECT_EQ(out, ny-1);
    out = QYoff<FullyPeriodicConfig>(ny-1, 1, g);
    EXPECT_EQ(out, 0);
    out = QZoff<FullyPeriodicConfig>(0, -1, g);
    EXPECT_EQ(out, nz-1);
    out = QZoff<FullyPeriodicConfig>(nz-1, 1, g);
    EXPECT_EQ(out, 0);

    out = QXoff<ChannelConfig>(0, -1, g);
    EXPECT_EQ(out, 0);
    out = QXoff<ChannelConfig>(nx-1, 1, g);
    EXPECT_EQ(out, nx-1);
    out = QYoff<ChannelConfig>(0, -1, g);
    EXPECT_EQ(out, 0);
    out = QYoff<ChannelConfig>(ny-1, 1, g);
    EXPECT_EQ(out, ny-1);
    out = QZoff<ChannelConfig>(0, -1, g);
    EXPECT_EQ(out, nz-1);
    out = QZoff<ChannelConfig>(nz-1, 1, g);
    EXPECT_EQ(out, 0);

}

// x split in two (local_nx = nx/2), as on an np=2 run of the cube unit grid.
// kHaloMPI = 1, so -1 and local_nx are valid ghost indices.
constexpr int kHalf = nx / 2;
constexpr LocalGrid kLowRank  {kHalf, ny, nz, 0,     0, 0, 1};
constexpr LocalGrid kHighRank {kHalf, ny, nz, kHalf, 0, 0, 1};

TEST(Offsets, UnsplitResolveMatchesAxisOffset) {
    for (int i = -1; i <= nx; ++i) {
        for (int s : {-1, 0, 1}) {
            EXPECT_EQ((ResolveAxisOffset<Periodic, Periodic>(i, s, 0, nx, nx)),
                      (QAxisOffset<Periodic, Periodic>(i, s, nx)));
            EXPECT_EQ((ResolveAxisOffset<Neumann, Neumann>(i, s, 0, nx, nx)),
                      (QAxisOffset<Neumann, Neumann>(i, s, nx)));
        }
    }
}

TEST(Offsets, SplitPeriodicSeamsReadGhosts) {
    // The global wrap (low rank's -1 -> global nx-1) must read the ghost the
    // halo exchange filled, not local index nx-1, which lies past local_nx.
    EXPECT_EQ(QXoff<FullyPeriodicConfig>(0, -1, kLowRank), -1);
    EXPECT_EQ(QXoff<FullyPeriodicConfig>(kHalf - 1, +1, kLowRank), kHalf);
    EXPECT_EQ(QXoff<FullyPeriodicConfig>(0, -1, kHighRank), -1);
    EXPECT_EQ(QXoff<FullyPeriodicConfig>(kHalf - 1, +1, kHighRank), kHalf);
    // Interior steps are untouched.
    EXPECT_EQ(QXoff<FullyPeriodicConfig>(1, -1, kHighRank), 0);
    EXPECT_EQ(QXoff<FullyPeriodicConfig>(1, +1, kHighRank), 2);
}

TEST(Offsets, SplitWalledAxis) {
    // ChannelConfig: Neumann x walls.
    // Low rank: physical wall below, rank seam above.
    EXPECT_EQ(QXoff<ChannelConfig>(0, -1, kLowRank), 0);
    EXPECT_EQ(QXoff<ChannelConfig>(kHalf - 1, +1, kLowRank), kHalf);
    // High rank: rank seam below (not a wall, although local x == 0), and the
    // physical wall above (although local x != nx-1).
    EXPECT_EQ(QXoff<ChannelConfig>(0, -1, kHighRank), -1);
    EXPECT_EQ(QXoff<ChannelConfig>(kHalf - 1, +1, kHighRank), kHalf - 1);
}
