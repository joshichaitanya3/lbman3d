#include <gtest/gtest.h>
#include "offsets.h"
#include "boundary.h"
#include <random>

/*
 * InteriorPassthrough: QAxisOffset(x, ±1, n) == x±1 for non-boundary x
 * NeumannClamp: QWallOffset<Neumann>(0, -1, n) == 0; QWallOffset<Neumann>(n-1, +1, n) == n-1
 * PeriodicWrap: QWallOffset<Periodic>(0, -1, 8) == 7; QWallOffset<Periodic>(7, +1, 8) == 0
 * AxisDispatch: For FullyPeriodicConfig, QXoff(0, -1) == nx-1. For ChannelConfig (Periodic Z, Neumann X/Y), QZoff(0,-1) == nz-1 but QYoff(0,-1) == 0.
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

    int out;
    out = QXoff<FullyPeriodicConfig>(0, -1);
    EXPECT_EQ(out, nx-1);
    out = QXoff<FullyPeriodicConfig>(nx-1, 1);
    EXPECT_EQ(out, 0);
    out = QYoff<FullyPeriodicConfig>(0, -1);
    EXPECT_EQ(out, ny-1);
    out = QYoff<FullyPeriodicConfig>(ny-1, 1);
    EXPECT_EQ(out, 0);
    out = QZoff<FullyPeriodicConfig>(0, -1);
    EXPECT_EQ(out, nz-1);
    out = QZoff<FullyPeriodicConfig>(nz-1, 1);
    EXPECT_EQ(out, 0);

    out = QXoff<ChannelConfig>(0, -1);
    EXPECT_EQ(out, 0);
    out = QXoff<ChannelConfig>(nx-1, 1);
    EXPECT_EQ(out, nx-1);
    out = QYoff<ChannelConfig>(0, -1);
    EXPECT_EQ(out, 0);
    out = QYoff<ChannelConfig>(ny-1, 1);
    EXPECT_EQ(out, ny-1);
    out = QZoff<ChannelConfig>(0, -1);
    EXPECT_EQ(out, nz-1);
    out = QZoff<ChannelConfig>(nz-1, 1);
    EXPECT_EQ(out, 0);

}
