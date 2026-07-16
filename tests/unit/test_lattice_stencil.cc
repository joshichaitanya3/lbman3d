#include <gtest/gtest.h>
#include "params.h"
#include "lattice_stencil.h"

using namespace Lattice;

TEST(LatticeStencil, WeightsSum) {
    double sum = 0.0;
    for (int i = 0; i < ndir; i++) sum += w[i];
    EXPECT_DOUBLE_EQ(sum, 1.0);
}

TEST(LatticeStencil, OppInvolution) {
    for (int i = 0; i < ndir; i++)
        EXPECT_EQ(i, opp[opp[i]]);
}

TEST(LatticeStencil, OppNegatesVelocity) {
    for (int i = 1; i < ndir; i++) {
        EXPECT_EQ(ex[opp[i]], -ex[i]);
        EXPECT_EQ(ey[opp[i]], -ey[i]);
        EXPECT_EQ(ez[opp[i]], -ez[i]);
    }
}

TEST(LatticeStencil, SpecXCorrectness) { // specX reverses only x component, keeps others same
    for (int i = 1; i < ndir; i++) {
        EXPECT_EQ(ex[specX[i]], -ex[i]);
        EXPECT_EQ(ey[specX[i]], +ey[i]);
        EXPECT_EQ(ez[specX[i]], +ez[i]);
    }
}

TEST(LatticeStencil, SpecYCorrectness) {
    for (int i = 1; i < ndir; i++) {
        EXPECT_EQ(ex[specY[i]], +ex[i]);
        EXPECT_EQ(ey[specY[i]], -ey[i]);
        EXPECT_EQ(ez[specY[i]], +ez[i]);
    }
}

TEST(LatticeStencil, SpecZCorrectness) {
    for (int i = 1; i < ndir; i++) {
        EXPECT_EQ(ex[specZ[i]], +ex[i]);
        EXPECT_EQ(ey[specZ[i]], +ey[i]);
        EXPECT_EQ(ez[specZ[i]], -ez[i]);
    }
}

TEST(LatticeStencil, MissingDirectionSetXLo) {
    int idx = 0;
    for (int i = 1; i < ndir; i++) {
        if (ex[i] > 0) {
            EXPECT_EQ(i, missingXLo[idx]);
            idx += 1;
        }
    }
}

TEST(LatticeStencil, MissingDirectionSetXHi) {
    int idx = 0;
    for (int i = 1; i < ndir; i++) {
        if (ex[i] < 0) {
            EXPECT_EQ(i, missingXHi[idx]);
            idx += 1;
        }
    }
}

TEST(LatticeStencil, MissingDirectionSetYLo) {
    int idx = 0;
    for (int i = 1; i < ndir; i++) {
        if (ey[i] > 0) {
            EXPECT_EQ(i, missingYLo[idx]);
            idx += 1;
        }
    }
}

TEST(LatticeStencil, MissingDirectionSetYHi) {
    int idx = 0;
    for (int i = 1; i < ndir; i++) {
        if (ey[i] < 0) {
            EXPECT_EQ(i, missingYHi[idx]);
            idx += 1;
        }
    }
}

TEST(LatticeStencil, MissingDirectionSetZLo) {
    int idx = 0;
    for (int i = 1; i < ndir; i++) {
        if (ez[i] > 0) {
            EXPECT_EQ(i, missingZLo[idx]);
            idx += 1;
        }
    }
}

TEST(LatticeStencil, MissingDirectionSetZHi) {
    int idx = 0;
    for (int i = 1; i < ndir; i++) {
        if (ez[i] < 0) {
            EXPECT_EQ(i, missingZHi[idx]);
            idx += 1;
        }
    }
}


