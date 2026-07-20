#include <gtest/gtest.h>
#include "physics_helpers.h"
#include <math.h>

TEST(BGKCollide, FixedPointAtEquilibrium) {
    double feq = 1.0;
    EXPECT_DOUBLE_EQ(feq, PointwiseBGKCollide(feq, feq, 0.0));
}

TEST(BGKCollide, RelaxationDirection) {
    double feq = 1.0;
    double f_lo = 0.8;
    double f_hi = 1.3;
    double f_star_lo = PointwiseBGKCollide(f_lo, feq, 0.0);
    double f_star_hi = PointwiseBGKCollide(f_hi, feq, 0.0);

    EXPECT_TRUE(std::abs(f_star_lo - feq) < std::abs(f_lo - feq));
    EXPECT_TRUE(std::abs(f_star_hi - feq) < std::abs(f_hi - feq));
}

TEST(BGKCollide, ForcingLinearity) {
    double forcing = 1.0;
    double f = 1.5;
    double feq = 1.3;
    double diff1 = PointwiseBGKCollide(f, feq, 2*forcing) - PointwiseBGKCollide(f, feq, forcing);
    double diff2 = PointwiseBGKCollide(f, feq,   forcing) - PointwiseBGKCollide(f, feq,       0);

    EXPECT_DOUBLE_EQ(diff1, diff2);
}

TEST(BGKCollide, ForcingAntisymmetry) {
    // At u=0, the Guo operator reduces to F_i = ω'·w_i·3·(e_i·F). Because
    // e_{opp[i]} = −e_i and w_{opp[i]} = w_i, the pair sums to exactly zero:
    // the force injects momentum but not mass. Verified for the three
    // face-normal pairs (1↔3, 2↔4, 5↔6) with a non-axis-aligned force so
    // all three projections are exercised.
    const Moments m{1.0, {0.0, 0.0, 0.0}};
    const Vec3 F{1e-5, 2e-5, 3e-5};
    const double u2 = 0.0, uF = 0.0;

    for (int i : {1, 2, 5}) {
        const int j = Lattice::opp[i];
        const Vec3 ei{
            static_cast<double>(Lattice::ex[i]),
            static_cast<double>(Lattice::ey[i]),
            static_cast<double>(Lattice::ez[i])
        };
        const Vec3 ej{
            static_cast<double>(Lattice::ex[j]),
            static_cast<double>(Lattice::ey[j]),
            static_cast<double>(Lattice::ez[j])
        };
        auto [feqi, fi] = ComputeFeqAndForcing(m, u2, uF, F, ei, Lattice::w[i]);
        auto [feqj, fj] = ComputeFeqAndForcing(m, u2, uF, F, ej, Lattice::w[j]);
        EXPECT_DOUBLE_EQ(fi + fj, 0.0) << "pair (" << i << ", " << j << ")";
    }
}
