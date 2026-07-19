#include <gtest/gtest.h>
#include "physics_helpers.h"
#include <math.h>

// inline CUDA_HOST_DEVICE double PointwiseBGKCollide(
//     double f,
//     double feq,
//     double forcing_term
// ) {
//     return omega * f + omega_prime * feq + DT * forcing_term;
// }

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
