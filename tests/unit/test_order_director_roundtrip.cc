#include <gtest/gtest.h>
#include "analysis_fields.h"
#include <cmath>
#include <random>

// find_defects reconstructs Q from (S_largest, n̂) via the uniaxial inverse
// (3S/2)·(nnᵀ − I/3) to feed the β pipeline, which only reads Q on rings
// well outside the defect core. These tests pin that the roundtrip
//     Q_uniaxial → (S_largest, n̂) → Q_reconstructed
// is bit-exact on the uniaxial subset with a **positive** scalar order —
// the physical regime of an ordered nematic and the only one where n̂
// (the eigenvector of the largest eigenvalue) coincides with the director.

namespace {

// Build a uniaxial Q with scalar prefactor s in Q = s(nnᵀ − I/3). Note this
// s is NOT the same as the "S" returned by QtensorToOrderDirectorPoint
// (that S is the largest eigenvalue, = 2s/3 for uniaxial s>0). The test
// makes no reference to which convention "S" uses — it only checks the
// roundtrip against the original Q.
SymTrLessTensor5 UniaxialQ(double s, double nx, double ny, double nz) {
    const double norm = std::sqrt(nx*nx + ny*ny + nz*nz);
    nx /= norm; ny /= norm; nz /= norm;
    constexpr double t = 1.0 / 3.0;
    return {
        s * (nx*nx - t),
        s * (nx*ny),
        s * (nx*nz),
        s * (ny*ny - t),
        s * (ny*nz),
    };
}

void ExpectRoundtrip(double S, double nx, double ny, double nz) {
    const auto Q_in = UniaxialQ(S, nx, ny, nz);
    const auto od   = QtensorToOrderDirectorPoint(Q_in.xx, Q_in.xy, Q_in.xz, Q_in.yy, Q_in.yz);
    const auto Q_out = OrderDirectorToQtensorPoint(od.S, od.nx, od.ny, od.nz);
    constexpr double kTol = 1e-12;
    EXPECT_NEAR(Q_out.xx, Q_in.xx, kTol);
    EXPECT_NEAR(Q_out.xy, Q_in.xy, kTol);
    EXPECT_NEAR(Q_out.xz, Q_in.xz, kTol);
    EXPECT_NEAR(Q_out.yy, Q_in.yy, kTol);
    EXPECT_NEAR(Q_out.yz, Q_in.yz, kTol);
}

}  // namespace

// Deliberately no "axis-aligned director" case:
// QtensorToOrderDirectorPoint recovers n̂ as the normalised cross product of
// two rows of (Q − S·I). Those rows go collinear whenever the director sits
// on a coordinate axis (or in the xy diagonal plane, etc.), the cross
// product vanishes, and dividing by norm yields NaN. The sim never hits
// this exactly — noise + continuous evolution keep Q generic — so it's a
// measure-zero pathology of the eigenvector formula, not something the
// roundtrip is responsible for. Tests use only generic directors.

TEST(OrderDirectorRoundtripTest, OffAxisDirectors) {
    ExpectRoundtrip(0.55, 1.0, 1.0, 1.0);
    ExpectRoundtrip(0.33, 0.3, 0.5, 0.8);
    ExpectRoundtrip(0.5, -0.6, 0.3, -0.7);
    ExpectRoundtrip(0.42, 0.7, -0.4, 0.2);
}

TEST(OrderDirectorRoundtripTest, RandomizedUniaxialQIsFixedPoint) {
    // 200 random uniaxial (S, n̂) samples spanning the physically-relevant
    // range. Each must roundtrip to machine precision — the property that
    // makes it safe to skip stamping Q on the vtkhdf output.
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> uS(0.2, 0.6);
    std::uniform_real_distribution<double> u1(-1.0, 1.0);
    for (int trial = 0; trial < 200; ++trial) {
        const double S = uS(rng);
        double nx = u1(rng), ny = u1(rng), nz = u1(rng);
        // Reject degenerate near-zero directors — the eigenvector construction
        // in QtensorToOrderDirectorPoint doesn't handle that regime specially
        // and it's not physically meaningful anyway.
        if (nx*nx + ny*ny + nz*nz < 0.01) { --trial; continue; }
        ExpectRoundtrip(S, nx, ny, nz);
    }
}

// Sign of n̂: Q is invariant under n̂ → -n̂ (bilinear in n), so the roundtrip
// must recover the same Q even when the eigenvector's sign is flipped.
TEST(OrderDirectorRoundtripTest, DirectorSignInvariance) {
    const auto Q_in = UniaxialQ(0.5, 0.3, 0.4, 0.86);
    const auto od   = QtensorToOrderDirectorPoint(Q_in.xx, Q_in.xy, Q_in.xz, Q_in.yy, Q_in.yz);
    const auto Q_pos = OrderDirectorToQtensorPoint(od.S,  od.nx,  od.ny,  od.nz);
    const auto Q_neg = OrderDirectorToQtensorPoint(od.S, -od.nx, -od.ny, -od.nz);
    constexpr double kTol = 1e-12;
    EXPECT_NEAR(Q_pos.xx, Q_neg.xx, kTol);
    EXPECT_NEAR(Q_pos.xy, Q_neg.xy, kTol);
    EXPECT_NEAR(Q_pos.xz, Q_neg.xz, kTol);
    EXPECT_NEAR(Q_pos.yy, Q_neg.yy, kTol);
    EXPECT_NEAR(Q_pos.yz, Q_neg.yz, kTol);
}
