#include <gtest/gtest.h>
#include "boundary.h"
#include "boundary_handler.h"
#include "analysis_fields.h"
#include <numbers>
#include <vector>
/*
 * AnchoredQMagnitude: HalfTrQ2 == S²/3
 * AnchoredQAlignedAlongZ: at theta=0, phi=0 → xx=yy=-S/3, all off-diag=0
 * QGhostNeumann: returns q_boundary unchanged
 * QGhostAnchoring: returns 2*anchored_comp - q_boundary
 * GradientUniformField: all derivatives zero for constant Q, any BC, including at boundary nodes
 * GradientLinearField: exact gradient 0.01 for qxx(x) = 0.01*x with periodic BC
 * LaplacianQuadratic: ∑ 7-point stencil = 2*scale for qxx(x) = scale*x² (stencil exact for degree-2 polynomials)
 */

TEST(AnchoredQ, AnchoredQMagnitude) {

    constexpr double S = 0.3;
    constexpr double theta = 0.47;
    constexpr double phi = 0.33;
    using BC = Anchoring<S, theta, phi>;
    SymTrLessTensor5 q = AnchoredQ<BC>();
    double exp = S*S/3.0;
    
    double halfTrQ2 = HalfTrQ2(q.xx, q.xy, q.xz, q.yy, q.yz);

    EXPECT_DOUBLE_EQ(halfTrQ2, exp);
}

TEST(AnchoredQ, AnchoredQAlignedAlongZ) {

    constexpr double S = 0.3;
    constexpr double theta = 0.0;
    constexpr double phi = 0.0;
    using BC = Anchoring<S, theta, phi>;
    SymTrLessTensor5 q = AnchoredQ<BC>();

    EXPECT_DOUBLE_EQ(q.xx, -S/3.0);
    EXPECT_DOUBLE_EQ(q.yy, -S/3.0);
    EXPECT_NEAR(q.xy, 0.0, std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(q.xz, 0.0, std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(q.yz, 0.0, std::numeric_limits<double>::epsilon());
}

TEST(AnchoredQ, QGhostNeumann) {

    double q_boundary = 0.1;
    double out = QGhost<QComp::XX, Neumann>(q_boundary);
    EXPECT_DOUBLE_EQ(out, q_boundary);

}

TEST(AnchoredQ, QGhostAnchoring) {
    constexpr double S = 0.3;
    constexpr double theta = 0.47;
    constexpr double phi = 0.33;
    using BC = Anchoring<S, theta, phi>;
    SymTrLessTensor5 q = AnchoredQ<BC>();
    double q_boundary = 0.1;
    double out = QGhost<QComp::XX, BC>(q_boundary);
    double expected = 2 * q.xx - q_boundary;
    EXPECT_DOUBLE_EQ(out, expected);

}

TEST(AnchoredQ, GradientUniformField) {
    constexpr double q0 = 0.42;
    std::vector<double> q(nx * ny * nz, q0);

    for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
        auto d = QGradientAndLaplacian<QComp::XX, FullyPeriodicConfig>(q.data(), x, y, z);
        EXPECT_NEAR(d.dx, 0.0, std::numeric_limits<double>::epsilon()) << "Periodic at (" << x << "," << y << "," << z << ")";
        EXPECT_NEAR(d.dy, 0.0, std::numeric_limits<double>::epsilon()) << "Periodic at (" << x << "," << y << "," << z << ")";
        EXPECT_NEAR(d.dz, 0.0, std::numeric_limits<double>::epsilon()) << "Periodic at (" << x << "," << y << "," << z << ")";
        EXPECT_NEAR(d.lap, 0.0, std::numeric_limits<double>::epsilon()) << "Periodic at (" << x << "," << y << "," << z << ")";
    }

    // Neumann (ChannelConfig): ghost equals boundary value, same result at walls.
    for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
        auto d = QGradientAndLaplacian<QComp::XX, ChannelConfig>(q.data(), x, y, z);
        EXPECT_NEAR(d.dx, 0.0, std::numeric_limits<double>::epsilon()) << "Neumann at (" << x << "," << y << "," << z << ")";
        EXPECT_NEAR(d.dy, 0.0, std::numeric_limits<double>::epsilon()) << "Neumann at (" << x << "," << y << "," << z << ")";
        EXPECT_NEAR(d.dz, 0.0, std::numeric_limits<double>::epsilon()) << "Neumann at (" << x << "," << y << "," << z << ")";
        EXPECT_NEAR(d.lap, 0.0, std::numeric_limits<double>::epsilon()) << "Neumann at (" << x << "," << y << "," << z << ")";
    }
}

TEST(AnchoredQ, GradientLinearField) {
    constexpr double scale = 0.01;
    std::vector<double> q(nx * ny * nz);
    for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x)
        q[idx(x, y, z)] = scale * x;

    // Interior point: central difference is exact for linear functions.
    // Periodic wrapping at edges is irrelevant for this interior query.
    const int x = nx / 2, y = ny / 2, z = nz / 2;
    auto d = QGradientAndLaplacian<QComp::XX, FullyPeriodicConfig>(q.data(), x, y, z);
    EXPECT_DOUBLE_EQ(d.dx, scale);
    EXPECT_NEAR(d.dy, 0.0, std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(d.dz, 0.0, std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(d.lap, 0.0, std::numeric_limits<double>::epsilon());
}

TEST(AnchoredQ, LaplacianQuadratic) {
    constexpr double scale = 0.01;
    std::vector<double> q(nx * ny * nz);
    for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x)
        q[idx(x, y, z)] = scale * x * x;

    // 7-point stencil is exact for degree-2 polynomials: ∇²(scale·x²) = 2·scale.
    const int x = nx / 2, y = ny / 2, z = nz / 2;
    auto d = QGradientAndLaplacian<QComp::XX, FullyPeriodicConfig>(q.data(), x, y, z);
    // 7 FP operations on values ~0.1-0.25 accumulate ~4e-16 cancellation error;
    // EXPECT_DOUBLE_EQ (4 ULPs ≈ 1.4e-17) is too tight here.
    EXPECT_NEAR(d.lap, 2.0 * scale, 1e-14);
}
