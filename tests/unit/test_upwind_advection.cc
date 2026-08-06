#include <gtest/gtest.h>
#include "model.h"
#include "boundary.h"
#include "local_grid.h"
#include <cmath>
#include <vector>

/*
 * Upwind advection for the Q equation.
 *
 * AdvectiveAxisTerm<Scheme>(u, dQ, d2Q) returns one axis contribution to
 * u.grad(Q). The upwind branch is built on the identities
 *
 *     Q(a) - Q(a-1) = dQ - d2Q/2        (backward, used when u > 0)
 *     Q(a+1) - Q(a) = dQ + d2Q/2        (forward,  used when u < 0)
 *
 * so these tests build the reference from the raw one-sided differences of an
 * explicit three-point stencil, never from the code's dQ/d2Q algebra. That way
 * they cannot be satisfied by copying the implementation.
 *
 * Both template branches are exercised from this one binary, independently of
 * whatever kQAdvection in sim_config.h happens to be set to.
 */

using namespace Params;

namespace {

// A three-point stencil in one direction, and the derivative quantities the
// boundary handler would hand to model.h for it.
struct Stencil3 {
    double minus, centre, plus;
    double dQ()  const { return (plus - minus) / 2.0; }
    double d2Q() const { return (minus + plus) - 2.0 * centre; }
    double backward() const { return centre - minus; }   // u > 0
    double forward()  const { return plus - centre; }    // u < 0
};

} // namespace

// The centred branch must ignore d2Q entirely.
TEST(UpwindAdvection, CentredBranchIsPlainCentredDifference) {
    const Stencil3 s{1.0, 4.0, 9.0};   // deliberately non-linear, so d2Q != 0
    ASSERT_NE(s.d2Q(), 0.0);

    for (double u : {-2.0, -0.5, 0.0, 0.5, 2.0})
        EXPECT_DOUBLE_EQ(AdvectiveAxisTerm<Advection::Centred>(u, s.dQ(), s.d2Q()), u * s.dQ());
}

// Positive velocity must pick the backward difference, negative the forward.
TEST(UpwindAdvection, UpwindPicksTheDifferenceTheFlowComesFrom) {
    const Stencil3 s{1.0, 4.0, 9.0};

    for (double u : {0.25, 1.0, 3.0})
        EXPECT_DOUBLE_EQ(AdvectiveAxisTerm<Advection::Upwind>(u, s.dQ(), s.d2Q()), u * s.backward())
            << "u = " << u << " should use the upstream (backward) difference";

    for (double u : {-0.25, -1.0, -3.0})
        EXPECT_DOUBLE_EQ(AdvectiveAxisTerm<Advection::Upwind>(u, s.dQ(), s.d2Q()), u * s.forward())
            << "u = " << u << " should use the upstream (forward) difference";
}

// At u = 0 there is no upstream side and no transport either way.
TEST(UpwindAdvection, VanishesAtZeroVelocity) {
    const Stencil3 s{1.0, 4.0, 9.0};
    EXPECT_DOUBLE_EQ(AdvectiveAxisTerm<Advection::Upwind>(0.0, s.dQ(), s.d2Q()), 0.0);
    EXPECT_DOUBLE_EQ(AdvectiveAxisTerm<Advection::Centred>(0.0, s.dQ(), s.d2Q()), 0.0);
}

// On a locally linear profile the second difference vanishes, so upwind and
// centred agree exactly — first-order upwinding is still exact for linear data.
TEST(UpwindAdvection, AgreesWithCentredOnLinearProfiles) {
    const Stencil3 s{2.0, 5.0, 8.0};   // constant slope 3
    ASSERT_DOUBLE_EQ(s.d2Q(), 0.0);

    for (double u : {-1.5, -0.3, 0.3, 1.5})
        EXPECT_DOUBLE_EQ(AdvectiveAxisTerm<Advection::Upwind>(u, s.dQ(), s.d2Q()),
                         AdvectiveAxisTerm<Advection::Centred>(u, s.dQ(), s.d2Q()));
}

// The upwind correction is exactly a diffusion term of coefficient |u|/2. This
// is the cost of the scheme, so pin it explicitly rather than leave it implied.
TEST(UpwindAdvection, CorrectionIsDiffusionOfCoefficientHalfAbsU) {
    const Stencil3 s{1.0, 4.0, 9.0};
    for (double u : {-2.0, -0.5, 0.5, 2.0}) {
        const double correction = AdvectiveAxisTerm<Advection::Centred>(u, s.dQ(), s.d2Q())
                                - AdvectiveAxisTerm<Advection::Upwind>(u, s.dQ(), s.d2Q());
        EXPECT_DOUBLE_EQ(correction, 0.5 * std::abs(u) * s.d2Q());
    }
}

// Reflection covariance. Under a -> -a the first derivative flips sign and the
// second difference does not, so with u -> -u the flux u * dQ/da is invariant.
// This pins that upwinding selects the upstream side in a frame-independent
// way: a scheme that always took the backward difference would fail here.
TEST(UpwindAdvection, IsInvariantUnderSimultaneousFlipOfFlowAndStencil) {
    const Stencil3 s{1.0, 4.0, 9.0};
    const Stencil3 m{s.plus, s.centre, s.minus};   // mirrored
    ASSERT_DOUBLE_EQ(m.dQ(), -s.dQ());
    ASSERT_DOUBLE_EQ(m.d2Q(), s.d2Q());

    for (double u : {0.4, 1.3, 2.8})
        EXPECT_DOUBLE_EQ(AdvectiveAxisTerm<Advection::Upwind>(u, s.dQ(), s.d2Q()),
                         AdvectiveAxisTerm<Advection::Upwind>(-u, m.dQ(), m.d2Q())) << "u = " << u;
}

// AdvectiveDerivative is what model.h actually calls. It must sum all three
// axes AND pair each velocity component with that same axis's derivatives —
// the pairing is the whole reason it takes Vec3 + QDerivs instead of six
// interchangeable doubles.
TEST(UpwindAdvection, AdvectiveDerivativeSumsThreeAxesWithMatchedPairing) {
    // Distinct curvature per axis, and mixed velocity signs so the upwind
    // branch selects a different side on each.
    const Stencil3 sx{1.0, 4.0, 9.0};
    const Stencil3 sy{0.0, 1.0, 5.0};
    const Stencil3 sz{2.0, 2.0, -3.0};
    const Vec3 u{2.0, -3.0, 5.0};
    const QDerivs d{sx.dQ(), sy.dQ(), sz.dQ(), 0.0, sx.d2Q(), sy.d2Q(), sz.d2Q()};

    // Reference built from raw one-sided differences, not from the helper.
    EXPECT_DOUBLE_EQ(AdvectiveDerivative<Advection::Upwind>(u, d),
                     u.x * sx.backward()     // u.x > 0 -> upstream is behind
                   + u.y * sy.forward()      // u.y < 0 -> upstream is ahead
                   + u.z * sz.backward());

    EXPECT_DOUBLE_EQ(AdvectiveDerivative<Advection::Centred>(u, d),
                     u.x * sx.dQ() + u.y * sy.dQ() + u.z * sz.dQ());

    // Transposing two axes' derivatives must change the answer, so the test
    // above constrains the pairing rather than merely the sum.
    const QDerivs swapped{sy.dQ(), sx.dQ(), sz.dQ(), 0.0, sy.d2Q(), sx.d2Q(), sz.d2Q()};
    EXPECT_NE(AdvectiveDerivative<Advection::Upwind>(u, swapped),
              AdvectiveDerivative<Advection::Upwind>(u, d));
}

// The default template argument must follow sim_config.h's kQAdvection, so the
// shipped configuration is what the solver actually runs.
TEST(UpwindAdvection, DefaultSchemeFollowsSimConfig) {
    const Stencil3 s{1.0, 4.0, 9.0};
    const Vec3 u{1.5, 0.0, 0.0};
    const QDerivs d{s.dQ(), 0.0, 0.0, 0.0, s.d2Q(), 0.0, 0.0};

    EXPECT_DOUBLE_EQ(AdvectiveDerivative(u, d),
                     AdvectiveDerivative<kQAdvection>(u, d));
}

// QGradientAndLaplacian must fill the per-axis second differences consistently
// with the Laplacian it already reported, on a real field with real BCs.
TEST(UpwindAdvection, GradientHelperFillsPerAxisSecondDifferences) {
    const LocalGrid g = LocalGrid::SingleRank();
    std::vector<double> q(static_cast<size_t>(g.HaloVolume()), 0.0);

    // A field with independent, non-trivial curvature along each axis.
    for (int z = 0; z < g.local_nz; ++z)
        for (int y = 0; y < g.local_ny; ++y)
            for (int x = 0; x < g.local_nx; ++x)
                q[static_cast<size_t>(g.halo_idx(x, y, z))] =
                    1.0 * x * x + 2.0 * y * y + 3.0 * z * z;

    // Interior point, so every neighbour is in-domain and no BC is involved.
    const int xi = g.local_nx / 2, yi = g.local_ny / 2, zi = g.local_nz / 2;
    const QDerivs d =
        QGradientAndLaplacian<QComp::XX, FullyPeriodicConfig>(q.data(), xi, yi, zi, g);

    // d2 of a*i^2 is exactly 2a.
    EXPECT_NEAR(d.d2x, 2.0, 1e-12);
    EXPECT_NEAR(d.d2y, 4.0, 1e-12);
    EXPECT_NEAR(d.d2z, 6.0, 1e-12);
    EXPECT_NEAR(d.lap, d.d2x + d.d2y + d.d2z, 1e-12);

    // And the first derivatives are unchanged by the new members.
    EXPECT_NEAR(d.dx, 2.0 * xi, 1e-12);
    EXPECT_NEAR(d.dy, 4.0 * yi, 1e-12);
    EXPECT_NEAR(d.dz, 6.0 * zi, 1e-12);
}

// The Laplacian member must stay bit-identical to the single six-neighbour sum,
// so enabling this feature cannot perturb existing centred-scheme results.
TEST(UpwindAdvection, LaplacianIsUnchangedBitwise) {
    const LocalGrid g = LocalGrid::SingleRank();
    std::vector<double> q(static_cast<size_t>(g.HaloVolume()), 0.0);
    for (int z = 0; z < g.local_nz; ++z)
        for (int y = 0; y < g.local_ny; ++y)
            for (int x = 0; x < g.local_nx; ++x)
                q[static_cast<size_t>(g.halo_idx(x, y, z))] =
                    std::sin(0.7 * x) * std::cos(0.3 * y) + 0.11 * z * z * z;

    for (int z = 0; z < g.local_nz; ++z) {
        for (int y = 0; y < g.local_ny; ++y) {
            for (int x = 0; x < g.local_nx; ++x) {
                const QDerivs d =
                    QGradientAndLaplacian<QComp::XX, FullyPeriodicConfig>(q.data(), x, y, z, g);
                const double q0 = q[static_cast<size_t>(g.halo_idx(x, y, z))];
                const int xm = (x - 1 + nx) % nx, xp = (x + 1) % nx;
                const int ym = (y - 1 + ny) % ny, yp = (y + 1) % ny;
                const int zm = (z - 1 + nz) % nz, zp = (z + 1) % nz;
                const double expected =
                    (q[static_cast<size_t>(g.halo_idx(xm, y, z))]
                   + q[static_cast<size_t>(g.halo_idx(xp, y, z))]
                   + q[static_cast<size_t>(g.halo_idx(x, ym, z))]
                   + q[static_cast<size_t>(g.halo_idx(x, yp, z))]
                   + q[static_cast<size_t>(g.halo_idx(x, y, zm))]
                   + q[static_cast<size_t>(g.halo_idx(x, y, zp))]) - 6.0 * q0;
                ASSERT_DOUBLE_EQ(d.lap, expected) << "at (" << x << "," << y << "," << z << ")";
            }
        }
    }
}
