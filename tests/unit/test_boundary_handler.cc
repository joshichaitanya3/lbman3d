#include <gtest/gtest.h>
#include "params.h"
#include "boundary_handler.h"

using namespace Params;

// ── StreamWallOffset ─────────────────────────────────────────────────────────

TEST(BoundaryHandler, StreamWallOffsetPeriodic) {
    EXPECT_EQ(StreamWallOffset<Periodic>(0, -1, nx), nx - 1);
    EXPECT_EQ(StreamWallOffset<Periodic>(nx - 1, +1, nx), 0);
}

TEST(BoundaryHandler, StreamWallOffsetNoSlip) {
    // NoSlip returns the raw (possibly out-of-domain) offset; callers rely
    // on InDomain to detect this and fall back to HandleBoundaryPoint.
    int lo = StreamWallOffset<NoSlip>(0, -1, nx);
    int hi = StreamWallOffset<NoSlip>(nx - 1, +1, nx);
    EXPECT_FALSE(InDomain(lo, 0, 0));
    EXPECT_FALSE(InDomain(hi, 0, 0));
}

// ── VelocityGhost ─────────────────────────────────────────────────────────────

TEST(BoundaryHandler, VelocityGhostNoSlip) {
    double v_boundary = 0.3;
    // Wall velocity is 0, so mid-point bounce-back gives ghost = -v_boundary
    // regardless of is_normal (NoSlip applies Dirichlet to every component).
    double ghost_normal = VelocityGhost<Axis::X, NoSlip>(v_boundary, true);
    double ghost_tangential = VelocityGhost<Axis::X, NoSlip>(v_boundary, false);
    EXPECT_DOUBLE_EQ(ghost_normal, -v_boundary);
    EXPECT_DOUBLE_EQ(ghost_tangential, -v_boundary);
}

TEST(BoundaryHandler, VelocityGhostMovingWall) {
    double v_boundary = 0.3;
    using Lid = MovingWall<0.2, 0.0, 0.0>;
    double ghost = VelocityGhost<Axis::X, Lid>(v_boundary, true);
    EXPECT_DOUBLE_EQ(ghost, 2.0 * 0.2 - v_boundary);
}

TEST(BoundaryHandler, VelocityGhostSpecularNormal) {
    double v_boundary = 0.3;
    // Normal component: no-penetration (Dirichlet 0), same formula as NoSlip.
    double ghost = VelocityGhost<Axis::X, SpecularReflection>(v_boundary, true);
    EXPECT_DOUBLE_EQ(ghost, -v_boundary);
}

TEST(BoundaryHandler, VelocityGhostSpecularTangential) {
    double v_boundary = 0.3;
    // Tangential component: zero-gradient, ghost equals the boundary value.
    double ghost = VelocityGhost<Axis::X, SpecularReflection>(v_boundary, false);
    EXPECT_DOUBLE_EQ(ghost, v_boundary);
}

// ── VelocityAxisGhostPair ────────────────────────────────────────────────────

TEST(BoundaryHandler, InteriorGhostPairPassthrough) {
    int n = nx;
    int i = n / 2; // away from both walls
    double v_minus = 1.1, v_center = 2.2, v_plus = 3.3;
    NeighborPair pair = VelocityAxisGhostPair<Axis::X, NoSlip, NoSlip>(
        i, n, v_minus, v_center, v_plus, true
    );
    EXPECT_DOUBLE_EQ(pair.minus, v_minus);
    EXPECT_DOUBLE_EQ(pair.plus, v_plus);
}

// ── VelocityAxisGradient ──────────────────────────────────────────────────────

TEST(BoundaryHandler, CentralDifferenceInterior) {
    // Linear profile ux(x) = 0.01*x; central difference is exact for a
    // linear function, and the BC tags are never consulted away from a wall.
    int n = nx;
    int i = n / 2;
    double v_minus  = 0.01 * (i - 1);
    double v_center = 0.01 * i;
    double v_plus   = 0.01 * (i + 1);
    double grad = VelocityAxisGradient<Axis::X, NoSlip, NoSlip>(i, n, v_minus, v_center, v_plus, true);
    EXPECT_DOUBLE_EQ(grad, 0.01);
}

TEST(BoundaryHandler, GradientAtNoSlipWall) {
    // At x=0 with a NoSlip Lo wall, the ghost is -v_center (wall velocity
    // 0, mid-point bounce-back), so the analytical gradient is
    // (v_plus - ghost) / 2 == (v_plus + v_center) / 2.
    double v_center = 0.3;
    double v_plus   = 0.5;
    double grad = VelocityAxisGradient<Axis::X, NoSlip, NoSlip>(0, nx, /*v_minus unused*/ 0.0, v_center, v_plus, true);
    EXPECT_DOUBLE_EQ(grad, (v_plus + v_center) / 2.0);
}
