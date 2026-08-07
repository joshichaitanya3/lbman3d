#include <gtest/gtest.h>
#include "model.h"
#include <cmath>

/*
 * The Ericksen (distortion) force written by PointwiseStepAndSetupBodyForce is
 * the reactive term -H:∇Q, i.e. -H_ij ∂_k Q_ij summed over ALL NINE index pairs
 * of two symmetric traceless 3x3 tensors. It is the half of the backflow
 * coupling that is not a stress divergence, and carries no velocity — so the
 * kQAdvection scheme does not apply to it.
 *
 * Both H and Q are stored as 5 independent components, with
 * H_zz = -(H_xx + H_yy) and ∂_k Q_zz = -(∂_k Q_xx + ∂_k Q_yy). Expanding the
 * zz pair against the trace condition produces two cross terms:
 *
 *   H:∇_k Q = 2(H_xx ∂Q_xx + H_yy ∂Q_yy + H_xy ∂Q_xy + H_xz ∂Q_xz + H_yz ∂Q_yz)
 *             + H_xx ∂Q_yy + H_yy ∂Q_xx
 *
 * so -H:∇_k Q carries those cross terms NEGATIVE. Dropping the sign on them is
 * the specific regression these tests guard: it yields an expression that is
 * neither +H:∇Q nor -H:∇Q, wrong by 2(H_xx ∂Q_yy + H_yy ∂Q_xx).
 *
 * The same all-positive contraction pattern is written correctly for Q:∇²Q in
 * analysis_fields.cc's TotalNematicFreeEnergy, which is the in-tree cross-check.
 *
 * Tests here build the reference from the nine-pair definition rather than from
 * model.h's grouped algebra, so they cannot be satisfied by copying the code.
 */

using namespace Params;

namespace {

struct SymTrLess5 { double xx, xy, xz, yy, yz; };

// Landau-de Gennes molecular field, matching PointwiseStepAndSetupBodyForce's
// H = L∇²Q - (A + C TrQ²)Q - B (Q² - TrQ²/3 I).
SymTrLess5 MolecularField(const SymTrLess5& Q, const SymTrLess5& lap) {
    const double TrQ2 = 2.0 * (Q.xx*Q.xx + Q.yy*Q.yy + Q.xx*Q.yy
                               + Q.xy*Q.xy + Q.xz*Q.xz + Q.yz*Q.yz);
    const double third = 1.0 / 3.0;
    const double Q2_xx = Q.xx*Q.xx + Q.xy*Q.xy + Q.xz*Q.xz - third * TrQ2;
    const double Q2_xy = Q.xx*Q.xy + Q.xy*Q.yy + Q.xz*Q.yz;
    const double Q2_xz = Q.xy*Q.yz - Q.xz*Q.yy;
    const double Q2_yy = Q.xy*Q.xy + Q.yy*Q.yy + Q.yz*Q.yz - third * TrQ2;
    const double Q2_yz = Q.xy*Q.xz - Q.yz*Q.xx;
    const double ld = A + C * TrQ2;
    return { L*lap.xx - ld*Q.xx - B*Q2_xx,
             L*lap.xy - ld*Q.xy - B*Q2_xy,
             L*lap.xz - ld*Q.xz - B*Q2_xz,
             L*lap.yy - ld*Q.yy - B*Q2_yy,
             L*lap.yz - ld*Q.yz - B*Q2_yz };
}

// -H_ij ∂_k Q_ij over all nine index pairs, zz reconstructed from the trace
// condition. This is the definition, independent of model.h's grouping.
double MinusHContractDQ(const SymTrLess5& H, const SymTrLess5& dQ) {
    const double Hzz  = -(H.xx + H.yy);
    const double dQzz = -(dQ.xx + dQ.yy);
    const double full = H.xx*dQ.xx + H.yy*dQ.yy + Hzz*dQzz
                      + 2.0 * (H.xy*dQ.xy + H.xz*dQ.xz + H.yz*dQ.yz);
    return -full;
}

// Invoke the production pointwise kernel with a fully specified stencil.
// Velocity and velocity gradients are zero: they do not enter the Ericksen force.
Vec3 CodeBackflow(const SymTrLess5& Q, const SymTrLess5& lap,
                  const SymTrLess5& gx, const SymTrLess5& gy, const SymTrLess5& gz) {
    QStencil qs{};
    qs.Q = SymTrLessTensor5{ Q.xx, Q.xy, Q.xz, Q.yy, Q.yz };
    qs.u = Vec3{ 0.0, 0.0, 0.0 };
    // The per-axis second differences are only read by AdvectiveAxisTerm, which
    // these tests do not exercise (no advection under test), so they are left
    // at zero rather than made consistent with lap.
    qs.dQxx = QDerivs{ gx.xx, gy.xx, gz.xx, lap.xx, 0.0, 0.0, 0.0 };
    qs.dQxy = QDerivs{ gx.xy, gy.xy, gz.xy, lap.xy, 0.0, 0.0, 0.0 };
    qs.dQxz = QDerivs{ gx.xz, gy.xz, gz.xz, lap.xz, 0.0, 0.0, 0.0 };
    qs.dQyy = QDerivs{ gx.yy, gy.yy, gz.yy, lap.yy, 0.0, 0.0, 0.0 };
    qs.dQyz = QDerivs{ gx.yz, gy.yz, gz.yz, lap.yz, 0.0, 0.0, 0.0 };
    qs.gradu = GradTensor{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    SymTrLessTensor5 q_new{};
    SymTrLessTensor5 sigma{};
    AntiSymTensor3 tau{};
    Vec3 ericksen{};
    PointwiseStepAndSetupBodyForce(qs, q_new, sigma, tau, ericksen);
    return ericksen;
}

} // namespace

// The minimal decisive case. With Q = 0 the molecular field reduces to L∇²Q,
// so setting lap_xx = 1/L gives exactly H_xx = 1 with every other component
// zero. A single nonzero gradient ∂_x Q_yy = 1 then isolates one cross term:
//
//   H_zz = -1, ∂_x Q_zz = -1  =>  H:∇_x Q = H_zz ∂_x Q_zz = +1  =>  -H:∇_x Q = -1
//
// The unfixed expression returns +H_xx ∂_x Q_yy = +1: right magnitude, wrong sign.
TEST(BackflowContraction, CrossTermSignCounterexample) {
    const SymTrLess5 Q{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    const SymTrLess5 lap{ 1.0 / L, 0.0, 0.0, 0.0, 0.0 };   // => H_xx = 1
    const SymTrLess5 gx{ 0.0, 0.0, 0.0, 1.0, 0.0 };        // => d_x Q_yy = 1
    const SymTrLess5 zero{ 0.0, 0.0, 0.0, 0.0, 0.0 };

    const SymTrLess5 H = MolecularField(Q, lap);
    ASSERT_NEAR(H.xx, 1.0, 1e-12) << "test setup: H_xx should be exactly 1";
    ASSERT_NEAR(H.yy, 0.0, 1e-12) << "test setup: H_yy should vanish";

    const Vec3 bf = CodeBackflow(Q, lap, gx, zero, zero);

    EXPECT_NEAR(bf.x, -1.0, 1e-12);
    EXPECT_NEAR(bf.x, MinusHContractDQ(H, gx), 1e-12);
}

// The mirrored cross term: H_yy against d_x Q_xx.
TEST(BackflowContraction, MirroredCrossTermSignCounterexample) {
    const SymTrLess5 Q{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    const SymTrLess5 lap{ 0.0, 0.0, 0.0, 1.0 / L, 0.0 };   // => H_yy = 1
    const SymTrLess5 gx{ 1.0, 0.0, 0.0, 0.0, 0.0 };        // => d_x Q_xx = 1
    const SymTrLess5 zero{ 0.0, 0.0, 0.0, 0.0, 0.0 };

    const SymTrLess5 H = MolecularField(Q, lap);
    ASSERT_NEAR(H.yy, 1.0, 1e-12) << "test setup: H_yy should be exactly 1";

    const Vec3 bf = CodeBackflow(Q, lap, gx, zero, zero);

    EXPECT_NEAR(bf.x, -1.0, 1e-12);
    EXPECT_NEAR(bf.x, MinusHContractDQ(H, gx), 1e-12);
}

// General case: nonzero Q (so the bulk LdG terms contribute to H) and all five
// gradient components nonzero in all three directions. Guards against a fix
// that happens to satisfy the isolated counterexamples above.
TEST(BackflowContraction, MatchesNineComponentContractionGeneralQ) {
    const SymTrLess5 Q  {  0.31, -0.07,  0.11, -0.18,  0.04 };
    const SymTrLess5 lap{  0.13,  0.05, -0.09,  0.21, -0.03 };
    const SymTrLess5 gx {  0.17, -0.23,  0.31,  0.11, -0.05 };
    const SymTrLess5 gy { -0.29,  0.07,  0.13, -0.19,  0.23 };
    const SymTrLess5 gz {  0.03,  0.19, -0.27,  0.09,  0.15 };

    const SymTrLess5 H = MolecularField(Q, lap);
    const Vec3 bf = CodeBackflow(Q, lap, gx, gy, gz);

    EXPECT_NEAR(bf.x, MinusHContractDQ(H, gx), 1e-12);
    EXPECT_NEAR(bf.y, MinusHContractDQ(H, gy), 1e-12);
    EXPECT_NEAR(bf.z, MinusHContractDQ(H, gz), 1e-12);
}

// A vanishing molecular field must give zero Ericksen force regardless of gradients.
// With Q = 0 and A = 0 the bulk terms vanish identically, so H = L∇²Q = 0.
TEST(BackflowContraction, VanishesForZeroMolecularField) {
    const SymTrLess5 Q{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    const SymTrLess5 lap{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    const SymTrLess5 gx{  0.17, -0.23,  0.31,  0.11, -0.05 };
    const SymTrLess5 gy{ -0.29,  0.07,  0.13, -0.19,  0.23 };
    const SymTrLess5 gz{  0.03,  0.19, -0.27,  0.09,  0.15 };

    const SymTrLess5 H = MolecularField(Q, lap);
    ASSERT_NEAR(H.xx, 0.0, 1e-12);
    ASSERT_NEAR(H.yy, 0.0, 1e-12);

    const Vec3 bf = CodeBackflow(Q, lap, gx, gy, gz);

    EXPECT_NEAR(bf.x, 0.0, 1e-12);
    EXPECT_NEAR(bf.y, 0.0, 1e-12);
    EXPECT_NEAR(bf.z, 0.0, 1e-12);
}

// The contraction is linear in ∇Q at fixed H (H depends only on Q and ∇²Q,
// which are held fixed here). Scaling every gradient must scale the force.
TEST(BackflowContraction, LinearInGradientsAtFixedH) {
    const SymTrLess5 Q  {  0.31, -0.07,  0.11, -0.18,  0.04 };
    const SymTrLess5 lap{  0.13,  0.05, -0.09,  0.21, -0.03 };
    const SymTrLess5 gx {  0.17, -0.23,  0.31,  0.11, -0.05 };
    const SymTrLess5 zero{ 0.0, 0.0, 0.0, 0.0, 0.0 };

    const double s = 2.5;
    const SymTrLess5 gx_scaled{ s*gx.xx, s*gx.xy, s*gx.xz, s*gx.yy, s*gx.yz };

    const Vec3 bf       = CodeBackflow(Q, lap, gx,        zero, zero);
    const Vec3 bf_scaled = CodeBackflow(Q, lap, gx_scaled, zero, zero);

    EXPECT_NEAR(bf_scaled.x, s * bf.x, 1e-12);
}
