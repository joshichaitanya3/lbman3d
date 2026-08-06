#include <gtest/gtest.h>
#include "model.h"
#include "boundary.h"
#include "local_grid.h"
#include "qtensor_fields.h"
#include "qtensor_solver.h"
#include <cmath>

/*
 * QTensorSolver::Initialize seeds a uniform director along z with amplitude
 * EquilibriumScalarOrder(). These tests pin that the seeded state is a genuine
 * fixed point of the bulk Q dynamics, so a run starts in nematic equilibrium
 * and any subsequent evolution comes from gradients, flow or activity.
 *
 * The check is done through PointwiseStepAndSetupBodyForce rather than by
 * calling Initialize: with zero gradients, zero Laplacian and zero velocity,
 * every advective, co-rotational and strain-alignment term drops out and the
 * only surviving contribution to q_new - Q is DT * GAMMA * H_bulk. So
 * "q_new == Q" is exactly the statement "the bulk molecular field vanishes".
 */

using namespace Params;

namespace {

// Q = S (nn - I/3) for n along a coordinate axis. The stored diagonal is
// (xx, yy); zz is implicit as -(xx + yy).
SymTrLessTensor5 UniaxialAlongX(double s) {
    return SymTrLessTensor5{2.0 * s / 3.0, 0.0, 0.0, -s / 3.0, 0.0};
}
SymTrLessTensor5 UniaxialAlongY(double s) {
    return SymTrLessTensor5{-s / 3.0, 0.0, 0.0, 2.0 * s / 3.0, 0.0};
}
// The uniform part of the initial condition (QTensorSolver::Initialize).
SymTrLessTensor5 UniaxialAlongZ(double s) {
    return SymTrLessTensor5{-s / 3.0, 0.0, 0.0, -s / 3.0, 0.0};
}

// Quiescent stencil: no gradients, no Laplacian, no flow. Value-initialisation
// zeroes every member of QStencil (all plain double aggregates).
QStencil Quiescent(const SymTrLessTensor5& q) {
    QStencil qs{};
    qs.Q = q;
    return qs;
}

// One bulk-only Beris-Edwards update.
SymTrLessTensor5 BulkStep(const SymTrLessTensor5& q) {
    SymTrLessTensor5 q_new{}, sigma{};
    AntiSymTensor3 tau{};
    Vec3 backflow{};
    PointwiseStepAndSetupBodyForce(Quiescent(q), q_new, sigma, tau, backflow);
    return q_new;
}

// Recover S from the uniaxial-along-z form, where qzz = -(qxx + qyy) = 2S/3.
double OrderFromQzz(const SymTrLessTensor5& q) { return -1.5 * (q.xx + q.yy); }

} // namespace

// EquilibriumScalarOrder must satisfy the polynomial it was derived from,
// 2 C S^2 + B S + 3 A = 0. Checked against the polynomial rather than against
// the closed form so the general A != 0 branch is covered too.
TEST(QTensorInit, EquilibriumSSatisfiesBulkPolynomial) {
    const double s = EquilibriumScalarOrder();
    EXPECT_NEAR(2.0 * C * s * s + B * s + 3.0 * A, 0.0, 1e-14);
    EXPECT_GT(s, 0.0) << "the ordered root must be positive";
}

TEST(QTensorInit, EquilibriumSMatchesClosedFormWhenAIsZero) {
    ASSERT_DOUBLE_EQ(A, 0.0) << "this params.h is expected to set A = 0";
    EXPECT_NEAR(EquilibriumScalarOrder(), -B / (2.0 * C), 1e-14);
}

// The seeded state does not move under the bulk dynamics. Checked for all three
// axes: the bulk free energy is isotropic, so the fixed point cannot depend on
// which direction the director points, and the solver's choice of z must not be
// load-bearing for equilibrium.
TEST(QTensorInit, InitialQIsABulkFixedPointOnEveryAxis) {
    const double s_eq = EquilibriumScalarOrder();
    const struct { const char* axis; SymTrLessTensor5 q; } cases[] = {
        {"x", UniaxialAlongX(s_eq)},
        {"y", UniaxialAlongY(s_eq)},
        {"z", UniaxialAlongZ(s_eq)},
    };

    for (const auto& c : cases) {
        const SymTrLessTensor5 q_new = BulkStep(c.q);
        EXPECT_NEAR(q_new.xx, c.q.xx, 1e-15) << "axis " << c.axis;
        EXPECT_NEAR(q_new.yy, c.q.yy, 1e-15) << "axis " << c.axis;
        EXPECT_NEAR(q_new.xy, 0.0, 1e-15) << "axis " << c.axis;
        EXPECT_NEAR(q_new.xz, 0.0, 1e-15) << "axis " << c.axis;
        EXPECT_NEAR(q_new.yz, 0.0, 1e-15) << "axis " << c.axis;
    }
}

// Discriminating guard: without this, the test above would also pass if DT or
// GAMMA were zero, or if the bulk field were dropped entirely.
TEST(QTensorInit, OffEquilibriumOrderIsNotAFixedPoint) {
    const double s_eq = EquilibriumScalarOrder();
    const SymTrLessTensor5 q = UniaxialAlongZ(2.0 * s_eq);
    const SymTrLessTensor5 q_new = BulkStep(q);

    // n = z_hat, so both stored diagonal components sit at -S/3 and must rise
    // toward -S_eq/3 as the implicit qzz = -(qxx + qyy) relaxes downward.
    EXPECT_GT(q_new.xx, q.xx + 1e-9) << "over-ordered Q must relax toward S_eq";
    EXPECT_GT(q_new.yy, q.yy + 1e-9);
    EXPECT_LT(OrderFromQzz(q_new), OrderFromQzz(q) - 1e-9);
}

// An over-ordered start relaxes to S_eq, which is what the previous hardcoded
// qxx = 0.66 (S ~ 0.99) spent the first few hundred steps of every run doing.
TEST(QTensorInit, OverOrderedStateRelaxesToEquilibriumS) {
    const double s_eq = EquilibriumScalarOrder();
    SymTrLessTensor5 q = UniaxialAlongZ(2.0 * s_eq);

    double prev = OrderFromQzz(q);
    for (int i = 0; i < 200000; ++i) {
        q = BulkStep(q);
        const double s = OrderFromQzz(q);
        EXPECT_LE(s, prev + 1e-12) << "relaxation must be monotone at step " << i;
        prev = s;
    }

    EXPECT_NEAR(OrderFromQzz(q), s_eq, 1e-6);
    // Uniaxiality along z is preserved throughout: qxx and qyy stay degenerate.
    EXPECT_NEAR(q.xx, -s_eq / 3.0, 1e-6);
    EXPECT_NEAR(q.yy, -s_eq / 3.0, 1e-6);
    EXPECT_NEAR(q.xy, 0.0, 1e-15);
}

// Guards the seeded values in QTensorSolver::Initialize itself, so that
// re-hardcoding a literal amplitude is caught rather than only the
// EquilibriumScalarOrder() helper being checked.
TEST(QTensorInit, InitializeSeedsEquilibriumOrderParameter) {
    QTensorFields qf{LocalGrid::SingleRank()};
    QTensorSolver<FullyPeriodicConfig> solver;
    solver.Initialize(qf);

    const double s_eq  = EquilibriumScalarOrder();
    const double qxx_0 = -s_eq / 3.0;   // director along z
    const double qyy_0 = -s_eq / 3.0;
    const LocalGrid& g = qf.grid;

    double sxx = 0, syy = 0, sxy = 0, sxz = 0, syz = 0;
    int n = 0;
    for (int z = 0; z < g.local_nz; ++z) {
        for (int y = 0; y < g.local_ny; ++y) {
            for (int x = 0; x < g.local_nx; ++x) {
                const int i = g.halo_idx(x, y, z);
                // Per-cell bounds are exact: the noise is drawn from
                // [-NOISE, NOISE] about the uniform part, so any shift of the
                // seeded amplitude larger than NOISE fails deterministically.
                ASSERT_GE(qf.qxx[i], qxx_0 - NOISE);
                ASSERT_LE(qf.qxx[i], qxx_0 + NOISE);
                ASSERT_GE(qf.qyy[i], qyy_0 - NOISE);
                ASSERT_LE(qf.qyy[i], qyy_0 + NOISE);
                ASSERT_LE(std::abs(qf.qxy[i]), NOISE);
                ASSERT_LE(std::abs(qf.qxz[i]), NOISE);
                ASSERT_LE(std::abs(qf.qyz[i]), NOISE);

                sxx += qf.qxx[i]; syy += qf.qyy[i]; sxy += qf.qxy[i];
                sxz += qf.qxz[i]; syz += qf.qyz[i];
                ++n;
            }
        }
    }
    ASSERT_EQ(n, nx * ny * nz);

    // The noise is zero-mean, so the domain average recovers the uniform part.
    // Tolerance is 4 sigma of the mean for a uniform draw on [-NOISE, NOISE].
    const double tol = 4.0 * (NOISE / std::sqrt(3.0)) / std::sqrt(1.0 * n);
    EXPECT_NEAR(sxx / n, qxx_0, tol);
    EXPECT_NEAR(syy / n, qyy_0, tol);
    EXPECT_NEAR(sxy / n, 0.0, tol);
    EXPECT_NEAR(sxz / n, 0.0, tol);
    EXPECT_NEAR(syz / n, 0.0, tol);
}
