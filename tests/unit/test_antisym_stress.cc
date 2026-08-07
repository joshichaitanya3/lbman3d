#include <gtest/gtest.h>
#include "model.h"
#include "boundary_handler.h"
#include "sim_config.h"
#include "local_grid.h"
#include <array>
#include <vector>

/*
 * The nematic stress is NOT symmetric. It splits as Pi = Sigma + Tau, with
 *
 *   S = -(2/3) lambda H - lambda (QH + HQ)     symmetric traceless
 *   A = QH - HQ                                antisymmetric, torque-carrying
 *
 * A is antisymmetric, so τ_yx = -τ_xy, and the momentum equation uses the row
 * convention f_alpha = d_beta Pi_alpha,beta. Two consequences these tests pin
 * down:
 *
 *  1. A must not be summed into the 5-slot symmetric container: doing so loses
 *     the lower-triangle signs. (PointwiseStepAndSetupBodyForce returns it in a
 *     separate AntiSymTensor3.)
 *  2. In the divergence, A enters fx with a PLUS sign but fy and fz with a MINUS
 *     sign in the terms that reference the lower triangle. Before this was fixed,
 *     A was folded in with a plus everywhere, which left fx correct and flipped
 *     the torque-carrying contribution in fy and fz.
 *
 * References are built from full 3x3 matrix products, not from the code's
 * component-wise algebra, so they cannot be satisfied by copying the code.
 */

using namespace Params;

namespace {

struct SymTrLess5 { double xx, xy, xz, yy, yz; };
using Mat3 = std::array<std::array<double, 3>, 3>;

// Expand the 5 stored components into the full symmetric traceless matrix.
Mat3 Full(const SymTrLess5& t) {
    return Mat3{{ {t.xx, t.xy, t.xz},
                  {t.xy, t.yy, t.yz},
                  {t.xz, t.yz, -(t.xx + t.yy)} }};
}

Mat3 MatMul(const Mat3& a, const Mat3& b) {
    Mat3 c{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += a[i][k] * b[k][j];
            c[i][j] = s;
        }
    return c;
}

// Landau-de Gennes molecular field, matching PointwiseStepAndSetupBodyForce.
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

struct PointwiseOut {
    SymTrLessTensor5 sym;
    AntiSymTensor3   antisym;
};

PointwiseOut RunPointwise(const SymTrLess5& Q, const SymTrLess5& lap) {
    QStencil qs{};
    qs.Q = SymTrLessTensor5{ Q.xx, Q.xy, Q.xz, Q.yy, Q.yz };
    qs.u = Vec3{ 0.0, 0.0, 0.0 };
    // Gradients do not enter the stress; only the Laplacians do, through H.
    // Second differences unused here: AdvectiveAxisTerm is not exercised.
    qs.dQxx = QDerivs{ 0.0, 0.0, 0.0, lap.xx, 0.0, 0.0, 0.0 };
    qs.dQxy = QDerivs{ 0.0, 0.0, 0.0, lap.xy, 0.0, 0.0, 0.0 };
    qs.dQxz = QDerivs{ 0.0, 0.0, 0.0, lap.xz, 0.0, 0.0, 0.0 };
    qs.dQyy = QDerivs{ 0.0, 0.0, 0.0, lap.yy, 0.0, 0.0, 0.0 };
    qs.dQyz = QDerivs{ 0.0, 0.0, 0.0, lap.yz, 0.0, 0.0, 0.0 };
    qs.gradu = GradTensor{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    PointwiseOut out{};
    SymTrLessTensor5 q_new{};
    Vec3 ericksen{};
    PointwiseStepAndSetupBodyForce(qs, q_new, out.sym, out.antisym, ericksen);
    return out;
}

const SymTrLess5 kQ  {  0.31, -0.07,  0.11, -0.18,  0.04 };
const SymTrLess5 kLap{  0.13,  0.05, -0.09,  0.21, -0.03 };

} // namespace

// ----- part 1: the pointwise split -----

// QH - HQ is antisymmetric for any two symmetric matrices. Confirms the
// reference itself before it is used to judge the code.
TEST(AntisymStress, ReferenceCommutatorIsAntisymmetric) {
    const SymTrLess5 H = MolecularField(kQ, kLap);
    const Mat3 QH = MatMul(Full(kQ), Full(H));
    const Mat3 HQ = MatMul(Full(H), Full(kQ));

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const double aij = QH[i][j] - HQ[i][j];
            const double aji = QH[j][i] - HQ[j][i];
            EXPECT_NEAR(aij, -aji, 1e-12) << "at (" << i << "," << j << ")";
        }
}

// The returned antisymmetric part equals the commutator QH - HQ.
TEST(AntisymStress, MatchesCommutator) {
    const SymTrLess5 H = MolecularField(kQ, kLap);
    const Mat3 QH = MatMul(Full(kQ), Full(H));
    const Mat3 HQ = MatMul(Full(H), Full(kQ));

    const PointwiseOut out = RunPointwise(kQ, kLap);

    EXPECT_NEAR(out.antisym.xy, QH[0][1] - HQ[0][1], 1e-12);
    EXPECT_NEAR(out.antisym.xz, QH[0][2] - HQ[0][2], 1e-12);
    EXPECT_NEAR(out.antisym.yz, QH[1][2] - HQ[1][2], 1e-12);
}

// The decisive pointwise check: the symmetric container must hold ONLY the
// symmetric part. Folding Tau in was the original defect.
//
// Compared on OFF-DIAGONAL components (xy, xz, yz) so that the +2 lambda
// tr(QH) Q_ij Onsager-conjugate term restored on the third-Beris-Edwards
// branch enters cleanly without pulling in the isotropic (2/3) lambda
// tr(QH) delta_ij piece, which sits on the diagonal only. The teeth this
// test cares about — that the antisymmetric commutator was not folded in —
// live off the diagonal too, so nothing is lost.
TEST(AntisymStress, SymmetricPartExcludesCommutator) {
    const SymTrLess5 H = MolecularField(kQ, kLap);
    const Mat3 Qf = Full(kQ);
    const Mat3 Hf = Full(H);
    const Mat3 QH = MatMul(Qf, Hf);
    const Mat3 HQ = MatMul(Hf, Qf);
    const double ktwo_thirds = 2.0 / 3.0;
    double tr_QH = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            tr_QH += Qf[i][j] * Hf[i][j];

    const PointwiseOut out = RunPointwise(kQ, kLap);

    // Off-diagonal Sigma:
    //   Sigma_ij = -(2/3) lambda H_ij - lambda (QH + HQ)_ij + 2 lambda tr(QH) Q_ij
    // (no delta_ij contribution off the diagonal)
    const double Sigma_xy = -ktwo_thirds * LAMBDA * H.xy - LAMBDA * (QH[0][1] + HQ[0][1])
                            + 2.0 * LAMBDA * tr_QH * Qf[0][1];
    const double Sigma_xz = -ktwo_thirds * LAMBDA * H.xz - LAMBDA * (QH[0][2] + HQ[0][2])
                            + 2.0 * LAMBDA * tr_QH * Qf[0][2];
    const double Sigma_yz = -ktwo_thirds * LAMBDA * H.yz - LAMBDA * (QH[1][2] + HQ[1][2])
                            + 2.0 * LAMBDA * tr_QH * Qf[1][2];

    EXPECT_NEAR(out.sym.xy, Sigma_xy, 1e-12);
    EXPECT_NEAR(out.sym.xz, Sigma_xz, 1e-12);
    EXPECT_NEAR(out.sym.yz, Sigma_yz, 1e-12);

    // And the commutator is genuinely nonzero here, so the check has teeth.
    ASSERT_GT(std::abs(out.antisym.xz), 1e-6)
        << "test fixture degenerate: Q and H commute, nothing to detect";
}

// Q and H commute when both are diagonal, so Tau must vanish identically.
TEST(AntisymStress, VanishesWhenQAndHCommute) {
    const SymTrLess5 Qdiag  { 0.3, 0.0, 0.0, -0.15, 0.0 };
    const SymTrLess5 LapDiag{ 0.2, 0.0, 0.0,  0.05, 0.0 };

    const PointwiseOut out = RunPointwise(Qdiag, LapDiag);

    EXPECT_NEAR(out.antisym.xy, 0.0, 1e-14);
    EXPECT_NEAR(out.antisym.xz, 0.0, 1e-14);
    EXPECT_NEAR(out.antisym.yz, 0.0, 1e-14);
}

// ----- part 2: the divergence signs -----

namespace {

// Evaluate PassiveStressDivergence on synthetic fields at an interior point.
// Interior is important: a linear ramp is discontinuous across a periodic seam,
// so the central difference must not straddle the wrap.
struct DivFields {
    LocalGrid g = LocalGrid::SingleRank();
    std::vector<double> sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, tau_xy, tau_xz, tau_yz;

    DivFields()
        : sigma_xx(g.HaloVolume(), 0.0), sigma_xy(g.HaloVolume(), 0.0), sigma_xz(g.HaloVolume(), 0.0),
          sigma_yy(g.HaloVolume(), 0.0), sigma_yz(g.HaloVolume(), 0.0), tau_xy(g.HaloVolume(), 0.0),
          tau_xz(g.HaloVolume(), 0.0), tau_yz(g.HaloVolume(), 0.0) {}

    // Fill one component with slope*coord along the named axis.
    void Ramp(std::vector<double>& field, char axis, double slope) {
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x) {
                    const int c = (axis == 'x') ? x : (axis == 'y') ? y : z;
                    field[g.halo_idx(x, y, z)] = slope * static_cast<double>(c);
                }
    }

    Vec3 Divergence() {
        return PassiveStressDivergence<FullyPeriodicConfig>(
            sigma_xx.data(), sigma_xy.data(), sigma_xz.data(), sigma_yy.data(), sigma_yz.data(),
            tau_xy.data(), tau_xz.data(), tau_yz.data(),
            nx / 2, ny / 2, nz / 2, g);
    }
};

constexpr double kSlope = 0.25;

} // namespace

// CONCERNS.md's stated failure input: a texture with τ_xz varying along x.
// f_z = d_x sigma_zx = d_x(Σ_xz - τ_xz), so f_z must pick up MINUS the slope.
// The pre-fix code returned +kSlope here.
TEST(AntisymStressDivergence, TxzRampAlongXGivesNegativeFz) {
    DivFields f;
    f.Ramp(f.tau_xz, 'x', kSlope);

    const Vec3 div = f.Divergence();

    EXPECT_NEAR(div.z, -kSlope, 1e-12);
    EXPECT_NEAR(div.x, 0.0, 1e-12) << "τ_xz varies only in x, so d_z τ_xz = 0";
    EXPECT_NEAR(div.y, 0.0, 1e-12);
}

// f_y = d_x sigma_yx = d_x(Σ_xy - τ_xy): also minus.
TEST(AntisymStressDivergence, TxyRampAlongXGivesNegativeFy) {
    DivFields f;
    f.Ramp(f.tau_xy, 'x', kSlope);

    const Vec3 div = f.Divergence();

    EXPECT_NEAR(div.y, -kSlope, 1e-12);
    EXPECT_NEAR(div.x, 0.0, 1e-12) << "τ_xy varies only in x, so d_y τ_xy = 0";
    EXPECT_NEAR(div.z, 0.0, 1e-12);
}

// The asymmetry that makes this subtle: f_x references the UPPER triangle, so
// there the antisymmetric part enters with a PLUS sign.
// f_x = d_y sigma_xy = d_y(Σ_xy + τ_xy).
TEST(AntisymStressDivergence, TxyRampAlongYGivesPositiveFx) {
    DivFields f;
    f.Ramp(f.tau_xy, 'y', kSlope);

    const Vec3 div = f.Divergence();

    EXPECT_NEAR(div.x, +kSlope, 1e-12);
    EXPECT_NEAR(div.y, 0.0, 1e-12) << "τ_xy varies only in y, so d_x τ_xy = 0";
    EXPECT_NEAR(div.z, 0.0, 1e-12);
}

// f_z = d_y sigma_zy = d_y(Σ_yz - τ_yz).
TEST(AntisymStressDivergence, TyzRampAlongYGivesNegativeFz) {
    DivFields f;
    f.Ramp(f.tau_yz, 'y', kSlope);

    const Vec3 div = f.Divergence();

    EXPECT_NEAR(div.z, -kSlope, 1e-12);
    EXPECT_NEAR(div.y, 0.0, 1e-12) << "τ_yz varies only in y, so d_z τ_yz = 0";
    EXPECT_NEAR(div.x, 0.0, 1e-12);
}

// A purely symmetric stress must be unaffected by the split: this is the
// regression guard that the refactor did not perturb the symmetric path.
TEST(AntisymStressDivergence, PurelySymmetricStressUnchanged) {
    DivFields f;
    f.Ramp(f.sigma_xy, 'x', kSlope);   // Σ_xy = slope*x, no antisymmetric part

    const Vec3 div = f.Divergence();

    EXPECT_NEAR(div.y, +kSlope, 1e-12) << "f_y = d_x Σ_xy, sign unaffected by A";
    EXPECT_NEAR(div.x, 0.0, 1e-12);
    EXPECT_NEAR(div.z, 0.0, 1e-12);
}

// Σ_zz = -(Σ_xx + Σ_yy) and τ has no diagonal, so the zz route is untouched.
TEST(AntisymStressDivergence, TracelessZZTermUnaffected) {
    DivFields f;
    f.Ramp(f.sigma_xx, 'z', kSlope);

    const Vec3 div = f.Divergence();

    EXPECT_NEAR(div.z, -kSlope, 1e-12) << "f_z = d_z Σ_zz = -d_z(Σ_xx + Σ_yy)";
    EXPECT_NEAR(div.x, 0.0, 1e-12);
    EXPECT_NEAR(div.y, 0.0, 1e-12);
}
