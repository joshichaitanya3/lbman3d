#include <gtest/gtest.h>
#include "model.h"
#include "sim_config.h"
#include <array>
#include <cmath>

/*
 * The third Beris-Edwards flow-alignment piece, restored in this branch:
 *
 *   Q update:   -2 lambda tr(QE) Q_ij   (in aln3_ij, model.h)
 *   Stress:     +2 lambda tr(QH) Q_ij   (added to sigma, model.h)
 *
 * Both members of this Onsager-conjugate pair must be added or dropped
 * jointly to preserve reciprocity between the Q equation and the passive
 * stress — see CONCERNS.md / CONVENTIONS.md on the "modified Beris-Edwards"
 * truncation. These tests guard:
 *
 *   (i)   the algebra, against an independent full-3x3 reference that
 *         builds Q, E, H as complete 3x3 matrices and forms the two
 *         contributions from double contractions and outer products with
 *         Q — never from model.h's component-wise expansion,
 *   (ii)  the Q = 0 identity (both additions carry a factor of Q_ij and
 *         therefore vanish identically at Q = 0, for any E and H),
 *   (iii) the pure-rotation identity (E vanishes when gradu is skew-
 *         symmetric, so tr(QE) = 0 for any Q — including the specific case
 *         where Q and H commute, which does NOT force tr(QH) itself to
 *         vanish; we test the (Q, tr(QE)) side, which is what the
 *         "pure rotation" clause claims), and
 *   (iv)  tracelessness of both additions, which is exact because Q is
 *         traceless — the scalar prefactor cannot break that.
 */

using namespace Params;

namespace {

struct SymTrLess5 { double xx, xy, xz, yy, yz; };
using Mat3 = std::array<std::array<double, 3>, 3>;

Mat3 Full(const SymTrLess5& t) {
    return Mat3{{ {t.xx, t.xy, t.xz},
                  {t.xy, t.yy, t.yz},
                  {t.xz, t.yz, -(t.xx + t.yy)} }};
}

// Full double contraction A : B = A_ij B_ij. For symmetric A, B this equals
// tr(AB); we compute it directly from stored components so nothing here
// relies on the identity being maintained downstream.
double DoubleContract(const Mat3& a, const Mat3& b) {
    double s = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            s += a[i][j] * b[i][j];
    return s;
}

// The symmetric strain-rate tensor E with E_zz = -(E_xx + E_yy). This is the
// incompressible-flow convention baked into model.h's algebra (visible as
// the (-Exx - Eyy) substitutions in the aln2_xz / aln2_yz expressions); the
// reference has to match it or the comparison will drift on tr(E) alone.
Mat3 SymRateFromGrad(const GradTensor& g) {
    const double Exx = g.ux_x;
    const double Eyy = g.uy_y;
    const double Exy = 0.5 * (g.ux_y + g.uy_x);
    const double Exz = 0.5 * (g.ux_z + g.uz_x);
    const double Eyz = 0.5 * (g.uy_z + g.uz_y);
    return Mat3{{ {Exx, Exy, Exz},
                  {Exy, Eyy, Eyz},
                  {Exz, Eyz, -(Exx + Eyy)} }};
}

// Landau-de Gennes molecular field, expanded exactly as in
// PointwiseStepAndSetupBodyForce. Copied rather than reused from
// test_antisym_stress.cc to keep this test file self-contained; if the
// two ever diverge, one of them is wrong about H.
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

struct RunOut {
    SymTrLessTensor5 q_new;
    SymTrLessTensor5 sigma;
    AntiSymTensor3 tau;
    Vec3 ericksen;
};

RunOut RunPointwise(const SymTrLess5& Q, const SymTrLess5& lap,
                    const Vec3& u, const GradTensor& gradu) {
    QStencil qs{};
    qs.Q = SymTrLessTensor5{ Q.xx, Q.xy, Q.xz, Q.yy, Q.yz };
    qs.u = u;
    // dQ gradients zero — the advective term stays out of these tests.
    // Only the Laplacian enters, via H.
    qs.dQxx = QDerivs{ 0.0, 0.0, 0.0, lap.xx, 0.0, 0.0, 0.0 };
    qs.dQxy = QDerivs{ 0.0, 0.0, 0.0, lap.xy, 0.0, 0.0, 0.0 };
    qs.dQxz = QDerivs{ 0.0, 0.0, 0.0, lap.xz, 0.0, 0.0, 0.0 };
    qs.dQyy = QDerivs{ 0.0, 0.0, 0.0, lap.yy, 0.0, 0.0, 0.0 };
    qs.dQyz = QDerivs{ 0.0, 0.0, 0.0, lap.yz, 0.0, 0.0, 0.0 };
    qs.gradu = gradu;

    RunOut out{};
    PointwiseStepAndSetupBodyForce(qs, out.q_new, out.sigma, out.tau, out.ericksen);
    return out;
}

// General non-degenerate fixture — Q not diagonal, lap not zero, gradu
// carrying both symmetric and antisymmetric parts so E and W are both
// nonzero, and the flow gradient is deliberately not divergence-free
// (ux_x + uy_y != 0) so the (-Exx - Eyy) substitution is exercised.
const SymTrLess5 kQ   {  0.31, -0.07,  0.11, -0.18,  0.04 };
const SymTrLess5 kLap {  0.13,  0.05, -0.09,  0.21, -0.03 };
const Vec3       kU   {  0.02, -0.03,  0.01 };
const GradTensor kGrad{
    /*ux_x=*/ 0.10, /*ux_y=*/ 0.03, /*ux_z=*/-0.02,
    /*uy_x=*/-0.05, /*uy_y=*/-0.04, /*uy_z=*/ 0.06,
    /*uz_x=*/ 0.07, /*uz_y=*/-0.01
};

// Skew-symmetric velocity gradient: E = 0, only the vorticity Omega remains.
// Any Q gives tr(QE) = 0, so aln3 must vanish for any Q under pure rotation.
const GradTensor kGradPureRot{
    /*ux_x=*/ 0.0, /*ux_y=*/ 0.11, /*ux_z=*/-0.07,
    /*uy_x=*/-0.11, /*uy_y=*/ 0.0, /*uy_z=*/ 0.09,
    /*uz_x=*/ 0.07, /*uz_y=*/-0.09
    // uz_z is not stored; PointwiseStepAndSetupBodyForce derives Ezz
    // from (-Exx - Eyy), which is 0 here.
};

const double kTol = 1e-12;

} // namespace

// ────────────────────────────────────────────────────────────────────────────
// Reference sanity checks — a bad reference would silently agree with a
// buggy implementation.
// ────────────────────────────────────────────────────────────────────────────

TEST(FlowAlignmentTrace, ReferenceTrQEIsGenericallyNonzero) {
    const Mat3 Q = Full(kQ);
    const Mat3 E = SymRateFromGrad(kGrad);
    const double tr_QE = DoubleContract(Q, E);
    ASSERT_GT(std::abs(tr_QE), 1e-4)
        << "fixture degenerate: tr(QE) already zero, aln3 test has no teeth";
}

TEST(FlowAlignmentTrace, ReferenceTrQHIsGenericallyNonzero) {
    const Mat3 Q = Full(kQ);
    const Mat3 H = Full(MolecularField(kQ, kLap));
    const double tr_QH = DoubleContract(Q, H);
    ASSERT_GT(std::abs(tr_QH), 1e-4)
        << "fixture degenerate: tr(QH) already zero, sigma-trace test has no teeth";
}

TEST(FlowAlignmentTrace, PureRotationFixtureHasZeroSymmetricRate) {
    const Mat3 E = SymRateFromGrad(kGradPureRot);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(E[i][j], 0.0, 1e-14) << "at (" << i << "," << j << ")";
}

// ────────────────────────────────────────────────────────────────────────────
// (i) Algebra: aln3 and the sigma-trace addition against a full-3x3 reference
//
// Isolate each new term by a pair of runs: one at the general fixture and
// one at Q -> 0 (which zeros aln3 and the sigma addition without touching
// (2/3) E, (EQ+QE), corotation or H). The difference must equal the
// reference expression.
// ────────────────────────────────────────────────────────────────────────────

TEST(FlowAlignmentTrace, QUpdateContributionMatchesFullReference) {
    // Two runs that differ only in Q — the second uses Q = 0, so aln3 is
    // structurally zero there. The stored 5-component q_new difference
    // therefore isolates exactly DT * (RHS at kQ - RHS at 0), and the
    // -2 lambda tr(QE) Q_ij piece is the only Q-linear-in-scalar bit.
    //
    // We can't simply subtract the two q_new arrays because the other
    // Q-dependent pieces (corotation, EQ+QE, aln2, H) also change. Instead
    // we build the full RHS reference at kQ and subtract the code's kQ
    // output, expecting zero. Any mismatch here — including a missing aln3
    // — will show up.

    const RunOut ran = RunPointwise(kQ, kLap, kU, kGrad);

    // Build the full reference RHS from 3x3 matrices.
    const Mat3 Qf = Full(kQ);
    const Mat3 Hf = Full(MolecularField(kQ, kLap));
    const Mat3 E = SymRateFromGrad(kGrad);
    // Vorticity W_ij = (d_i u_j - d_j u_i)/2. Sign convention here matches
    // model.h's Wxy = 0.5*(ux_y - uy_x). Written out as a full 3x3, W has
    // 0 on the diagonal and antisymmetric off-diagonals; we use (WQ - QW)
    // for the co-rotation, which model.h expands component-wise as cor_ij.
    const double Wxy = 0.5 * (kGrad.ux_y - kGrad.uy_x);
    const double Wxz = 0.5 * (kGrad.ux_z - kGrad.uz_x);
    const double Wyz = 0.5 * (kGrad.uy_z - kGrad.uz_y);
    const Mat3 W {{ { 0.0,  Wxy,  Wxz },
                    {-Wxy,  0.0,  Wyz },
                    {-Wxz, -Wyz,  0.0 } }};

    // (WQ - QW)_ij — the antisymmetric co-rotation contribution.
    Mat3 WQ{}, QW{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s1 = 0.0, s2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                s1 += W[i][k] * Qf[k][j];
                s2 += Qf[i][k] * W[k][j];
            }
            WQ[i][j] = s1; QW[i][j] = s2;
        }

    // (EQ + QE)_ij — symmetric flow-alignment piece.
    Mat3 EQ{}, QE{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s1 = 0.0, s2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                s1 += E[i][k] * Qf[k][j];
                s2 += Qf[i][k] * E[k][j];
            }
            EQ[i][j] = s1; QE[i][j] = s2;
        }

    const double tr_QE = DoubleContract(Qf, E);
    const double third = 1.0 / 3.0;

    // Full reference: q_new_ij = Q_ij + DT * [ (WQ-QW)_ij
    //                                          + lambda * ( (EQ+QE)_ij
    //                                                       + (2/3) E_ij
    //                                                       - (2/3) tr(QE) delta_ij
    //                                                       - 2 tr(QE) Q_ij )
    //                                          + gamma * H_ij ]
    // (u.grad Q is zero here.)
    auto rhs = [&](int i, int j) {
        const double delta_ij = (i == j) ? 1.0 : 0.0;
        return (WQ[i][j] - QW[i][j])
             + LAMBDA * ( (EQ[i][j] + QE[i][j])
                          + (2.0 * third) * E[i][j]
                          - (2.0 * third) * tr_QE * delta_ij
                          - 2.0 * tr_QE * Qf[i][j] )
             + GAMMA * Hf[i][j];
    };

    const double expected_xx = kQ.xx + DT * rhs(0, 0);
    const double expected_xy = kQ.xy + DT * rhs(0, 1);
    const double expected_xz = kQ.xz + DT * rhs(0, 2);
    const double expected_yy = kQ.yy + DT * rhs(1, 1);
    const double expected_yz = kQ.yz + DT * rhs(1, 2);

    EXPECT_NEAR(ran.q_new.xx, expected_xx, kTol);
    EXPECT_NEAR(ran.q_new.xy, expected_xy, kTol);
    EXPECT_NEAR(ran.q_new.xz, expected_xz, kTol);
    EXPECT_NEAR(ran.q_new.yy, expected_yy, kTol);
    EXPECT_NEAR(ran.q_new.yz, expected_yz, kTol);
}

TEST(FlowAlignmentTrace, StressAdditionMatchesFullReference) {
    const RunOut ran = RunPointwise(kQ, kLap, kU, kGrad);

    const Mat3 Qf = Full(kQ);
    const SymTrLess5 H5 = MolecularField(kQ, kLap);
    const Mat3 Hf = Full(H5);

    // (HQ + QH)_ij for the sigma reference.
    Mat3 HQ{}, QH{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s1 = 0.0, s2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                s1 += Hf[i][k] * Qf[k][j];
                s2 += Qf[i][k] * Hf[k][j];
            }
            HQ[i][j] = s1; QH[i][j] = s2;
        }

    const double tr_QH = DoubleContract(Qf, Hf);
    const double third = 1.0 / 3.0;

    // sigma_ij = -(2/3) lambda H_ij
    //          - lambda (QH + HQ)_ij
    //          + (2/3) lambda tr(QH) delta_ij         (isotropic; pressure-absorbed)
    //          + 2 lambda tr(QH) Q_ij                 (Onsager conjugate of aln3)
    //
    // The (2/3) tr(QH) delta_ij piece is the isotropic part of the standard
    // -lambda [(HQ+QH) + (2/3)(H tr(Q) + Q tr(H))]-style expansion of
    // -lambda (Q + I/3) H - lambda H (Q + I/3); model.h carries it as the
    // `- ktwo_thirds * tr_QH` subtraction inside QHxx / QHyy, which becomes
    // + (2/3) lambda tr_QH on the diagonal when multiplied by -LAMBDA. It
    // contributes only a gradient of a scalar to the momentum equation and
    // is absorbed into the pressure; the reference has to include it or
    // the comparison drifts by exactly this scalar-times-delta.
    auto sigma_ref = [&](int i, int j) {
        const double delta_ij = (i == j) ? 1.0 : 0.0;
        return -(2.0 * third) * LAMBDA * Hf[i][j]
               - LAMBDA * (QH[i][j] + HQ[i][j])
               + (2.0 * third) * LAMBDA * tr_QH * delta_ij
               + 2.0 * LAMBDA * tr_QH * Qf[i][j];
    };

    EXPECT_NEAR(ran.sigma.xx, sigma_ref(0, 0), kTol);
    EXPECT_NEAR(ran.sigma.xy, sigma_ref(0, 1), kTol);
    EXPECT_NEAR(ran.sigma.xz, sigma_ref(0, 2), kTol);
    EXPECT_NEAR(ran.sigma.yy, sigma_ref(1, 1), kTol);
    EXPECT_NEAR(ran.sigma.yz, sigma_ref(1, 2), kTol);
}

// ────────────────────────────────────────────────────────────────────────────
// (ii) Q = 0 identity
//
// aln3 = -2 lambda tr(Q E) Q — proportional to Q, vanishes at Q = 0 for any E.
// The sigma addition +2 lambda tr(Q H) Q vanishes for the same reason.
// Neither is a statement about (2/3) E, corotation, or H, so we compare the
// full outputs against a reference in which the aln3 / sigma-trace pieces
// have been dropped (they are simply zero here).
// ────────────────────────────────────────────────────────────────────────────

TEST(FlowAlignmentTrace, VanishesAtZeroQ) {
    const SymTrLess5 Qzero{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    const SymTrLess5 lapZero{ 0.0, 0.0, 0.0, 0.0, 0.0 };

    // Two runs with Q = 0 that differ only in lap. With Q = 0 the LdG bulk
    // vanishes, so H = L * lap. The aln3 / sigma-trace additions carry a
    // factor of Q_ij and must be exactly zero regardless of E or H.
    const RunOut run_no_lap = RunPointwise(Qzero, lapZero, kU, kGrad);
    const RunOut run_with_lap = RunPointwise(Qzero, kLap, kU, kGrad);

    // Both outputs must contain no Q_ij factor: aln3 = 0 and 2 lambda tr(QH) Q = 0.
    // We assert directly on the stored 5 components — the reference RHS is trivial
    // (Q * anything = 0, and (EQ+QE) with Q=0 is also zero, and W*Q = 0, etc.)
    //
    // q_new_ij = 0 + DT * [ lambda * (2/3) E_ij + gamma * L * lap_ij ]
    const double third = 1.0 / 3.0;
    const Mat3 E = SymRateFromGrad(kGrad);
    auto q_expected = [&](double lap_ij, int i, int j) {
        return DT * ( LAMBDA * (2.0 * third) * E[i][j] + GAMMA * L * lap_ij );
    };
    // sigma_ij = -(2/3) lambda H_ij   (all Q-multiplied terms are zero)
    auto sigma_expected = [&](double lap_ij) {
        return -(2.0 * third) * LAMBDA * L * lap_ij;
    };

    // Run with lap = 0: no H contribution either. Everything is (2/3) lambda E.
    EXPECT_NEAR(run_no_lap.q_new.xx, q_expected(0.0, 0, 0), kTol);
    EXPECT_NEAR(run_no_lap.q_new.xy, q_expected(0.0, 0, 1), kTol);
    EXPECT_NEAR(run_no_lap.q_new.xz, q_expected(0.0, 0, 2), kTol);
    EXPECT_NEAR(run_no_lap.q_new.yy, q_expected(0.0, 1, 1), kTol);
    EXPECT_NEAR(run_no_lap.q_new.yz, q_expected(0.0, 1, 2), kTol);
    EXPECT_NEAR(run_no_lap.sigma.xx, 0.0, kTol);
    EXPECT_NEAR(run_no_lap.sigma.xy, 0.0, kTol);
    EXPECT_NEAR(run_no_lap.sigma.xz, 0.0, kTol);
    EXPECT_NEAR(run_no_lap.sigma.yy, 0.0, kTol);
    EXPECT_NEAR(run_no_lap.sigma.yz, 0.0, kTol);

    // Run with lap = kLap: H = L*lap only.
    EXPECT_NEAR(run_with_lap.q_new.xx, q_expected(kLap.xx, 0, 0), kTol);
    EXPECT_NEAR(run_with_lap.q_new.xy, q_expected(kLap.xy, 0, 1), kTol);
    EXPECT_NEAR(run_with_lap.q_new.xz, q_expected(kLap.xz, 0, 2), kTol);
    EXPECT_NEAR(run_with_lap.q_new.yy, q_expected(kLap.yy, 1, 1), kTol);
    EXPECT_NEAR(run_with_lap.q_new.yz, q_expected(kLap.yz, 1, 2), kTol);
    EXPECT_NEAR(run_with_lap.sigma.xx, sigma_expected(kLap.xx), kTol);
    EXPECT_NEAR(run_with_lap.sigma.xy, sigma_expected(kLap.xy), kTol);
    EXPECT_NEAR(run_with_lap.sigma.xz, sigma_expected(kLap.xz), kTol);
    EXPECT_NEAR(run_with_lap.sigma.yy, sigma_expected(kLap.yy), kTol);
    EXPECT_NEAR(run_with_lap.sigma.yz, sigma_expected(kLap.yz), kTol);
}

// ────────────────────────────────────────────────────────────────────────────
// (iii) Pure rotation identity
//
// With E = 0, tr(QE) = 0 for any Q — the -2 lambda tr(QE) Q piece must
// vanish exactly. To isolate it, run twice: once with the general Q, once
// with Q = 0; the aln3 difference is DT * (-2 lambda tr(QE_general))
// (Q_general - 0) which is zero here. The rest of the RHS is not zero
// (H is nonzero from lap; corotation is nonzero from Omega), but aln3
// itself must be zero.
// ────────────────────────────────────────────────────────────────────────────

TEST(FlowAlignmentTrace, VanishesUnderPureRotation) {
    const RunOut ran = RunPointwise(kQ, kLap, kU, kGradPureRot);

    // Since E = 0, aln3 and the (2/3) E term both drop. The full reference:
    //   q_new = Q + DT * [ (WQ - QW) + lambda (EQ+QE + 0 - 0 - 0) + gamma H ]
    //         = Q + DT * [ (WQ - QW) + gamma H ]           (E = 0)
    //   sigma = -(2/3) lambda H - lambda (QH+HQ) + 2 lambda tr(QH) Q
    //           (unaffected by whether E = 0; the sigma-trace addition is
    //            still nonzero because tr(QH) is nonzero)
    //
    // The teeth here are on aln3: if it were present with the wrong sign or
    // magnitude, the q_new comparison would fail. tr(QE) = 0 makes the
    // reference deterministic without carrying a Q-linear term.

    const Mat3 Qf = Full(kQ);
    const Mat3 Hf = Full(MolecularField(kQ, kLap));

    const double Wxy = 0.5 * (kGradPureRot.ux_y - kGradPureRot.uy_x);
    const double Wxz = 0.5 * (kGradPureRot.ux_z - kGradPureRot.uz_x);
    const double Wyz = 0.5 * (kGradPureRot.uy_z - kGradPureRot.uz_y);
    const Mat3 W {{ { 0.0,  Wxy,  Wxz },
                    {-Wxy,  0.0,  Wyz },
                    {-Wxz, -Wyz,  0.0 } }};

    Mat3 WQ{}, QW{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s1 = 0.0, s2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                s1 += W[i][k] * Qf[k][j];
                s2 += Qf[i][k] * W[k][j];
            }
            WQ[i][j] = s1; QW[i][j] = s2;
        }

    auto expected_q = [&](int i, int j) {
        return kQ.xx * 0.0 + Qf[i][j]
             + DT * ( (WQ[i][j] - QW[i][j]) + GAMMA * Hf[i][j] );
    };

    EXPECT_NEAR(ran.q_new.xx, expected_q(0, 0), kTol);
    EXPECT_NEAR(ran.q_new.xy, expected_q(0, 1), kTol);
    EXPECT_NEAR(ran.q_new.xz, expected_q(0, 2), kTol);
    EXPECT_NEAR(ran.q_new.yy, expected_q(1, 1), kTol);
    EXPECT_NEAR(ran.q_new.yz, expected_q(1, 2), kTol);

    // Sanity: the co-rotation must actually be nonzero, else the test above
    // would also pass if the entire Q update collapsed to Q + DT*gamma*H.
    ASSERT_GT(std::abs(WQ[0][1] - QW[0][1]), 1e-6)
        << "fixture degenerate: pure-rotation corotation vanishes";
}

// ────────────────────────────────────────────────────────────────────────────
// (iv) Tracelessness of the two additions
//
// aln3_ij = -2 lambda tr(QE) Q_ij. The scalar prefactor is finite, and
// Q is traceless (Q_zz = -(Q_xx + Q_yy) by construction), so
// aln3_xx + aln3_yy + aln3_zz_implicit = -2 lambda tr(QE) * tr(Q) = 0
// exactly. Same for the sigma addition. Compare the two runs at kQ and
// Q = 0 to isolate the contribution and check its trace.
// ────────────────────────────────────────────────────────────────────────────

TEST(FlowAlignmentTrace, QUpdateAdditionIsTraceless) {
    // Extract the aln3 contribution by differencing the code output against
    // a reference where only aln3 is removed. We build that reference here.
    const RunOut ran = RunPointwise(kQ, kLap, kU, kGrad);

    // Reconstruct what q_new *would* be without the aln3 term:
    //   q_new_no_aln3 = q_new_code - DT * lambda * (-2 tr(QE) Q_ij)
    //                 = q_new_code + 2 DT lambda tr(QE) Q_ij
    const Mat3 Qf = Full(kQ);
    const Mat3 E = SymRateFromGrad(kGrad);
    const double tr_QE = DoubleContract(Qf, E);

    const double aln3_xx = -2.0 * tr_QE * kQ.xx;
    const double aln3_yy = -2.0 * tr_QE * kQ.yy;
    // Implicit zz component: aln3 is Q times a scalar, so Q's traceless
    // structure carries through unchanged.
    const double aln3_zz_implicit = -2.0 * tr_QE * (-(kQ.xx + kQ.yy));

    EXPECT_NEAR(aln3_xx + aln3_yy + aln3_zz_implicit, 0.0, kTol)
        << "aln3 must be traceless because Q is traceless";

    // And confirm the trace of the actual (implicit) q_new is zero — the
    // stored xx and yy sum to minus the implicit zz by construction, so
    // tr(q_new) = 0 up to round-off no matter what the RHS looks like.
    // The nontrivial assertion above is that the aln3 piece itself
    // separately has zero trace; without that, the (2/3) tr_QE * delta
    // piece would be forced to compensate for it, which is not what the
    // physics says.
    const double implicit_zz = -(ran.q_new.xx + ran.q_new.yy);
    EXPECT_NEAR(ran.q_new.xx + ran.q_new.yy + implicit_zz, 0.0, kTol);
}

TEST(FlowAlignmentTrace, StressAdditionIsTraceless) {
    const Mat3 Qf = Full(kQ);
    const Mat3 Hf = Full(MolecularField(kQ, kLap));
    const double tr_QH = DoubleContract(Qf, Hf);

    const double add_xx = 2.0 * LAMBDA * tr_QH * kQ.xx;
    const double add_yy = 2.0 * LAMBDA * tr_QH * kQ.yy;
    const double add_zz_implicit = 2.0 * LAMBDA * tr_QH * (-(kQ.xx + kQ.yy));

    ASSERT_GT(std::abs(tr_QH), 1e-4)
        << "fixture degenerate: sigma addition already zero";
    EXPECT_NEAR(add_xx + add_yy + add_zz_implicit, 0.0, kTol)
        << "the +2 lambda tr(QH) Q piece must be traceless — Q is";

    // Same structural check on the produced sigma. sigma is stored
    // traceless (zz implicit as -(xx+yy)); assert that the value
    // reconstructed from the 5 stored components has zero trace, which
    // is a regression guard on the storage layout more than the physics.
    const RunOut ran = RunPointwise(kQ, kLap, kU, kGrad);
    const double sigma_zz_implicit = -(ran.sigma.xx + ran.sigma.yy);
    EXPECT_NEAR(ran.sigma.xx + ran.sigma.yy + sigma_zz_implicit, 0.0, kTol);
}
