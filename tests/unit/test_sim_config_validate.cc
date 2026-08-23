#include <gtest/gtest.h>
#include "sim_config_attrs.h"
#include "boundary.h"
#include "boundary_names.h"

// The hard/soft split is the contract find_defects surfaces to the user. If
// this test starts failing, either the split was intentionally changed (and
// you should be updating find_defects' docs too) or a field slipped from one
// bucket to the other by accident. Either way, look — don't blind-update.

namespace {

// Build a snapshot that matches the current build's compile-time SimBC +
// Params exactly. Every ValidateAgainstBuild call against this returns
// ok() == true.
SimConfigAttr::SimConfigSnapshot MatchingSnapshot() {
    using namespace Params;
    SimConfigAttr::SimConfigSnapshot s;
    s.bc_name    = std::string(SimBC::name);
    s.xlo_qbc    = BCName<typename SimBC::XLo::QBC>::get();
    s.xlo_ubc    = BCName<typename SimBC::XLo::UBC>::get();
    s.xhi_qbc    = BCName<typename SimBC::XHi::QBC>::get();
    s.xhi_ubc    = BCName<typename SimBC::XHi::UBC>::get();
    s.ylo_qbc    = BCName<typename SimBC::YLo::QBC>::get();
    s.ylo_ubc    = BCName<typename SimBC::YLo::UBC>::get();
    s.yhi_qbc    = BCName<typename SimBC::YHi::QBC>::get();
    s.yhi_ubc    = BCName<typename SimBC::YHi::UBC>::get();
    s.zlo_qbc    = BCName<typename SimBC::ZLo::QBC>::get();
    s.zlo_ubc    = BCName<typename SimBC::ZLo::UBC>::get();
    s.zhi_qbc    = BCName<typename SimBC::ZHi::QBC>::get();
    s.zhi_ubc    = BCName<typename SimBC::ZHi::UBC>::get();
    s.qadvection = SimConfigAttr::QAdvectionName(::kQAdvection);
    s.git_commit = ::kGitCommit;
    s.nx           = nx;
    s.ny           = ny;
    s.nz           = nz;
    s.DX           = DX;
    s.DY           = DY;
    s.DZ           = DZ;
    s.DT           = DT;
    s.TAUF         = TAUF;
    s.A            = A;
    s.B            = B;
    s.C            = C;
    s.L            = L;
    s.ALPHA        = ALPHA;
    s.LAMBDA       = LAMBDA;
    s.GAMMA        = GAMMA;
    s.MU           = MU;
    s.NOISE        = NOISE;
    s.kNumSteps    = ::kNumSteps;
    s.kSaveInterval= ::kSaveInterval;
    return s;
}

bool HasField(const std::vector<SimConfigAttr::Mismatch>& v, const char* name) {
    for (const auto& m : v) if (m.field == name) return true;
    return false;
}

}  // namespace

TEST(SimConfigValidateTest, MatchingSnapshotIsClean) {
    const auto s = MatchingSnapshot();
    const auto r = SimConfigAttr::ValidateAgainstBuild<SimBC>(s);
    EXPECT_TRUE(r.ok());
    EXPECT_FALSE(r.git_commit_mismatch);
}

TEST(SimConfigValidateTest, LMismatchIsHard) {
    // L (Frank elasticity) directly drives DefectAnalysis::OmegaRingRadius,
    // which sets where β samples Q. Wrong L = wrong β. Hard.
    auto s = MatchingSnapshot();
    s.L *= 1.5;
    const auto r = SimConfigAttr::ValidateAgainstBuild<SimBC>(s);
    EXPECT_FALSE(r.ok());
    EXPECT_TRUE(HasField(r.hard, "L"));
    EXPECT_FALSE(HasField(r.soft, "L"));
}

TEST(SimConfigValidateTest, ABCMismatchesAreHard) {
    // A, B, C also feed DefectCoreXi via the LdG polynomial.
    for (const char* field : {"A", "B", "C"}) {
        auto s = MatchingSnapshot();
        if (field == std::string("A")) s.A += 0.1;
        if (field == std::string("B")) s.B += 0.1;
        if (field == std::string("C")) s.C += 0.1;
        const auto r = SimConfigAttr::ValidateAgainstBuild<SimBC>(s);
        EXPECT_FALSE(r.ok()) << "field=" << field;
        EXPECT_TRUE(HasField(r.hard, field)) << "field=" << field;
    }
}

TEST(SimConfigValidateTest, GridMismatchIsHard) {
    auto s = MatchingSnapshot();
    s.nx += 1;
    const auto r = SimConfigAttr::ValidateAgainstBuild<SimBC>(s);
    EXPECT_FALSE(r.ok());
    EXPECT_TRUE(HasField(r.hard, "nx"));
}

TEST(SimConfigValidateTest, QBCMismatchIsHard) {
    auto s = MatchingSnapshot();
    s.xlo_qbc = "Periodic";   // whatever the build's XLo QBC is, "Periodic"
                              // will differ from Neumann under SlitNoSlipConfig.
    const auto r = SimConfigAttr::ValidateAgainstBuild<SimBC>(s);
    // If the build's actual XLo::QBC is already "Periodic" this test would
    // not detect a mismatch — in that case, force a different one.
    if (r.ok()) {
        s.xlo_qbc = "Neumann";
        const auto r2 = SimConfigAttr::ValidateAgainstBuild<SimBC>(s);
        EXPECT_FALSE(r2.ok());
        EXPECT_TRUE(HasField(r2.hard, "XLo_QBC"));
    } else {
        EXPECT_TRUE(HasField(r.hard, "XLo_QBC"));
    }
}

TEST(SimConfigValidateTest, UBCMismatchIsSoft) {
    // UBC only affects LBM streaming, which never runs in the offline path.
    // A given lbm_*.vtkhdf frame carries the same (S, n̂) regardless of UBC.
    auto s = MatchingSnapshot();
    s.xlo_ubc = "Foo";  // any value that isn't the current build's XLo::UBC
    const auto r = SimConfigAttr::ValidateAgainstBuild<SimBC>(s);
    EXPECT_TRUE(HasField(r.soft, "XLo_UBC"));
    EXPECT_FALSE(HasField(r.hard, "XLo_UBC"));
}

TEST(SimConfigValidateTest, LambdaMismatchIsSoft) {
    // LAMBDA drives Q dynamics, not the (S, n̂) snapshot analysis. Soft.
    auto s = MatchingSnapshot();
    s.LAMBDA += 0.1;
    const auto r = SimConfigAttr::ValidateAgainstBuild<SimBC>(s);
    EXPECT_TRUE(HasField(r.soft, "LAMBDA"));
    EXPECT_FALSE(HasField(r.hard, "LAMBDA"));
}

TEST(SimConfigValidateTest, GitCommitMismatchFlaggedButNotFatal) {
    // Different commit, both non-empty → mismatch flagged but not folded into
    // hard/soft (find_defects reports it as a warning either way).
    auto s = MatchingSnapshot();
    s.git_commit = "0000000000000000000000000000000000000000";
    const auto r = SimConfigAttr::ValidateAgainstBuild<SimBC>(s);
    // If the build has no git commit (empty), the mismatch is silently
    // accepted (empty on either side means "unknown", per design).
    if (std::string(::kGitCommit).empty()) {
        EXPECT_FALSE(r.git_commit_mismatch);
    } else {
        EXPECT_TRUE(r.git_commit_mismatch);
    }
}

TEST(SimConfigValidateTest, EmptyGitCommitIsNeverAMismatch) {
    // "No git info on either side" is a common case (CI without a checkout,
    // or a tarball build) — must not be flagged.
    auto s = MatchingSnapshot();
    s.git_commit = "";
    const auto r = SimConfigAttr::ValidateAgainstBuild<SimBC>(s);
    EXPECT_FALSE(r.git_commit_mismatch);
}
