#include <gtest/gtest.h>
#include "boundary.h"
#include "boundary_names.h"

// Pin the wire strings the BCName<T> trait produces. These are stamped onto
// every VTKHDF frame; a change here changes the on-disk contract and would
// silently break find_defects' validator on old data.

TEST(BoundaryNamesTest, TagsHaveCanonicalStrings) {
    EXPECT_EQ(BCName<Periodic>::get(),           "Periodic");
    EXPECT_EQ(BCName<Neumann>::get(),            "Neumann");
    EXPECT_EQ(BCName<NoSlip>::get(),             "NoSlip");
    EXPECT_EQ(BCName<SpecularReflection>::get(), "SpecularReflection");
}

TEST(BoundaryNamesTest, AnchoringEncodesTemplateParams) {
    using A = Anchoring<0.5, 0.0, 0.0>;
    // Just enough of the encoded string to lock in that template args go in
    // and don't get lost. Byte-exact wire format is pinned separately below.
    const std::string s = BCName<A>::get();
    EXPECT_NE(s.find("Anchoring"), std::string::npos);
    EXPECT_NE(s.find("S=0.5"), std::string::npos);
    EXPECT_NE(s.find("theta=0"), std::string::npos);
    EXPECT_NE(s.find("phi=0"), std::string::npos);
}

TEST(BoundaryNamesTest, MovingWallEncodesTemplateParams) {
    using W = MovingWall<0.01, 0.0, 0.0>;
    const std::string s = BCName<W>::get();
    EXPECT_NE(s.find("MovingWall"), std::string::npos);
    EXPECT_NE(s.find("Ux=0.01"), std::string::npos);
    EXPECT_NE(s.find("Uy=0"), std::string::npos);
    EXPECT_NE(s.find("Uz=0"), std::string::npos);
}

// Byte-exact pinning of the wire format for non-trivial doubles. The
// `{:.17g}` format spec must produce identical output across formatter
// implementations for a given double bit pattern — this test would fail
// loudly if a future compat::format upgrade silently changed the encoding
// (which would invalidate every previously-stamped VTKHDF file).
//
// Chosen values are irrational / mantissa-limited so they exercise more
// than the trivial "0" / "1" cases:
//   0.5                 → exact in binary, prints as "0.5"
//   0.1                 → NOT exact in binary; shortest-roundtrip is "0.1"
//   0.7853981633974483  → the closest double to π/4; needs all 16 digits
TEST(BoundaryNamesTest, DoubleFormatIsBitExact) {
    using A = Anchoring<0.5, 0.7853981633974483, 0.1>;
    EXPECT_EQ(BCName<A>::get(),
              "Anchoring(S=0.5,theta=0.78539816339744828,phi=0.10000000000000001)");

    using W = MovingWall<0.5, 0.1, 0.7853981633974483>;
    EXPECT_EQ(BCName<W>::get(),
              "MovingWall(Ux=0.5,Uy=0.10000000000000001,Uz=0.78539816339744828)");
}

// Compile-time check on the periodicity-consistency trait — a positive test
// only, since a negative one would need a `requires`-clause SFINAE dance to
// avoid triggering static_assert at compile time. This still catches the
// case where a valid BC starts being (incorrectly) flagged inconsistent.
TEST(BoundaryNamesTest, ConsistentBCPasses) {
    static_assert(bc_periodicity_consistent_v<FullyPeriodicConfig>);
    static_assert(bc_periodicity_consistent_v<ChannelConfig>);
    SUCCEED();
}
