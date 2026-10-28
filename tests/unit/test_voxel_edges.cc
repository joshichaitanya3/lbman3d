#include <gtest/gtest.h>
#include <params.h>
#include <sim_config.h>
#include "analysis/defect_finder.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

// The disclination graph has one edge per pair of firing faces that share a
// voxel. EmitVoxelEdges emits 12 of a voxel's 15 face pairs while visiting
// that voxel and the other three (1-2, 1-5, 2-5 in the figure in
// defect_finder.tpp) while visiting its +x or +y neighbour. A voxel loop that
// stops one layer short on a walled axis therefore drops those three pairs
// for the last voxel layer, which is where a line ending on the wall lives.
//
// The reference here is built from the voxel's six faces directly, never from
// the finder's lookup helpers, and checks every pair of every voxel.

namespace {

// All six faces walled: every axis loses a plaquette layer.
struct AllWalledConfig {
    using XLo = WallSpec<Neumann, NoSlip>;
    using XHi = WallSpec<Neumann, NoSlip>;
    using YLo = WallSpec<Neumann, NoSlip>;
    using YHi = WallSpec<Neumann, NoSlip>;
    using ZLo = WallSpec<Neumann, NoSlip>;
    using ZHi = WallSpec<Neumann, NoSlip>;
    static constexpr std::string_view name = "AllWalled";
};

enum class Family { X, Y, Z };

// One plaquette: its face family and its index within that family.
struct Face {
    Family family;
    int x, y, z;
};

uint8_t& Flag(DefectFields& df, const Face& f) {
    switch (f.family) {
        case Family::X: return df.def_x[df.FlatX(f.x, f.y, f.z)];
        case Family::Y: return df.def_y[df.FlatY(f.x, f.y, f.z)];
        default:        return df.def_z[df.FlatZ(f.x, f.y, f.z)];
    }
}

FaceId Fid(const DefectFields& df, const Face& f) {
    switch (f.family) {
        case Family::X: return df.FidX(f.x, f.y, f.z);
        case Family::Y: return df.FidY(f.x, f.y, f.z);
        default:        return df.FidZ(f.x, f.y, f.z);
    }
}

std::pair<FaceId, FaceId> Canonical(FaceId a, FaceId b) {
    return {std::min(a, b), std::max(a, b)};
}

// The six faces of voxel (x, y, z), numbered as in the defect_finder.tpp
// figure: pt 3/1 low/high x, pt 4/2 low/high y, pt 6/5 low/high z. The `% n`
// only wraps on a periodic axis; on a walled one x+1 <= n-1 already.
std::array<Face, 6> VoxelFaces(int x, int y, int z) {
    return {{
        {Family::X, (x + 1) % Params::nx, y, z},  // pt 1
        {Family::Y, x, (y + 1) % Params::ny, z},  // pt 2
        {Family::X, x, y, z},                     // pt 3
        {Family::Y, x, y, z},                     // pt 4
        {Family::Z, x, y, (z + 1) % Params::nz},  // pt 5
        {Family::Z, x, y, z},                     // pt 6
    }};
}

template<typename BC>
class VoxelEdgesTest : public ::testing::Test {
protected:
    DefectFields fields{periodicity_by_axis<BC>};
    DefectFinder<BC> finder;

    // A voxel exists wherever all six of its faces do: n per axis if
    // periodic, n-1 if walled.
    static constexpr int vx = kPX<BC> ? Params::nx : Params::nx - 1;
    static constexpr int vy = kPY<BC> ? Params::ny : Params::ny - 1;
    static constexpr int vz = kPZ<BC> ? Params::nz : Params::nz - 1;
};

using BCs = ::testing::Types<FullyPeriodicConfig, SlitNoSlipConfig, AllWalledConfig>;

struct BCName {
    template<typename BC>
    static std::string GetName(int) { return std::string(BC::name); }
};

TYPED_TEST_SUITE(VoxelEdgesTest, BCs, BCName);

// n >= 3 on every axis, so no voxel meets itself across a periodic wrap and
// two distinct faces share at most one voxel. "Exactly one edge" then holds
// for every pair.
static_assert(Params::nx >= 3 && Params::ny >= 3 && Params::nz >= 3);

TYPED_TEST(VoxelEdgesTest, NoFiringFacesEmitNoEdges) {
    EXPECT_TRUE(this->finder.EmitVoxelEdges(this->fields).empty());
}

TYPED_TEST(VoxelEdgesTest, SingleFiringFaceEmitsNoEdges) {
    DefectFields& df = this->fields;
    int n_failures = 0;
    std::string first_failure;
    auto check_family = [&](std::vector<uint8_t>& flags, const char* family) {
        for (std::size_t i = 0; i < flags.size(); ++i) {
            flags[i] = 1;
            const auto edges = this->finder.EmitVoxelEdges(df);
            flags[i] = 0;
            if (!edges.empty() && n_failures++ == 0) {
                first_failure = std::string(family) + "[" + std::to_string(i) + "] emitted "
                              + std::to_string(edges.size()) + " edges";
            }
        }
    };
    check_family(df.def_x, "def_x");
    check_family(df.def_y, "def_y");
    check_family(df.def_z, "def_z");
    EXPECT_EQ(n_failures, 0) << "first failure: " << first_failure;
}

// For every voxel and every one of its 15 face pairs, firing exactly those two
// faces must emit exactly one edge, between them. Exactly one rather than "at
// least one": a pair emitted from two different voxel visits would be
// double-counted.
TYPED_TEST(VoxelEdgesTest, EveryVoxelFacePairEmitsExactlyThatEdge) {
    DefectFields& df = this->fields;
    constexpr std::array<int, 6> pt = {1, 2, 3, 4, 5, 6};

    int n_failures = 0;
    std::vector<std::string> examples;
    for (int z = 0; z < this->vz; ++z) {
        for (int y = 0; y < this->vy; ++y) {
            for (int x = 0; x < this->vx; ++x) {
                const auto faces = VoxelFaces(x, y, z);
                for (int a = 0; a < 6; ++a) {
                    for (int b = a + 1; b < 6; ++b) {
                        Flag(df, faces[a]) = 1;
                        Flag(df, faces[b]) = 1;
                        const auto edges = this->finder.EmitVoxelEdges(df);
                        Flag(df, faces[a]) = 0;
                        Flag(df, faces[b]) = 0;

                        const auto expected = Canonical(Fid(df, faces[a]), Fid(df, faces[b]));
                        const bool ok = edges.size() == 1
                                     && Canonical(edges[0].first, edges[0].second) == expected;
                        if (!ok && n_failures++ < 8) {
                            examples.push_back(
                                "voxel (" + std::to_string(x) + "," + std::to_string(y) + ","
                                + std::to_string(z) + ") pair " + std::to_string(pt[a]) + "-"
                                + std::to_string(pt[b]) + ": " + std::to_string(edges.size())
                                + " edges emitted");
                        }
                    }
                }
            }
        }
    }

    std::string report;
    for (const auto& e : examples) report += "\n  " + e;
    EXPECT_EQ(n_failures, 0) << "first failures:" << report;
}


// ── Director-level regression ────────────────────────────────────────────────
// A straight +1/2 wedge line from the x = 0 wall to the x = nx-1 wall, tilted
// in z so that it enters the last voxel layer through the voxel's high-z face
// (pt 5) and leaves through the wall plaquette (pt 1). That 1-5 pair is one of
// the three emitted from the +x neighbour, so it guards the plumbing from
// winding numbers through to the extracted path, not just EmitVoxelEdges.
//
// All walled, so the field needs no periodic image: a director that is not
// periodic would put spurious defects on the seams. y = 3.5 keeps the line off
// every vertex and parallel to the y-normal plaquettes.
TEST(VoxelEdgesDirector, WedgeLineEndingOnXWallKeepsWallPlaquette) {
    static_assert(Params::nx == 8 && Params::ny == 8 && Params::nz == 8,
                  "line geometry below is laid out for the 8^3 unit grid");
    using BC = AllWalledConfig;

    // Line through p0 along t = (1, 0, -1/4)/|.|, so z = 4 - (x - 6.4)/4:
    //   x = 0: z = 5.6,  crosses z = 5 at x = 2.4,  z = 4 at x = 6.4,  x = 7: z = 3.85.
    const std::array<double, 3> p0 = {6.4, 3.5, 4.0};
    const double tn = std::sqrt(1.0 + 0.25 * 0.25);
    const std::array<double, 3> t  = {1.0 / tn, 0.0, -0.25 / tn};
    const std::array<double, 3> e1 = {0.0, 1.0, 0.0};      // ⊥ t
    const std::array<double, 3> e2 = {-t[2], 0.0, t[0]};   // t × e1

    AnalysisFields af;
    for (int z = 0; z < Params::nz; ++z) {
        for (int y = 0; y < Params::ny; ++y) {
            for (int x = 0; x < Params::nx; ++x) {
                const std::array<double, 3> d = {x - p0[0], y - p0[1], z - p0[2]};
                const double u = d[0]*e1[0] + d[1]*e1[1] + d[2]*e1[2];
                const double v = d[0]*e2[0] + d[1]*e2[1] + d[2]*e2[2];
                const double half_phi = 0.5 * std::atan2(v, u);
                for (int c = 0; c < 3; ++c) {
                    af.director_[af.grid.halo_dirIdx(x, y, z, c)] =
                        std::cos(half_phi) * e1[c] + std::sin(half_phi) * e2[c];
                }
            }
        }
    }

    DefectFields df(periodicity_by_axis<BC>);
    DefectFinder<BC> finder;
    finder.FindDefects(af, df);

    // Plaquette centres the line pierces, low-x wall to high-x wall.
    std::vector<std::array<double, 3>> expected = {
        {0, 3.5, 5.5}, {1, 3.5, 5.5}, {2, 3.5, 5.5},
        {2.5, 3.5, 5},                                  // def_z(2,3,5)
        {3, 3.5, 4.5}, {4, 3.5, 4.5}, {5, 3.5, 4.5}, {6, 3.5, 4.5},
        {6.5, 3.5, 4},                                  // def_z(6,3,4): pt 5 of voxel (6,3,3)
        {7, 3.5, 3.5},                                  // def_x(7,3,3): wall plaquette, pt 1
    };

    ASSERT_EQ(df.disclinations.size(), 1u);
    const std::vector<double>& pts = df.disclinations[0].points;
    ASSERT_EQ(pts.size(), 3 * expected.size());
    std::vector<std::array<double, 3>> got;
    for (std::size_t i = 0; i < pts.size(); i += 3) got.push_back({pts[i], pts[i+1], pts[i+2]});

    // FindPaths may walk the line from either end.
    if (got.front() != expected.front()) std::reverse(got.begin(), got.end());
    EXPECT_EQ(got, expected);
}

} // namespace
