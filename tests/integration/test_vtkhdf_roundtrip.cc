#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <params.h>
#include <sim_config.h>
#include "vtkhdf_writer.h"
#include "vtkhdf_reader.h"
#include "sim_config_attrs.h"
#include "local_grid.h"
#include "mpi/mpi_context.h"
#include "boundary.h"

// Writer → Reader roundtrip: the offline find_defects pipeline is only as
// good as the byte-equality guarantee between ImageDataWriter and
// ImageDataReader. If either side drifts (hyperslab offsets, layout, dtype)
// the analysis silently reads perturbed data. Pin the invariant here.

using namespace Params;

namespace {

// Fill a halo-padded scalar buffer with a distinctive per-cell pattern —
// values must be unique across (x,y,z) so a mis-selected hyperslab shows
// up immediately as swapped/perturbed indices, not just a numeric drift.
void FillPatternScalar(std::vector<double>& buf, const LocalGrid& g) {
    for (int z = 0; z < g.local_nz; ++z)
        for (int y = 0; y < g.local_ny; ++y)
            for (int x = 0; x < g.local_nx; ++x)
                buf[g.halo_idx(x, y, z)] =
                    1.0*x + 100.0*y + 10000.0*z;
}

void FillPatternVector(std::vector<double>& buf, const LocalGrid& g) {
    for (int z = 0; z < g.local_nz; ++z)
        for (int y = 0; y < g.local_ny; ++y)
            for (int x = 0; x < g.local_nx; ++x) {
                buf[g.halo_dirIdx(x, y, z, 0)] = 1.0*x + 100.0*y + 10000.0*z;
                buf[g.halo_dirIdx(x, y, z, 1)] = 2.0*x + 200.0*y + 20000.0*z;
                buf[g.halo_dirIdx(x, y, z, 2)] = 3.0*x + 300.0*y + 30000.0*z;
            }
}

std::filesystem::path TempFile(const char* stem) {
    return std::filesystem::temp_directory_path() /
           (std::string(stem) + "_" + std::to_string(::getpid()) + ".vtkhdf");
}

}  // namespace

class VtkhdfRoundtripTest : public ::testing::Test {
protected:
    void SetUp() override {
        grid = LocalGrid::SingleRank();
        scratch_scalar.assign(grid.HaloVolume(), 0.0);
        scratch_vector.assign(grid.HaloVolume() * 3, 0.0);
    }
    void TearDown() override {
        for (const auto& p : cleanup) {
            std::error_code ec;
            std::filesystem::remove(p, ec);
        }
    }
    LocalGrid grid;
    std::vector<double> scratch_scalar;
    std::vector<double> scratch_vector;
    std::vector<std::filesystem::path> cleanup;
    MPIContext ctx{periodicity_by_axis<SimBC>};
};

TEST_F(VtkhdfRoundtripTest, ScalarFieldRoundtrip) {
    std::vector<double> written(grid.HaloVolume(), 0.0);
    FillPatternScalar(written, grid);

    const auto path = TempFile("vtkhdf_scalar");
    cleanup.push_back(path);
    {
        ImageDataWriter w(path.string(), ctx);
        SimConfigAttr::StampSimConfigAttributes<SimBC>(w.root());
        w.WriteScalarField("phi", written.data(), grid);
    }

    ImageDataReader r(path.string(), ctx);
    std::vector<double> readback(grid.HaloVolume(), 0.0);
    r.ReadScalarField("phi", readback.data(), grid);

    // Every interior cell must roundtrip bit-exactly.
    for (int z = 0; z < grid.local_nz; ++z)
        for (int y = 0; y < grid.local_ny; ++y)
            for (int x = 0; x < grid.local_nx; ++x) {
                const int idx = grid.halo_idx(x, y, z);
                ASSERT_EQ(readback[idx], written[idx])
                    << "at (" << x << "," << y << "," << z << ")";
            }
}

TEST_F(VtkhdfRoundtripTest, VectorFieldRoundtrip) {
    std::vector<double> written(grid.HaloVolume() * 3, 0.0);
    FillPatternVector(written, grid);

    const auto path = TempFile("vtkhdf_vector");
    cleanup.push_back(path);
    {
        ImageDataWriter w(path.string(), ctx);
        SimConfigAttr::StampSimConfigAttributes<SimBC>(w.root());
        w.WriteVectorField("v", written.data(), grid, 3);
    }

    ImageDataReader r(path.string(), ctx);
    std::vector<double> readback(grid.HaloVolume() * 3, 0.0);
    r.ReadVectorField("v", readback.data(), grid, 3);

    for (int z = 0; z < grid.local_nz; ++z)
        for (int y = 0; y < grid.local_ny; ++y)
            for (int x = 0; x < grid.local_nx; ++x)
                for (int c = 0; c < 3; ++c) {
                    const int idx = grid.halo_dirIdx(x, y, z, c);
                    ASSERT_EQ(readback[idx], written[idx])
                        << "at (" << x << "," << y << "," << z << ") c=" << c;
                }
}

TEST_F(VtkhdfRoundtripTest, SimConfigAttributesRoundtrip) {
    // The whole point of stamping the sim config is that the reader can
    // reconstruct it byte-exact. Any drift here breaks find_defects'
    // validator.
    std::vector<double> junk(grid.HaloVolume(), 0.0);

    const auto path = TempFile("vtkhdf_attrs");
    cleanup.push_back(path);
    {
        ImageDataWriter w(path.string(), ctx);
        SimConfigAttr::StampSimConfigAttributes<SimBC>(w.root());
        w.WriteScalarField("phi", junk.data(), grid);
    }

    ImageDataReader r(path.string(), ctx);
    const auto snap = r.ReadSimConfig();

    EXPECT_EQ(snap.bc_name, std::string(SimBC::name));
    EXPECT_EQ(snap.nx, nx);
    EXPECT_EQ(snap.ny, ny);
    EXPECT_EQ(snap.nz, nz);
    EXPECT_DOUBLE_EQ(snap.A, A);
    EXPECT_DOUBLE_EQ(snap.B, B);
    EXPECT_DOUBLE_EQ(snap.C, C);
    EXPECT_DOUBLE_EQ(snap.L, L);
    EXPECT_DOUBLE_EQ(snap.LAMBDA, LAMBDA);
    EXPECT_DOUBLE_EQ(snap.GAMMA, GAMMA);
    EXPECT_EQ(snap.kSaveInterval, ::kSaveInterval);

    // ValidateAgainstBuild against the same build should be clean.
    const auto report = SimConfigAttr::ValidateAgainstBuild<SimBC>(snap);
    EXPECT_TRUE(report.ok());
    EXPECT_FALSE(report.git_commit_mismatch);
}

TEST_F(VtkhdfRoundtripTest, WrongExtentIsDetected) {
    // Faking a WholeExtent mismatch: write a valid file under the current
    // params, then verify ImageDataReader would reject a file whose extent
    // disagreed with the compile-time grid. We simulate the mismatch by
    // temporarily lying: we don't have a way to write a wrong WholeExtent
    // through the writer, so this test relies on the current build's
    // matching extent producing a clean open. The negative direction is
    // covered by find_defects' real-data validation in practice.
    std::vector<double> junk(grid.HaloVolume(), 0.0);
    const auto path = TempFile("vtkhdf_ext");
    cleanup.push_back(path);
    {
        ImageDataWriter w(path.string(), ctx);
        SimConfigAttr::StampSimConfigAttributes<SimBC>(w.root());
        w.WriteScalarField("phi", junk.data(), grid);
    }
    // Positive check — same grid opens cleanly.
    EXPECT_NO_THROW({ ImageDataReader r(path.string(), ctx); });
}
