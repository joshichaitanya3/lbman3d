#include <gtest/gtest.h>
#include <mpi.h>
#include <hdf5.h>
#include <vector>
#include <ranges>
#include <string>
#include <cstdio>
#include <unistd.h>
#include "mpi/mpi_context.h"
#include "vtkhdf_writer.h"
#include <params.h>

using namespace Params;

// Rank 0 picks a path using its PID (unique per mpiexec invocation) and
// broadcasts it. This avoids deriving the path from world_size, which can be
// wrong on systems where OpenMPI's PMI state leaks between successive mpiexec
// calls (one process attaches to the previous job's stale communicator and
// sees world_size=1 instead of the expected N).
static std::string unique_path(const char* tag) {
    char buf[256] = {};
    if (MPIContext::IsRoot())
        std::snprintf(buf, sizeof(buf), "./lbman3d_test_vtkhdf_%s_%d.vtkhdf",
                      tag, static_cast<int>(getpid()));
    MPI_Bcast(buf, sizeof(buf), MPI_CHAR, 0, MPI_COMM_WORLD);
    return buf;
}

// Scalar pattern: value at global (xg, yg, zg) = (xg+1)*(yg+1)*(zg+1).
// Vector pattern: component i gets base*(i+1), where base = (xg+1)*(yg+1)*(zg+1).
// Both are separable, positive, and unique on the 8x8x8 unit grid.

TEST(VTKHDFWrite, ScalarFieldRoundtrip) {
    MPIContext mpi;
    LocalGrid grid = mpi.MakeLocalGrid();

    std::vector<double> field(grid.HaloVolume(), 0.0);
    for (int z : std::views::iota(0, grid.local_nz))
        for (int y : std::views::iota(0, grid.local_ny))
            for (int x : std::views::iota(0, grid.local_nx))
                field[grid.halo_idx(x, y, z)] =
                    static_cast<double>((grid.offset_x + x + 1) *
                                        (grid.offset_y + y + 1) *
                                        (grid.offset_z + z + 1));

    const std::string path = unique_path("scalar");
    {
        ImageDataWriter writer(path, mpi);
        writer.WriteScalarField("test_field", field.data(), grid);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (!MPIContext::IsRoot()) return;

    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(fid, 0) << "could not open output file " << path;

    hid_t ds = H5Dopen2(fid, "/VTKHDF/PointData/test_field", H5P_DEFAULT);
    ASSERT_GE(ds, 0) << "could not open dataset";

    std::vector<double> buf(nx * ny * nz);
    herr_t st = H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    EXPECT_EQ(st, 0) << "H5Dread failed";

    H5Dclose(ds);
    H5Fclose(fid);
    std::remove(path.c_str());

    // Dataset layout is [nz, ny, nx] (z slowest, x fastest).
    for (int zg = 0; zg < nz; ++zg)
        for (int yg = 0; yg < ny; ++yg)
            for (int xg = 0; xg < nx; ++xg) {
                double expected = static_cast<double>((xg+1)*(yg+1)*(zg+1));
                EXPECT_DOUBLE_EQ(buf[zg*ny*nx + yg*nx + xg], expected)
                    << "mismatch at global (" << xg << ',' << yg << ',' << zg << ')';
            }
}

TEST(VTKHDFWrite, VectorFieldRoundtrip) {
    MPIContext mpi;
    LocalGrid grid = mpi.MakeLocalGrid();

    constexpr int kComponents = 3;
    std::vector<double> field(grid.HaloVolume() * kComponents, 0.0);
    for (int z : std::views::iota(0, grid.local_nz))
        for (int y : std::views::iota(0, grid.local_ny))
            for (int x : std::views::iota(0, grid.local_nx)) {
                double base = static_cast<double>((grid.offset_x + x + 1) *
                                                   (grid.offset_y + y + 1) *
                                                   (grid.offset_z + z + 1));
                for (int i = 0; i < kComponents; ++i)
                    field[grid.halo_idx(x, y, z, i)] = base * (i + 1);
            }

    const std::string path = unique_path("vector");
    {
        ImageDataWriter writer(path, mpi);
        writer.WriteVectorField("test_field", field.data(), grid, kComponents);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (!MPIContext::IsRoot()) return;

    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(fid, 0) << "could not open output file " << path;

    hid_t ds = H5Dopen2(fid, "/VTKHDF/PointData/test_field", H5P_DEFAULT);
    ASSERT_GE(ds, 0) << "could not open dataset";

    std::vector<double> buf(nx * ny * nz * kComponents);
    herr_t st = H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    EXPECT_EQ(st, 0) << "H5Dread failed";

    H5Dclose(ds);
    H5Fclose(fid);
    std::remove(path.c_str());

    // Dataset layout is [nz, ny, nx, components] (z slowest, component fastest).
    for (int zg = 0; zg < nz; ++zg)
        for (int yg = 0; yg < ny; ++yg)
            for (int xg = 0; xg < nx; ++xg) {
                double base = static_cast<double>((xg+1)*(yg+1)*(zg+1));
                for (int i = 0; i < kComponents; ++i) {
                    double expected = base * (i + 1);
                    EXPECT_DOUBLE_EQ(buf[(zg*ny*nx + yg*nx + xg)*kComponents + i], expected)
                        << "mismatch at global (" << xg << ',' << yg << ',' << zg
                        << ") component " << i;
                }
            }
}
