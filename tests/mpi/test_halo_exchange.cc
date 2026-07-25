#include <mpi.h>
#include <gtest/gtest.h>
#include "mpi/mpi_context.h"
#include "mpi/halo_exchange.h"
#include <array>
#include <set>
#include <vector>
#include "params.h"
#include "physics_helpers.h"

TEST(HaloExchange, PackXYBufferContents) {
    MPIContext mpi;
    LocalGrid grid = mpi.MakeLocalGrid();
    HaloExchange halo(grid, mpi);

    // fill with value encoding position
    std::vector<double> field(Params::nx * Params::ny * Params::nz);
    for (int z = 1; z <= Params::nz; ++z)
        for (int y = 1; y <= Params::ny; ++y)
            for (int x = 1; x <= Params::nx; ++x)
                field[idx(x, y, z)] = x + y * 100.0 + z * 10000.0;

    halo.PackField(field.data(), 0, /*d=z*/2);

    // lo-z face (send_buf_[4]): should contain z=1 values
    for (int y = 1; y <= Params::ny; ++y)
        for (int x = 1; x <= Params::nx; ++x)
            EXPECT_DOUBLE_EQ(halo.send_buf_[4][(y-1)*Params::nx + (x-1)],
                             x + y * 100.0 + 1 * 10000.0);
}

TEST(HaloExchange, PackXZBufferContents) {
    MPIContext mpi;
    LocalGrid grid = mpi.MakeLocalGrid();
    HaloExchange halo(grid, mpi);

    // fill with value encoding position
    std::vector<double> field(Params::nx * Params::ny * Params::nz);
    for (int z = 1; z <= Params::nz; ++z)
        for (int y = 1; y <= Params::ny; ++y)
            for (int x = 1; x <= Params::nx; ++x)
                field[idx(x, y, z)] = x + y * 100.0 + z * 10000.0;

    halo.PackField(field.data(), 0, /*d=y*/1);

    // hi-y face (send_buf_[3]): should contain y=ny values
    for (int z = 1; z <= Params::nz; ++z)
        for (int x = 1; x <= Params::nx; ++x)
            EXPECT_DOUBLE_EQ(halo.send_buf_[3][(z-1)*Params::nx + (x-1)],
                             x + Params::ny * 100.0 + z * 10000.0);
}

TEST(HaloExchange, PackYZBufferContents) {
    MPIContext mpi;
    LocalGrid grid = mpi.MakeLocalGrid();
    HaloExchange halo(grid, mpi);

    // fill with value encoding position
    std::vector<double> field(Params::nx * Params::ny * Params::nz);
    for (int z = 1; z <= Params::nz; ++z)
        for (int y = 1; y <= Params::ny; ++y)
            for (int x = 1; x <= Params::nx; ++x)
                field[idx(x, y, z)] = x + y * 100.0 + z * 10000.0;

    halo.PackField(field.data(), 0, /*d=x*/0);

    // lo-x face (send_buf_[0]): should contain x=1 values
    for (int z = 1; z <= Params::nz; ++z)
        for (int y = 1; y <= Params::ny; ++y)
            EXPECT_DOUBLE_EQ(halo.send_buf_[0][(z-1)*Params::ny + (y-1)],
                             1 + y * 100.0 + z * 10000.0);
}

TEST(HaloExchange, PackYZMultipleFields) {
    MPIContext mpi;
    LocalGrid grid = mpi.MakeLocalGrid();
    HaloExchange halo(grid, mpi);

    // fill with value encoding position
    std::vector<double> field(Params::nx * Params::ny * Params::nz);
    for (int z = 1; z <= Params::nz; ++z)
        for (int y = 1; y <= Params::ny; ++y)
            for (int x = 1; x <= Params::nx; ++x)
                field[idx(x, y, z)] = x + y * 100.0 + z * 10000.0;

    // fill with value encoding position
    std::vector<double> neg_field(Params::nx * Params::ny * Params::nz);
    for (int z = 1; z <= Params::nz; ++z)
        for (int y = 1; y <= Params::ny; ++y)
            for (int x = 1; x <= Params::nx; ++x)
                neg_field[idx(x, y, z)] = -(x + y * 100.0 + z * 10000.0);
    
    halo.PackField(field.data()    , 0, /*d=x*/0);
    halo.PackField(neg_field.data(), 1, /*d=x*/0);

    int face_size = halo.max_yz;
    // lo-x face (send_buf_[0]): should contain x=1 values
    for (int z = 1; z <= Params::nz; ++z) {
        for (int y = 1; y <= Params::ny; ++y) {
            EXPECT_DOUBLE_EQ(halo.send_buf_[0][(z-1)*Params::ny + (y-1)],
                             1 + y * 100.0 + z * 10000.0);
            EXPECT_DOUBLE_EQ(halo.send_buf_[0][face_size + (z-1)*Params::ny + (y-1)],
            -(1 + y * 100.0 + z * 10000.0));
        }
    }
}

TEST(HaloExchange, BufferSizes) {
    MPIContext mpi;
    HaloExchange halo(mpi.MakeLocalGrid(), mpi);
    size_t nf = static_cast<size_t>(std::max(Lattice::ndir, 8));
    EXPECT_EQ(halo.send_buf_[0].size(), nf * halo.max_yz);
    EXPECT_EQ(halo.send_buf_[2].size(), nf * halo.max_xz);
    EXPECT_EQ(halo.send_buf_[4].size(), nf * halo.max_xy);
}

