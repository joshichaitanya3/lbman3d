#include <mpi.h>
#include <gtest/gtest.h>
#include "mpi/mpi_context.h"
#include "mpi/halo_exchange_qtensor.h"
#include <vector>
#include "params.h"

TEST(HaloExchangeQTensor, PackXYBufferContents) {
    MPIContext mpi;
    LocalGrid grid = mpi.MakeLocalGrid();
    HaloExchangeQTensor halo(grid, mpi);

    // fill with value encoding position
    std::vector<double> field(grid.HaloVolume());
    for (int z = 0; z < grid.local_nz; ++z)
        for (int y = 0; y < grid.local_ny; ++y)
            for (int x = 0; x < grid.local_nx; ++x)
                field[grid.halo_idx(x, y, z)] = (x+1) + (y+1) * 100.0 + (z+1) * 10000.0;

    halo.PackField(field.data(), 0, /*d=z*/2);

    // lo-z face (send_buf_[4]): should contain z=1 values
    for (int y = 1; y <= grid.local_ny; ++y)
        for (int x = 1; x <= grid.local_nx; ++x)
            EXPECT_DOUBLE_EQ(halo.send_buf_[4][(y-1)*grid.local_nx + (x-1)],
                             x + y * 100.0 + 1 * 10000.0);
}

TEST(HaloExchangeQTensor, PackXZBufferContents) {
    MPIContext mpi;
    LocalGrid grid = mpi.MakeLocalGrid();
    HaloExchangeQTensor halo(grid, mpi);

    // fill with value encoding position
    std::vector<double> field(grid.HaloVolume());
    for (int z = 0; z < grid.local_nz; ++z)
        for (int y = 0; y < grid.local_ny; ++y)
            for (int x = 0; x < grid.local_nx; ++x)
                field[grid.halo_idx(x, y, z)] = (x+1) + (y+1) * 100.0 + (z+1) * 10000.0;

    halo.PackField(field.data(), 0, /*d=y*/1);

    // hi-y face (send_buf_[3]): should contain y=ny values
    for (int z = 1; z <= grid.local_nz; ++z)
        for (int x = 1; x <= grid.local_nx; ++x)
            EXPECT_DOUBLE_EQ(halo.send_buf_[3][(z-1)*grid.local_nx + (x-1)],
                             x + grid.local_ny * 100.0 + z * 10000.0);
}

TEST(HaloExchangeQTensor, PackYZBufferContents) {
    MPIContext mpi;
    LocalGrid grid = mpi.MakeLocalGrid();
    HaloExchangeQTensor halo(grid, mpi);

    // fill with value encoding position
    std::vector<double> field(grid.HaloVolume());
    for (int z = 0; z < grid.local_nz; ++z)
        for (int y = 0; y < grid.local_ny; ++y)
            for (int x = 0; x < grid.local_nx; ++x)
                field[grid.halo_idx(x, y, z)] = (x+1) + (y+1) * 100.0 + (z+1) * 10000.0;

    halo.PackField(field.data(), 0, /*d=x*/0);

    // lo-x face (send_buf_[0]): should contain x=1 values
    for (int z = 1; z <= grid.local_nz; ++z)
        for (int y = 1; y <= grid.local_ny; ++y)
            EXPECT_DOUBLE_EQ(halo.send_buf_[0][(z-1)*grid.local_ny + (y-1)],
                             1 + y * 100.0 + z * 10000.0);
}

TEST(HaloExchangeQTensor, PackYZMultipleFields) {
    MPIContext mpi;
    LocalGrid grid = mpi.MakeLocalGrid();
    HaloExchangeQTensor halo(grid, mpi);

    // fill with value encoding position
    std::vector<double> field(grid.HaloVolume());
    for (int z = 0; z < grid.local_nz; ++z)
        for (int y = 0; y < grid.local_ny; ++y)
            for (int x = 0; x < grid.local_nx; ++x)
                field[grid.halo_idx(x, y, z)] = (x+1) + (y+1) * 100.0 + (z+1) * 10000.0;

    // fill with value encoding position
    std::vector<double> neg_field(grid.HaloVolume());
    for (int z = 0; z < grid.local_nz; ++z)
        for (int y = 0; y < grid.local_ny; ++y)
            for (int x = 0; x < grid.local_nx; ++x)
                neg_field[grid.halo_idx(x, y, z)] = -((x+1) + (y+1) * 100.0 + (z+1) * 10000.0);

    halo.PackField(field.data()    , 0, /*d=x*/0);
    halo.PackField(neg_field.data(), 1, /*d=x*/0);

    int face_size = halo.max_yz;
    // lo-x face (send_buf_[0]): should contain x=1 values
    for (int z = 1; z <= grid.local_nz; ++z) {
        for (int y = 1; y <= grid.local_ny; ++y) {
            EXPECT_DOUBLE_EQ(halo.send_buf_[0][(z-1)*grid.local_ny + (y-1)],
                             1 + y * 100.0 + z * 10000.0);
            EXPECT_DOUBLE_EQ(halo.send_buf_[0][face_size + (z-1)*grid.local_ny + (y-1)],
            -(1 + y * 100.0 + z * 10000.0));
        }
    }
}

TEST(HaloExchangeQTensor, BufferSizes) {
    MPIContext mpi;
    HaloExchangeQTensor halo(mpi.MakeLocalGrid(), mpi);
    // ExchangePassiveStresses is the widest pack (10 fields per face).
    constexpr size_t nf = HaloExchangeQTensor::kMaxFields;
    EXPECT_EQ(halo.send_buf_[0].size(), nf * halo.max_yz);
    EXPECT_EQ(halo.send_buf_[2].size(), nf * halo.max_xz);
    EXPECT_EQ(halo.send_buf_[4].size(), nf * halo.max_xy);
}
