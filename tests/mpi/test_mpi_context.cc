#include <mpi.h>
#include <gtest/gtest.h>
#include "mpi/mpi_context.h"
#include <array>
#include <set>
#include <vector>

TEST(MPIContext, DimsProductEqualsWorldSize) {
    MPIContext ctx;
    EXPECT_EQ(ctx.dims[0] * ctx.dims[1] * ctx.dims[2], ctx.world_size);
}

TEST(MPIContext, CoordsBoundedByDims) {
    MPIContext ctx;
    for (int d = 0; d < 3; ++d)
        EXPECT_LT(ctx.coords[d], ctx.dims[d]);
}

TEST(MPIContext, CoordsAreUniqueAcrossRanks) {
    MPIContext ctx;
    // Gather all coords to rank 0 and check for duplicates
    std::array<int, 3> my_coords = {ctx.coords[0], ctx.coords[1], ctx.coords[2]};
    std::vector<int> all_coords(ctx.world_size * 3);
    MPI_Gather(my_coords.data(), 3, MPI_INT,
               all_coords.data(), 3, MPI_INT, 0, ctx.cart_comm);
    if (ctx.world_rank == 0) {
        std::set<std::array<int,3>> seen;
        for (int r = 0; r < ctx.world_size; ++r) {
            std::array<int,3> c = {all_coords[3*r], all_coords[3*r+1], all_coords[3*r+2]};
            EXPECT_TRUE(seen.insert(c).second) << "duplicate coords at rank " << r;
        }
    }
}

TEST(MPIContext, CartCommHasCorrectSize) {
    MPIContext ctx;
    int size;
    MPI_Comm_size(ctx.cart_comm, &size);
    EXPECT_EQ(size, ctx.world_size);
}
