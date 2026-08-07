#include <gtest/gtest.h>
#include "mpi/mpi_context.h"

#ifndef LBM_ENABLE_MPI   // guard so it only compiles in CPU build
TEST(MPIContextStub, DefaultValues) {
    MPIContext ctx;
    EXPECT_EQ(ctx.world_rank, 0);
    EXPECT_EQ(ctx.world_size, 1);
    EXPECT_EQ(ctx.dims[0] * ctx.dims[1] * ctx.dims[2], 1);
    EXPECT_EQ(ctx.coords[0], 0);
    EXPECT_EQ(ctx.coords[1], 0);
    EXPECT_EQ(ctx.coords[2], 0);
}
#endif
