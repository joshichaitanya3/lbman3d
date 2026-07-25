#include <mpi.h>
#include <gtest/gtest.h>
#include "mpi/mpi_context.h"
#include "mpi/halo_exchange.h"
#include <array>
#include <set>
#include <vector>
#include "params.h"
#include "physics_helpers.h"
#include "fluid_fields.h"
#include "qtensor_fields.h"

TEST(HaloExchange, ExchangeSendsCorrectFace) {
    MPIContext mpi;
    HaloExchange halo(mpi.MakeLocalGrid(), mpi);

    FluidFields ff;
    QTensorFields qf;
    std::fill(ff.ux.begin(), ff.ux.end(), mpi.world_rank);
    std::fill(ff.uy.begin(), ff.uy.end(), mpi.world_rank);
    std::fill(ff.uz.begin(), ff.uz.end(), mpi.world_rank);

    std::fill(qf.qxx.begin(), qf.qxx.end(), mpi.world_rank);
    std::fill(qf.qxy.begin(), qf.qxy.end(), mpi.world_rank);
    std::fill(qf.qxz.begin(), qf.qxz.end(), mpi.world_rank);
    std::fill(qf.qyy.begin(), qf.qyy.end(), mpi.world_rank);
    std::fill(qf.qyz.begin(), qf.qyz.end(), mpi.world_rank);
    
    halo.ExchangeQTensor(qf, ff);

    EXPECT_DOUBLE_EQ(halo.recv_buf_[4][0], halo.neighbor_lo_[2]);
}
