// PR VII-f — 2-rank halo round-trip test for HaloExchangePassiveStressesNvshmem.
//
// Mirrors test_exchange_qtensor_nvshmem.cu structure but exercises the 13-
// field passive-stress exchange (5 Q + 5 Σ + 3 τ) instead of the 8-field
// Q + velocity exchange. Each rank fills its owned Q/Σ/τ cells with its own
// MPI rank number, runs ExchangePassiveStresses, and asserts every field's
// ghost cells now hold the neighbouring rank's number. This validates the
// full pack → put → barrier → unpack chain for the wider field bundle
// without depending on any physics kernel.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <vector>

#include "cuda/halo_exchange_passive_stresses_nvshmem.h"
#include "device_fields.h"
#include "local_grid.h"
#include "mpi/mpi_context.h"
#include "params.h"

namespace {

class ExchangePassiveStressesNvshmem : public ::testing::Test {
protected:
    static inline MPIContext* mpi = nullptr;
    static inline LocalGrid grid;
    static inline BackendInfo backend;
    static inline HaloExchangePassiveStressesNvshmem* halo = nullptr;
    static inline DeviceFields* d_fields = nullptr;

    static void SetUpTestSuite() {
        mpi = new MPIContext(/*periods=*/{1, 1, 1});
        grid = mpi->MakeLocalGrid();
        backend = InitializeComputeBackend(*mpi, grid);
        d_fields = new DeviceFields(grid);
        halo = new HaloExchangePassiveStressesNvshmem(grid, *mpi);
    }

    static void TearDownTestSuite() {
        delete halo;     halo     = nullptr;
        delete d_fields; d_fields = nullptr;
        delete mpi;      mpi      = nullptr;
    }

    static void FillFieldOwned(double* d_field, double value) {
        std::vector<double> host(grid.HaloVolume(), 0.0);
        for (int z = 0; z < grid.local_nz; ++z)
            for (int y = 0; y < grid.local_ny; ++y)
                for (int x = 0; x < grid.local_nx; ++x)
                    host[grid.halo_idx(x, y, z)] = value;
        cudaMemcpy(d_field, host.data(), host.size() * sizeof(double),
                   cudaMemcpyHostToDevice);
    }

    static std::vector<double> CopyFieldToHost(const double* d_field) {
        std::vector<double> host(grid.HaloVolume(), 0.0);
        cudaMemcpy(host.data(), d_field, host.size() * sizeof(double),
                   cudaMemcpyDeviceToHost);
        return host;
    }
};

// Single-rank fast path: ExchangePassiveStresses early-returns on
// world_size == 1, so ghosts stay at their initial value.
TEST_F(ExchangePassiveStressesNvshmem, SingleRankIsNoOp) {
    if (mpi->world_size != 1) GTEST_SKIP() << "np != 1";

    const double owned_tag = 42.0;
    FillFieldOwned(d_fields->d_Sigma_xx, owned_tag);
    halo->ExchangePassiveStresses(*d_fields);
    cudaDeviceSynchronize();

    auto host = CopyFieldToHost(d_fields->d_Sigma_xx);
    for (int y = 0; y < grid.local_ny; ++y)
        for (int x = 0; x < grid.local_nx; ++x) {
            EXPECT_DOUBLE_EQ(host[grid.halo_idx(x, y, -1)],             0.0);
            EXPECT_DOUBLE_EQ(host[grid.halo_idx(x, y, grid.local_nz)],  0.0);
        }
    // Owned untouched too.
    for (int z = 0; z < grid.local_nz; ++z)
        for (int y = 0; y < grid.local_ny; ++y)
            for (int x = 0; x < grid.local_nx; ++x)
                EXPECT_DOUBLE_EQ(host[grid.halo_idx(x, y, z)], owned_tag);
}

// After ExchangePassiveStresses, ghosts on axes where the neighbour is a
// real rank (not MPI_PROC_NULL) must carry the neighbour's tag on ALL 13
// exchanged fields. Ghosts at physical walls (MPI_PROC_NULL) are untouched
// and stay at their initial value (0.0 in this fixture).
TEST_F(ExchangePassiveStressesNvshmem, GhostsHoldNeighbourRankOnAllFields) {
    if (mpi->world_size == 1) GTEST_SKIP() << "np == 1 exchange is a no-op";
    const double my_tag = static_cast<double>(mpi->world_rank) + 1.0;

    double* fields[13] = {
        d_fields->d_qxx, d_fields->d_qxy, d_fields->d_qxz,
        d_fields->d_qyy, d_fields->d_qyz,
        d_fields->d_Sigma_xx, d_fields->d_Sigma_xy, d_fields->d_Sigma_xz,
        d_fields->d_Sigma_yy, d_fields->d_Sigma_yz,
        d_fields->d_Tau_xy, d_fields->d_Tau_xz, d_fields->d_Tau_yz
    };

    for (int f = 0; f < 13; ++f) FillFieldOwned(fields[f], my_tag);

    halo->ExchangePassiveStresses(*d_fields);
    cudaDeviceSynchronize();

    auto expected_tag_lo = [&](int axis) -> double {
        return halo->neighbor_lo_[axis] == MPI_PROC_NULL
            ? 0.0
            : static_cast<double>(halo->neighbor_lo_[axis]) + 1.0;
    };
    auto expected_tag_hi = [&](int axis) -> double {
        return halo->neighbor_hi_[axis] == MPI_PROC_NULL
            ? 0.0
            : static_cast<double>(halo->neighbor_hi_[axis]) + 1.0;
    };

    for (int f = 0; f < 13; ++f) {
        auto host = CopyFieldToHost(fields[f]);

        for (int z = 0; z < grid.local_nz; ++z) {
            for (int y = 0; y < grid.local_ny; ++y) {
                EXPECT_DOUBLE_EQ(host[grid.halo_idx(-1, y, z)],
                                 expected_tag_lo(0)) << "field=" << f;
                EXPECT_DOUBLE_EQ(host[grid.halo_idx(grid.local_nx, y, z)],
                                 expected_tag_hi(0)) << "field=" << f;
            }
        }
        for (int z = 0; z < grid.local_nz; ++z) {
            for (int x = 0; x < grid.local_nx; ++x) {
                EXPECT_DOUBLE_EQ(host[grid.halo_idx(x, -1, z)],
                                 expected_tag_lo(1)) << "field=" << f;
                EXPECT_DOUBLE_EQ(host[grid.halo_idx(x, grid.local_ny, z)],
                                 expected_tag_hi(1)) << "field=" << f;
            }
        }
        for (int y = 0; y < grid.local_ny; ++y) {
            for (int x = 0; x < grid.local_nx; ++x) {
                EXPECT_DOUBLE_EQ(host[grid.halo_idx(x, y, -1)],
                                 expected_tag_lo(2)) << "field=" << f;
                EXPECT_DOUBLE_EQ(host[grid.halo_idx(x, y, grid.local_nz)],
                                 expected_tag_hi(2)) << "field=" << f;
            }
        }
    }
}

// Owned cells must be untouched — a stray pack that reads ghost or an unpack
// that writes owned would flip these. Check on one Σ and one τ so a
// slot-ordering bug in either the symmetric or antisymmetric block surfaces.
TEST_F(ExchangePassiveStressesNvshmem, OwnedCellsPreserved) {
    const double my_tag = static_cast<double>(mpi->world_rank) + 1.0;
    FillFieldOwned(d_fields->d_Sigma_yz, my_tag);
    FillFieldOwned(d_fields->d_Tau_xy,   my_tag);

    halo->ExchangePassiveStresses(*d_fields);
    cudaDeviceSynchronize();

    for (double* f : {d_fields->d_Sigma_yz, d_fields->d_Tau_xy}) {
        auto host = CopyFieldToHost(f);
        for (int z = 0; z < grid.local_nz; ++z)
            for (int y = 0; y < grid.local_ny; ++y)
                for (int x = 0; x < grid.local_nx; ++x)
                    EXPECT_DOUBLE_EQ(host[grid.halo_idx(x, y, z)], my_tag);
    }
}

}  // namespace
