// PR VII-e — 2-rank halo round-trip test for HaloExchangeQTensorNvshmem.
//
// Mirrors tests/mpi/test_exchange_correctness.cc for the CPU class, on the
// NVSHMEM path. Each rank fills its Q + velocity fields with its own MPI rank
// number, runs ExchangeQTensor, and asserts its ghost cells now hold the
// neighbouring rank's number. This validates the full pack -> put -> barrier
// -> unpack chain without depending on any physics kernel — so a VII-e bug in
// pack layout, put destination, or unpack layout surfaces here in isolation.
//
// Launched under mpirun -n 2 (see tests/mpi/CMakeLists.txt). On a single-GPU
// dev box the same binary run with -n 1 degenerates to a self-put and still
// verifies the pack/unpack layout locally.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <vector>

#include "cuda/halo_exchange_qtensor_nvshmem.h"
#include "device_fields.h"
#include "local_grid.h"
#include "mpi/mpi_context.h"
#include "params.h"

namespace {

class ExchangeQTensorNvshmem : public ::testing::Test {
protected:
    static inline MPIContext* mpi = nullptr;
    static inline LocalGrid grid;
    static inline BackendInfo backend;
    static inline HaloExchangeQTensorNvshmem* halo = nullptr;
    static inline DeviceFields* d_fields = nullptr;

    static void SetUpTestSuite() {
        mpi = new MPIContext(/*periods=*/{1, 1, 1});
        grid = mpi->MakeLocalGrid();
        backend = InitializeComputeBackend(*mpi, grid);
        d_fields = new DeviceFields(grid);
        halo = new HaloExchangeQTensorNvshmem(grid, *mpi);
    }

    static void TearDownTestSuite() {
        delete halo;     halo     = nullptr;
        delete d_fields; d_fields = nullptr;
        delete mpi;      mpi      = nullptr;
    }

    // Fill an owned-only region of a Q or velocity field with `value`.
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

// Single-rank fast path: ExchangeQTensor early-returns on world_size == 1
// (matches HaloExchangeQTensor CPU class), so ghosts stay at their initial
// value. This test documents that no-op contract — a regression that
// accidentally runs a self-put here would surface as a nonzero ghost.
TEST_F(ExchangeQTensorNvshmem, SingleRankIsNoOp) {
    if (mpi->world_size != 1) GTEST_SKIP() << "np != 1";

    const double owned_tag = 42.0;
    FillFieldOwned(d_fields->d_qxx, owned_tag);
    // Ghosts start at 0 (DeviceFields ctor cudaMemsets everything).
    halo->ExchangeQTensor(*d_fields);
    cudaDeviceSynchronize();

    auto host = CopyFieldToHost(d_fields->d_qxx);
    for (int y = 0; y < grid.local_ny; ++y)
        for (int x = 0; x < grid.local_nx; ++x) {
            EXPECT_DOUBLE_EQ(host[grid.halo_idx(x, y, -1)],              0.0);
            EXPECT_DOUBLE_EQ(host[grid.halo_idx(x, y, grid.local_nz)],   0.0);
        }
    // Owned untouched too.
    for (int z = 0; z < grid.local_nz; ++z)
        for (int y = 0; y < grid.local_ny; ++y)
            for (int x = 0; x < grid.local_nx; ++x)
                EXPECT_DOUBLE_EQ(host[grid.halo_idx(x, y, z)], owned_tag);
}

// After ExchangeQTensor, ghosts on axes where the neighbour is a real rank
// (not MPI_PROC_NULL) must carry the neighbour's tag; ghosts at physical walls
// (MPI_PROC_NULL) are untouched by the exchange and stay at their initial
// value (0.0 in this fixture). Skipped at np == 1 (see SingleRankIsNoOp).
TEST_F(ExchangeQTensorNvshmem, GhostsHoldNeighbourRankOnAllFields) {
    if (mpi->world_size == 1) GTEST_SKIP() << "np == 1 exchange is a no-op";
    const double my_tag = static_cast<double>(mpi->world_rank) + 1.0;  // avoid 0

    FillFieldOwned(d_fields->d_qxx, my_tag);
    FillFieldOwned(d_fields->d_qxy, my_tag);
    FillFieldOwned(d_fields->d_qxz, my_tag);
    FillFieldOwned(d_fields->d_qyy, my_tag);
    FillFieldOwned(d_fields->d_qyz, my_tag);
    FillFieldOwned(d_fields->d_ux.data().get(), my_tag);
    FillFieldOwned(d_fields->d_uy.data().get(), my_tag);
    FillFieldOwned(d_fields->d_uz.data().get(), my_tag);

    halo->ExchangeQTensor(*d_fields);
    cudaDeviceSynchronize();

    // Expected ghost tag on each face: the neighbour's rank + 1, or 0 at a
    // physical wall (MPI_PROC_NULL — no put lands there and pre-init is zero).
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

    // Check every halo-exchanged field on every face.
    double* fields[8] = {
        d_fields->d_qxx, d_fields->d_qxy, d_fields->d_qxz,
        d_fields->d_qyy, d_fields->d_qyz,
        d_fields->d_ux.data().get(),
        d_fields->d_uy.data().get(),
        d_fields->d_uz.data().get()
    };

    for (int f = 0; f < 8; ++f) {
        auto host = CopyFieldToHost(fields[f]);

        // lo-x / hi-x ghost planes
        for (int z = 0; z < grid.local_nz; ++z) {
            for (int y = 0; y < grid.local_ny; ++y) {
                EXPECT_DOUBLE_EQ(host[grid.halo_idx(-1, y, z)],
                                 expected_tag_lo(0)) << "field=" << f
                                                     << " ghost=(-1," << y << "," << z << ")";
                EXPECT_DOUBLE_EQ(host[grid.halo_idx(grid.local_nx, y, z)],
                                 expected_tag_hi(0)) << "field=" << f;
            }
        }
        // lo-y / hi-y ghost planes
        for (int z = 0; z < grid.local_nz; ++z) {
            for (int x = 0; x < grid.local_nx; ++x) {
                EXPECT_DOUBLE_EQ(host[grid.halo_idx(x, -1, z)],
                                 expected_tag_lo(1)) << "field=" << f;
                EXPECT_DOUBLE_EQ(host[grid.halo_idx(x, grid.local_ny, z)],
                                 expected_tag_hi(1)) << "field=" << f;
            }
        }
        // lo-z / hi-z ghost planes
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

// Owned cells must be untouched by the exchange — a stray pack that reads
// ghost or an unpack that writes owned would flip these.
TEST_F(ExchangeQTensorNvshmem, OwnedCellsPreserved) {
    const double my_tag = static_cast<double>(mpi->world_rank) + 1.0;
    FillFieldOwned(d_fields->d_qxx, my_tag);

    halo->ExchangeQTensor(*d_fields);
    cudaDeviceSynchronize();

    auto host = CopyFieldToHost(d_fields->d_qxx);
    for (int z = 0; z < grid.local_nz; ++z)
        for (int y = 0; y < grid.local_ny; ++y)
            for (int x = 0; x < grid.local_nx; ++x)
                EXPECT_DOUBLE_EQ(host[grid.halo_idx(x, y, z)], my_tag);
}

}  // namespace
