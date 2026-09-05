// PR VII-e — Local pack unit test for HaloExchangeQTensorNvshmem.
//
// Purpose (per src/cuda/CLAUDE.md "Local dev loop"): verify that the pack
// kernels write into the send buffers with the expected slot layout, so that
// pack bugs surface at nranks = 1 (where NVSHMEM put targets self and would
// hide layout errors trivially) rather than only on the cluster.
//
// Approach: fill a device field with a position-encoding value, ask the halo to
// pack it into a single field slot on one axis, copy the send buffer back to
// the host, and assert the slot layout matches the CPU pack (same as
// tests/mpi/test_halo_exchange_qtensor.cc for the CPU class).

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <vector>

#include "cuda/halo_exchange_qtensor_nvshmem.h"
#include "device_fields.h"      // InitializeComputeBackend
#include "local_grid.h"
#include "mpi/mpi_context.h"
#include "params.h"

namespace {

// Position encoding matches tests/mpi/test_halo_exchange_qtensor.cc so a
// side-by-side pack-buffer diff against the CPU class is trivially readable.
inline double PositionValue(int x, int y, int z) {
    return (x + 1) + (y + 1) * 100.0 + (z + 1) * 10000.0;
}

// Fill an owned-only field on the device with PositionValue(x, y, z). Ghost
// cells stay at cudaMemset's zero — pack only reads owned boundaries.
void FillFieldPositionEncoded(double* d_field, const LocalGrid& g) {
    std::vector<double> host(g.HaloVolume(), 0.0);
    for (int z = 0; z < g.local_nz; ++z)
        for (int y = 0; y < g.local_ny; ++y)
            for (int x = 0; x < g.local_nx; ++x)
                host[g.halo_idx(x, y, z)] = PositionValue(x, y, z);
    cudaMemcpy(d_field, host.data(), host.size() * sizeof(double),
               cudaMemcpyHostToDevice);
}

// One-time NVSHMEM bootstrap shared by all tests in this TU. Fixture-lazy
// (SetUpTestSuite, not static inline) so it runs after main()'s MPI_Init.
class PackQTensorNvshmem : public ::testing::Test {
protected:
    static inline MPIContext* mpi = nullptr;
    static inline LocalGrid grid;
    static inline BackendInfo backend;
    static inline HaloExchangeQTensorNvshmem* halo = nullptr;
    static inline double* d_field = nullptr;

    static void SetUpTestSuite() {
        mpi = new MPIContext(/*periods=*/{1, 1, 1});
        grid = mpi->MakeLocalGrid();
        backend = InitializeComputeBackend(*mpi, grid);
        halo = new HaloExchangeQTensorNvshmem(grid, *mpi);
        // Owned-plus-halo scratch — same shape as any real Q field.
        cudaMalloc(&d_field, grid.HaloVolume() * sizeof(double));
        cudaMemset(d_field, 0, grid.HaloVolume() * sizeof(double));
        FillFieldPositionEncoded(d_field, grid);
    }

    static void TearDownTestSuite() {
        if (d_field) { cudaFree(d_field); d_field = nullptr; }
        delete halo; halo = nullptr;
        delete mpi;  mpi  = nullptr;
    }

    // Copy the axis's lo/hi send buffers back to host for inspection.
    static void CopySendBuffersToHost(int axis,
                                      std::vector<double>& out_lo,
                                      std::vector<double>& out_hi) {
        const std::size_t face = halo->face_area(axis);
        const std::size_t n    = HaloExchangeQTensorNvshmem::kMaxFields * face;
        out_lo.assign(n, 0.0);
        out_hi.assign(n, 0.0);
        cudaMemcpy(out_lo.data(), halo->send_buf_[2 * axis    ],
                   n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_hi.data(), halo->send_buf_[2 * axis + 1],
                   n * sizeof(double), cudaMemcpyDeviceToHost);
    }
};

// -------- X-axis (lo/hi YZ faces) --------
TEST_F(PackQTensorNvshmem, PackXAxisLoFace) {
    halo->PackSingleFieldForTest(d_field, /*field_idx=*/0, /*axis=*/0);
    cudaDeviceSynchronize();

    std::vector<double> lo, hi;
    CopySendBuffersToHost(0, lo, hi);

    // send_buf_[0] (lo-x): should hold owned-boundary values at x = 0.
    for (int z = 0; z < grid.local_nz; ++z) {
        for (int y = 0; y < grid.local_ny; ++y) {
            EXPECT_DOUBLE_EQ(lo[z * grid.local_ny + y],
                             PositionValue(0, y, z));
        }
    }
}

TEST_F(PackQTensorNvshmem, PackXAxisHiFace) {
    halo->PackSingleFieldForTest(d_field, /*field_idx=*/0, /*axis=*/0);
    cudaDeviceSynchronize();

    std::vector<double> lo, hi;
    CopySendBuffersToHost(0, lo, hi);

    // send_buf_[1] (hi-x): should hold owned-boundary values at x = local_nx - 1.
    for (int z = 0; z < grid.local_nz; ++z) {
        for (int y = 0; y < grid.local_ny; ++y) {
            EXPECT_DOUBLE_EQ(hi[z * grid.local_ny + y],
                             PositionValue(grid.local_nx - 1, y, z));
        }
    }
}

// -------- Y-axis (lo/hi XZ faces) --------
TEST_F(PackQTensorNvshmem, PackYAxisLoFace) {
    halo->PackSingleFieldForTest(d_field, /*field_idx=*/0, /*axis=*/1);
    cudaDeviceSynchronize();

    std::vector<double> lo, hi;
    CopySendBuffersToHost(1, lo, hi);

    for (int z = 0; z < grid.local_nz; ++z) {
        for (int x = 0; x < grid.local_nx; ++x) {
            EXPECT_DOUBLE_EQ(lo[z * grid.local_nx + x],
                             PositionValue(x, 0, z));
        }
    }
}

TEST_F(PackQTensorNvshmem, PackYAxisHiFace) {
    halo->PackSingleFieldForTest(d_field, /*field_idx=*/0, /*axis=*/1);
    cudaDeviceSynchronize();

    std::vector<double> lo, hi;
    CopySendBuffersToHost(1, lo, hi);

    for (int z = 0; z < grid.local_nz; ++z) {
        for (int x = 0; x < grid.local_nx; ++x) {
            EXPECT_DOUBLE_EQ(hi[z * grid.local_nx + x],
                             PositionValue(x, grid.local_ny - 1, z));
        }
    }
}

// -------- Z-axis (lo/hi XY faces) --------
TEST_F(PackQTensorNvshmem, PackZAxisLoFace) {
    halo->PackSingleFieldForTest(d_field, /*field_idx=*/0, /*axis=*/2);
    cudaDeviceSynchronize();

    std::vector<double> lo, hi;
    CopySendBuffersToHost(2, lo, hi);

    for (int y = 0; y < grid.local_ny; ++y) {
        for (int x = 0; x < grid.local_nx; ++x) {
            EXPECT_DOUBLE_EQ(lo[y * grid.local_nx + x],
                             PositionValue(x, y, 0));
        }
    }
}

TEST_F(PackQTensorNvshmem, PackZAxisHiFace) {
    halo->PackSingleFieldForTest(d_field, /*field_idx=*/0, /*axis=*/2);
    cudaDeviceSynchronize();

    std::vector<double> lo, hi;
    CopySendBuffersToHost(2, lo, hi);

    for (int y = 0; y < grid.local_ny; ++y) {
        for (int x = 0; x < grid.local_nx; ++x) {
            EXPECT_DOUBLE_EQ(hi[y * grid.local_nx + x],
                             PositionValue(x, y, grid.local_nz - 1));
        }
    }
}

// Multi-field pack: kMaxFields slots must not alias — a wrong per-field stride
// would silently overwrite slot-0's data and both tests above would still pass.
// This test only makes teeth if we pack more than one field.
TEST_F(PackQTensorNvshmem, PackXAxisTwoFieldsIndependent) {
    // Pack the same field twice into slots 0 and 1 via nfields = 2.
    // If per-field stride matches face_area(0) exactly, slot 1's contents are
    // identical to slot 0's; a wrong stride would corrupt slot 0.
    halo->PackSingleFieldForTest(d_field, /*field_idx=*/1, /*axis=*/0);
    cudaDeviceSynchronize();

    std::vector<double> lo, hi;
    CopySendBuffersToHost(0, lo, hi);

    const std::size_t stride = halo->face_area(0);
    for (int z = 0; z < grid.local_nz; ++z) {
        for (int y = 0; y < grid.local_ny; ++y) {
            const int packed = z * grid.local_ny + y;
            EXPECT_DOUBLE_EQ(lo[0 * stride + packed], PositionValue(0, y, z));
            EXPECT_DOUBLE_EQ(lo[1 * stride + packed], PositionValue(0, y, z));
        }
    }
}

}  // namespace
