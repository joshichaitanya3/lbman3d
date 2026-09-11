// PR VII-f — Local pack unit test for HaloExchangePassiveStressesNvshmem.
//
// Same rationale as test_pack_qtensor_nvshmem.cu (see src/cuda/CLAUDE.md
// "Local dev loop"): verify the pack kernels write into the send buffers with
// the expected slot layout, so a wrong per-field stride surfaces at
// nranks = 1 (where NVSHMEM put targets self and would hide layout errors
// trivially) rather than only on the cluster. The distinct assertion here vs.
// the QTensor pack test is the per-field stride at kMaxFields = 13 (vs. 8) —
// a shared-per-face-stride bug would slip past the QTensor test.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <vector>

#include "cuda/halo_exchange_passive_stresses_nvshmem.h"
#include "device_fields.h"      // InitializeComputeBackend
#include "local_grid.h"
#include "mpi/mpi_context.h"
#include "params.h"

namespace {

// Position encoding matches test_pack_qtensor_nvshmem.cu so pack-buffer
// contents are directly comparable between the two exchanges' unit tests.
inline double PositionValue(int x, int y, int z) {
    return (x + 1) + (y + 1) * 100.0 + (z + 1) * 10000.0;
}

void FillFieldPositionEncoded(double* d_field, const LocalGrid& g) {
    std::vector<double> host(g.HaloVolume(), 0.0);
    for (int z = 0; z < g.local_nz; ++z)
        for (int y = 0; y < g.local_ny; ++y)
            for (int x = 0; x < g.local_nx; ++x)
                host[g.halo_idx(x, y, z)] = PositionValue(x, y, z);
    cudaMemcpy(d_field, host.data(), host.size() * sizeof(double),
               cudaMemcpyHostToDevice);
}

class PackPassiveStressesNvshmem : public ::testing::Test {
protected:
    static inline MPIContext* mpi = nullptr;
    static inline LocalGrid grid;
    static inline BackendInfo backend;
    static inline HaloExchangePassiveStressesNvshmem* halo = nullptr;
    static inline double* d_field = nullptr;

    static void SetUpTestSuite() {
        mpi = new MPIContext(/*periods=*/{1, 1, 1});
        grid = mpi->MakeLocalGrid();
        backend = InitializeComputeBackend(*mpi, grid);
        halo = new HaloExchangePassiveStressesNvshmem(grid, *mpi);
        cudaMalloc(&d_field, grid.HaloVolume() * sizeof(double));
        cudaMemset(d_field, 0, grid.HaloVolume() * sizeof(double));
        FillFieldPositionEncoded(d_field, grid);
    }

    static void TearDownTestSuite() {
        if (d_field) { cudaFree(d_field); d_field = nullptr; }
        delete halo; halo = nullptr;
        delete mpi;  mpi  = nullptr;
    }

    static void CopySendBuffersToHost(int axis,
                                      std::vector<double>& out_lo,
                                      std::vector<double>& out_hi) {
        const std::size_t face = halo->face_area(axis);
        const std::size_t n    = HaloExchangePassiveStressesNvshmem::kMaxFields * face;
        out_lo.assign(n, 0.0);
        out_hi.assign(n, 0.0);
        cudaMemcpy(out_lo.data(), halo->send_buf_[2 * axis    ],
                   n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_hi.data(), halo->send_buf_[2 * axis + 1],
                   n * sizeof(double), cudaMemcpyDeviceToHost);
    }
};

// -------- X-axis (lo/hi YZ faces) --------
TEST_F(PackPassiveStressesNvshmem, PackXAxisLoFace) {
    halo->PackSingleFieldForTest(d_field, /*field_idx=*/0, /*axis=*/0);
    cudaDeviceSynchronize();

    std::vector<double> lo, hi;
    CopySendBuffersToHost(0, lo, hi);

    for (int z = 0; z < grid.local_nz; ++z) {
        for (int y = 0; y < grid.local_ny; ++y) {
            EXPECT_DOUBLE_EQ(lo[z * grid.local_ny + y],
                             PositionValue(0, y, z));
        }
    }
}

TEST_F(PackPassiveStressesNvshmem, PackXAxisHiFace) {
    halo->PackSingleFieldForTest(d_field, /*field_idx=*/0, /*axis=*/0);
    cudaDeviceSynchronize();

    std::vector<double> lo, hi;
    CopySendBuffersToHost(0, lo, hi);

    for (int z = 0; z < grid.local_nz; ++z) {
        for (int y = 0; y < grid.local_ny; ++y) {
            EXPECT_DOUBLE_EQ(hi[z * grid.local_ny + y],
                             PositionValue(grid.local_nx - 1, y, z));
        }
    }
}

// -------- Y-axis (lo/hi XZ faces) --------
TEST_F(PackPassiveStressesNvshmem, PackYAxisLoFace) {
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

TEST_F(PackPassiveStressesNvshmem, PackYAxisHiFace) {
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
TEST_F(PackPassiveStressesNvshmem, PackZAxisLoFace) {
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

TEST_F(PackPassiveStressesNvshmem, PackZAxisHiFace) {
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

// Highest-slot stride check. The QTensor pack test only exercises slot 1
// (kMaxFields = 8); this class widens to 13, so a broken per-field stride
// that only bites at large field indices would slip past the QTensor test.
// Pack the same field into every slot from 0 to kMaxFields-1 and assert the
// last slot matches slot 0.
TEST_F(PackPassiveStressesNvshmem, PackXAxisFullFieldRange) {
    const std::size_t last = HaloExchangePassiveStressesNvshmem::kMaxFields - 1;
    halo->PackSingleFieldForTest(d_field, /*field_idx=*/last, /*axis=*/0);
    cudaDeviceSynchronize();

    std::vector<double> lo, hi;
    CopySendBuffersToHost(0, lo, hi);

    const std::size_t stride = halo->face_area(0);
    for (int z = 0; z < grid.local_nz; ++z) {
        for (int y = 0; y < grid.local_ny; ++y) {
            const int packed = z * grid.local_ny + y;
            EXPECT_DOUBLE_EQ(lo[0    * stride + packed], PositionValue(0, y, z));
            EXPECT_DOUBLE_EQ(lo[last * stride + packed], PositionValue(0, y, z));
        }
    }
}

}  // namespace
