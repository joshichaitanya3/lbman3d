// PR VII-g — Local pack unit test for HaloExchangeLbmNvshmem (face-only).
//
// Rationale (per src/cuda/CLAUDE.md "Local dev loop"): pack kernels write
// into send buffers with a specific slot layout, and a wrong stride or
// wrong source coordinate would be silent at nranks = 1 (NVSHMEM put
// targets self and would hide layout errors). Here we verify the pack:
//
// 1. Fill the +X (or -X, or +Y, ...) GHOST layer of d_f with a
//    position-encoded value at each of the 5 crossing dirs for that face.
// 2. Call ExchangeLBM's pack step (via a friend accessor).
// 3. Copy send_buf back to host.
// 4. Assert every slot matches the ghost value it was supposed to read.
//
// Rank-encoded assertions live in test_exchange_lbm_nvshmem.cu — this
// file is pure pack-layout coverage.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <vector>

#include "cuda/halo_exchange_lbm_nvshmem.h"
#include "cuda/halo_pack_kernels.h"     // for LbmFaceCrossings + Launch*
#include "device_fields.h"
#include "lattice_stencil.h"
#include "local_grid.h"
#include "mpi/mpi_context.h"
#include "params.h"

namespace {

// Position + direction encoding: reading back the buffer's contents lets us
// identify which (transverse_a, transverse_b, dir) each slot came from.
inline double PositionValue(int a, int b, int dir) {
    return (a + 1) + (b + 1) * 100.0 + (dir + 1) * 10000.0;
}

// Fill BOTH -X and +X ghost layers of d_f in a single cudaMemcpy so the
// second fill does not clobber the first. Values at (y, z, dir) get
// PositionValue(y, z, dir); the two dir sets never collide (missingXLo
// and missingXHi are disjoint).
//
// Layout note: d_f on device is direction-SLOWEST (all cells for one dir
// are contiguous), but `halo_idx(x,y,z,i)` on host returns direction-fastest
// (per the `#ifdef __CUDA_ARCH__` branch). We therefore write the host
// buffer in the device convention explicitly: base = i * halo_volume +
// halo_idx(x,y,z), so a raw cudaMemcpy lands each dir slot where the device
// kernel expects it. Same idiom as DeviceFields::Initialize's transpose.
void FillGhostXFaces(double* d_f, const LocalGrid& g,
                     const int (&lo_dirs)[5], const int (&hi_dirs)[5]) {
    const std::size_t hv = static_cast<size_t>(g.HaloVolume());
    std::vector<double> host(hv * Lattice::ndir, 0.0);
    for (int z = 0; z < g.local_nz; ++z)
        for (int y = 0; y < g.local_ny; ++y)
            for (int k = 0; k < 5; ++k) {
                host[lo_dirs[k] * hv + g.halo_idx(-1,           y, z)] = PositionValue(y, z, lo_dirs[k]);
                host[hi_dirs[k] * hv + g.halo_idx(g.local_nx,   y, z)] = PositionValue(y, z, hi_dirs[k]);
            }
    cudaMemcpy(d_f, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice);
}

// Copy a raw send buffer back to host for inspection.
std::vector<double> CopyBufToHost(const double* d_buf, std::size_t n) {
    std::vector<double> host(n, 0.0);
    cudaMemcpy(host.data(), d_buf, n * sizeof(double), cudaMemcpyDeviceToHost);
    return host;
}

class PackLbmNvshmem : public ::testing::Test {
protected:
    static inline MPIContext* mpi = nullptr;
    static inline LocalGrid grid;
    static inline BackendInfo backend;
    static inline HaloExchangeLbmNvshmem* halo = nullptr;
    static inline double* d_f = nullptr;
    static inline std::size_t d_f_nelems = 0;

    static void SetUpTestSuite() {
        mpi = new MPIContext(/*periods=*/{1, 1, 1});
        grid = mpi->MakeLocalGrid();
        backend = InitializeComputeBackend(*mpi, grid);
        halo = new HaloExchangeLbmNvshmem(grid, *mpi, /*is_wall=*/{});
        d_f_nelems = static_cast<size_t>(grid.HaloVolume()) * Lattice::ndir;
        cudaMalloc(&d_f, d_f_nelems * sizeof(double));
        cudaMemset(d_f, 0, d_f_nelems * sizeof(double));
    }

    static void TearDownTestSuite() {
        if (d_f) { cudaFree(d_f); d_f = nullptr; }
        delete halo; halo = nullptr;
        delete mpi;  mpi  = nullptr;
    }

    // Small helper to build a LbmFaceCrossings on the host (mirror of the
    // one in nvshmem_halo.cu — kept private there, replicated here so the
    // test does not need friend access).
    static LbmFaceCrossings MakeCrossings(const int (&dirs)[5],
                                          const int (&e_a)[Lattice::ndir],
                                          const int (&e_b)[Lattice::ndir]) {
        LbmFaceCrossings c{};
        for (int k = 0; k < 5; ++k) {
            c.dir[k]         = dirs[k];
            c.e_trans_a[k]   = e_a[dirs[k]];
            c.e_trans_b[k]   = e_b[dirs[k]];
        }
        return c;
    }
};

// -------- X-axis pack: verify -X and +X ghost slots land in send_lo/send_hi
// with the correct per-dir stride and transverse indexing.
TEST_F(PackLbmNvshmem, PackXAxisLoAndHi) {
    // Fill -X ghost at (-1, y, z) for dirs missingXHi (ex<0), and +X ghost
    // at (local_nx, y, z) for dirs missingXLo (ex>0). Both in one memcpy.
    cudaMemset(d_f, 0, d_f_nelems * sizeof(double));
    FillGhostXFaces(d_f, grid, Lattice::missingXHi, Lattice::missingXLo);

    // Pack directly through the shared launcher (avoids the class's split_[]
    // gate — this test always exercises the pack even at nranks=1).
    const LbmFaceCrossings lo_x = MakeCrossings(Lattice::missingXHi,
                                                Lattice::ey, Lattice::ez);
    const LbmFaceCrossings hi_x = MakeCrossings(Lattice::missingXLo,
                                                Lattice::ey, Lattice::ez);
    LaunchPackLbmAxisX(d_f, halo->send_buf_[0], halo->send_buf_[1],
                       lo_x, hi_x, grid);
    cudaDeviceSynchronize();

    const std::size_t n = HaloExchangeLbmNvshmem::kCrossingDirs * halo->face_area(0);
    auto lo = CopyBufToHost(halo->send_buf_[0], n);
    auto hi = CopyBufToHost(halo->send_buf_[1], n);

    const int face_area = grid.local_ny * grid.local_nz;
    for (int k = 0; k < 5; ++k) {
        for (int z = 0; z < grid.local_nz; ++z) {
            for (int y = 0; y < grid.local_ny; ++y) {
                const int slot = k * face_area + z * grid.local_ny + y;
                EXPECT_DOUBLE_EQ(lo[slot], PositionValue(y, z, Lattice::missingXHi[k]))
                    << "lo-X pack mismatch at k=" << k << ", y=" << y << ", z=" << z;
                EXPECT_DOUBLE_EQ(hi[slot], PositionValue(y, z, Lattice::missingXLo[k]))
                    << "hi-X pack mismatch at k=" << k << ", y=" << y << ", z=" << z;
            }
        }
    }
}

// -------- Y-axis pack.
TEST_F(PackLbmNvshmem, PackYAxisLoAndHi) {
    cudaMemset(d_f, 0, d_f_nelems * sizeof(double));
    // Fill -Y ghost with dirs missingYHi, +Y ghost with dirs missingYLo.
    // Device dir-slowest layout — see FillGhostXFace's note.
    {
        const std::size_t hv = static_cast<size_t>(grid.HaloVolume());
        std::vector<double> host(d_f_nelems, 0.0);
        for (int z = 0; z < grid.local_nz; ++z)
            for (int x = 0; x < grid.local_nx; ++x)
                for (int k = 0; k < 5; ++k) {
                    host[Lattice::missingYHi[k] * hv + grid.halo_idx(x, -1,           z)] = PositionValue(x, z, Lattice::missingYHi[k]);
                    host[Lattice::missingYLo[k] * hv + grid.halo_idx(x, grid.local_ny, z)] = PositionValue(x, z, Lattice::missingYLo[k]);
                }
        cudaMemcpy(d_f, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice);
    }

    const LbmFaceCrossings lo_y = MakeCrossings(Lattice::missingYHi,
                                                Lattice::ex, Lattice::ez);
    const LbmFaceCrossings hi_y = MakeCrossings(Lattice::missingYLo,
                                                Lattice::ex, Lattice::ez);
    LaunchPackLbmAxisY(d_f, halo->send_buf_[2], halo->send_buf_[3],
                       lo_y, hi_y, grid);
    cudaDeviceSynchronize();

    const std::size_t n = HaloExchangeLbmNvshmem::kCrossingDirs * halo->face_area(1);
    auto lo = CopyBufToHost(halo->send_buf_[2], n);
    auto hi = CopyBufToHost(halo->send_buf_[3], n);

    const int face_area = grid.local_nx * grid.local_nz;
    for (int k = 0; k < 5; ++k) {
        for (int z = 0; z < grid.local_nz; ++z) {
            for (int x = 0; x < grid.local_nx; ++x) {
                const int slot = k * face_area + z * grid.local_nx + x;
                EXPECT_DOUBLE_EQ(lo[slot], PositionValue(x, z, Lattice::missingYHi[k]));
                EXPECT_DOUBLE_EQ(hi[slot], PositionValue(x, z, Lattice::missingYLo[k]));
            }
        }
    }
}

// -------- Z-axis pack.
TEST_F(PackLbmNvshmem, PackZAxisLoAndHi) {
    cudaMemset(d_f, 0, d_f_nelems * sizeof(double));
    // Device dir-slowest layout — see FillGhostXFace's note.
    {
        const std::size_t hv = static_cast<size_t>(grid.HaloVolume());
        std::vector<double> host(d_f_nelems, 0.0);
        for (int y = 0; y < grid.local_ny; ++y)
            for (int x = 0; x < grid.local_nx; ++x)
                for (int k = 0; k < 5; ++k) {
                    host[Lattice::missingZHi[k] * hv + grid.halo_idx(x, y, -1)]           = PositionValue(x, y, Lattice::missingZHi[k]);
                    host[Lattice::missingZLo[k] * hv + grid.halo_idx(x, y, grid.local_nz)] = PositionValue(x, y, Lattice::missingZLo[k]);
                }
        cudaMemcpy(d_f, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice);
    }

    const LbmFaceCrossings lo_z = MakeCrossings(Lattice::missingZHi,
                                                Lattice::ex, Lattice::ey);
    const LbmFaceCrossings hi_z = MakeCrossings(Lattice::missingZLo,
                                                Lattice::ex, Lattice::ey);
    LaunchPackLbmAxisZ(d_f, halo->send_buf_[4], halo->send_buf_[5],
                       lo_z, hi_z, grid);
    cudaDeviceSynchronize();

    const std::size_t n = HaloExchangeLbmNvshmem::kCrossingDirs * halo->face_area(2);
    auto lo = CopyBufToHost(halo->send_buf_[4], n);
    auto hi = CopyBufToHost(halo->send_buf_[5], n);

    const int face_area = grid.local_nx * grid.local_ny;
    for (int k = 0; k < 5; ++k) {
        for (int y = 0; y < grid.local_ny; ++y) {
            for (int x = 0; x < grid.local_nx; ++x) {
                const int slot = k * face_area + y * grid.local_nx + x;
                EXPECT_DOUBLE_EQ(lo[slot], PositionValue(x, y, Lattice::missingZHi[k]));
                EXPECT_DOUBLE_EQ(hi[slot], PositionValue(x, y, Lattice::missingZLo[k]));
            }
        }
    }
}

}  // namespace
