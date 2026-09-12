// PR VII-g — 2-rank halo round-trip test for HaloExchangeLbmNvshmem
// (face-only stage).
//
// Fills each rank's ghost layer with rank-encoded values at the 5 crossing
// dirs per face, runs ExchangeLBM, and asserts every rank's owned boundary
// now holds the OPPOSITE face's neighbour's tag (because push streaming
// deposits into the ±X ghost, which then travels to the ±X neighbour's
// mirror owned boundary).
//
// Uses FullyPeriodicConfig (no walls) so the skip-unpack predicate never
// fires — that path is exercised separately when a wall test lands.
// Under FullyPeriodicConfig on a 2-PE {2,1,1} decomposition, only the X
// axis is split, so the ±X face puts are the only ones that fire; ±Y and
// ±Z stay in their initial (zero) recv buffers, and their ghosts and
// owned cells are untouched.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <vector>

#include "cuda/halo_exchange_lbm_nvshmem.h"
#include "device_fields.h"
#include "lattice_stencil.h"
#include "local_grid.h"
#include "mpi/mpi_context.h"
#include "params.h"

namespace {

class ExchangeLbmNvshmem : public ::testing::Test {
protected:
    static inline MPIContext* mpi = nullptr;
    static inline LocalGrid grid;
    static inline BackendInfo backend;
    static inline HaloExchangeLbmNvshmem* halo = nullptr;
    static inline DeviceFields* d_fields = nullptr;

    static void SetUpTestSuite() {
        mpi = new MPIContext(/*periods=*/{1, 1, 1});
        grid = mpi->MakeLocalGrid();
        backend = InitializeComputeBackend(*mpi, grid);
        d_fields = new DeviceFields(grid, backend.symmetric_halo_volume);
        halo = new HaloExchangeLbmNvshmem(grid, *mpi, /*is_wall=*/{});
    }

    static void TearDownTestSuite() {
        delete halo;     halo     = nullptr;
        delete d_fields; d_fields = nullptr;
        delete mpi;      mpi      = nullptr;
    }

    // Fill THIS rank's ghost layer of d_f with `tag` at 5 crossing dirs per
    // face. All other slots stay at zero.
    //
    // Layout note: d_f on device is direction-SLOWEST — index (x,y,z,dir)
    // sits at byte offset `dir * halo_volume + halo_idx(x,y,z)`. Host-side
    // `halo_idx(x,y,z,dir)` returns direction-fastest. To land bytes where
    // the device kernel expects them, index the host staging buffer in the
    // device convention explicitly (mirror of DeviceFields::Initialize).
    static void FillGhostsWithTag(double* d_f, double tag) {
        const std::size_t hv = static_cast<size_t>(grid.HaloVolume());
        const std::size_t n  = hv * Lattice::ndir;
        std::vector<double> host(n, 0.0);

        auto put = [&](int x, int y, int z, int dir) {
            host[dir * hv + grid.halo_idx(x, y, z)] = tag;
        };
        auto fill_x_face = [&](int ghost_x, const int (&dirs)[5]) {
            for (int z = 0; z < grid.local_nz; ++z)
                for (int y = 0; y < grid.local_ny; ++y)
                    for (int k = 0; k < 5; ++k)
                        put(ghost_x, y, z, dirs[k]);
        };
        auto fill_y_face = [&](int ghost_y, const int (&dirs)[5]) {
            for (int z = 0; z < grid.local_nz; ++z)
                for (int x = 0; x < grid.local_nx; ++x)
                    for (int k = 0; k < 5; ++k)
                        put(x, ghost_y, z, dirs[k]);
        };
        auto fill_z_face = [&](int ghost_z, const int (&dirs)[5]) {
            for (int y = 0; y < grid.local_ny; ++y)
                for (int x = 0; x < grid.local_nx; ++x)
                    for (int k = 0; k < 5; ++k)
                        put(x, y, ghost_z, dirs[k]);
        };

        // -X ghost: my streaming pushed dirs with ex<0 (missingXHi).
        // +X ghost: dirs with ex>0 (missingXLo). Analogous on Y and Z.
        fill_x_face(-1,             Lattice::missingXHi);
        fill_x_face(grid.local_nx,  Lattice::missingXLo);
        fill_y_face(-1,             Lattice::missingYHi);
        fill_y_face(grid.local_ny,  Lattice::missingYLo);
        fill_z_face(-1,             Lattice::missingZHi);
        fill_z_face(grid.local_nz,  Lattice::missingZLo);

        cudaMemcpy(d_f, host.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    }

    static std::vector<double> CopyToHost(const double* d_f) {
        const std::size_t n = static_cast<size_t>(grid.HaloVolume()) * Lattice::ndir;
        std::vector<double> host(n, 0.0);
        cudaMemcpy(host.data(), d_f, n * sizeof(double), cudaMemcpyDeviceToHost);
        return host;
    }

    // Device-layout indexing helper for reading the copied-back host buffer.
    // The buffer holds raw device bytes (dir-slowest), so we must NOT use
    // grid.halo_idx(x, y, z, dir) which gives dir-fastest on the host.
    static double At(const std::vector<double>& host, int x, int y, int z, int dir) {
        const std::size_t hv = static_cast<size_t>(grid.HaloVolume());
        return host[dir * hv + grid.halo_idx(x, y, z)];
    }
};

// Single-rank fast path: ExchangeLBM early-returns on world_size == 1,
// so the ghosts stay at their initial value and no owned cell is touched.
TEST_F(ExchangeLbmNvshmem, SingleRankIsNoOp) {
    if (mpi->world_size != 1) GTEST_SKIP() << "np != 1";

    cudaMemset(d_fields->d_f, 0,
               static_cast<size_t>(grid.HaloVolume()) * Lattice::ndir * sizeof(double));
    FillGhostsWithTag(d_fields->d_f, 42.0);
    halo->ExchangeLBM(*d_fields);
    cudaDeviceSynchronize();

    auto host = CopyToHost(d_fields->d_f);
    // Owned cells should still be zero.
    for (int z = 0; z < grid.local_nz; ++z)
        for (int y = 0; y < grid.local_ny; ++y)
            for (int x = 0; x < grid.local_nx; ++x)
                for (int i = 0; i < Lattice::ndir; ++i)
                    EXPECT_DOUBLE_EQ(At(host, x, y, z, i), 0.0)
                        << "owned cell touched at (" << x << "," << y << "," << z << "," << i << ")";
}

// Round-trip: after ExchangeLBM, every rank's owned boundary should hold
// the neighbour's tag at the corresponding crossing dirs.
//
// Concretely, my ±X owned boundary receives from my ±X neighbour, at dirs
// that came from the neighbour's ±X ghost (which the neighbour packed with
// missingX{Lo,Hi}). See the class header for the exact routing.
TEST_F(ExchangeLbmNvshmem, GhostsRoundTripToNeighbourOwned) {
    if (mpi->world_size == 1) GTEST_SKIP() << "np == 1 exchange is a no-op";

    const double my_tag = static_cast<double>(mpi->world_rank) + 1.0;  // avoid 0
    cudaMemset(d_fields->d_f, 0,
               static_cast<size_t>(grid.HaloVolume()) * Lattice::ndir * sizeof(double));
    FillGhostsWithTag(d_fields->d_f, my_tag);
    halo->ExchangeLBM(*d_fields);
    cudaDeviceSynchronize();

    auto host = CopyToHost(d_fields->d_f);

    // Per class comment: send_buf_[+X] → recv_buf_[-X] on +X neighbour.
    // So MY -X owned boundary (x=0) at dirs missingXLo (ex>0) receives
    // from my -X neighbour's +X ghost pack. Expected tag: neighbour_[0]
    // rank + 1, or 0.0 if neighbour is MPI_PROC_NULL.
    auto expected_lo_tag = [&](int axis) -> double {
        return halo->neighbor_[2*axis] == MPI_PROC_NULL
            ? 0.0
            : static_cast<double>(halo->neighbor_[2*axis]) + 1.0;
    };
    auto expected_hi_tag = [&](int axis) -> double {
        return halo->neighbor_[2*axis + 1] == MPI_PROC_NULL
            ? 0.0
            : static_cast<double>(halo->neighbor_[2*axis + 1]) + 1.0;
    };

    // Check -X owned (x=0) at dirs missingXLo.
    if (halo->split_[0]) {
        for (int k = 0; k < 5; ++k) {
            const int dir = Lattice::missingXLo[k];
            for (int z = 0; z < grid.local_nz; ++z) {
                for (int y = 0; y < grid.local_ny; ++y) {
                    EXPECT_DOUBLE_EQ(At(host, 0, y, z, dir),
                                     expected_lo_tag(0))
                        << "-X owned: k=" << k << " dir=" << dir
                        << " (y,z)=(" << y << "," << z << ")";
                }
            }
        }
        // +X owned (x=local_nx-1) at dirs missingXHi.
        for (int k = 0; k < 5; ++k) {
            const int dir = Lattice::missingXHi[k];
            for (int z = 0; z < grid.local_nz; ++z) {
                for (int y = 0; y < grid.local_ny; ++y) {
                    EXPECT_DOUBLE_EQ(At(host, grid.local_nx - 1, y, z, dir),
                                     expected_hi_tag(0))
                        << "+X owned: k=" << k << " dir=" << dir;
                }
            }
        }
    }

    // Y and Z checks: analogous, but only exercise them where the ExchangeLBM
    // actually ran (split_[axis]). Under {2,1,1}, only X is split, so these
    // loops are no-ops. When they DO run (e.g. under a {1,2,1} decomp), the
    // discriminating cells are those that are NOT already on a split X or Z
    // boundary — otherwise the "overlap" between an X-owned corner and a
    // Y-face crossing dir (e.g. dir 7 ∈ missingXLo ∩ missingYLo) makes the
    // expected value ambiguous.
    if (halo->split_[1]) {
        for (int k = 0; k < 5; ++k) {
            const int dir = Lattice::missingYLo[k];
            for (int z = 0; z < grid.local_nz; ++z)
                for (int x = 0; x < grid.local_nx; ++x) {
                    // Skip the X-boundary corner where X-unpack (if split_[0])
                    // may also have written this dir slot.
                    if (halo->split_[0] &&
                        ((x == 0 && Lattice::ex[dir] > 0) ||
                         (x == grid.local_nx - 1 && Lattice::ex[dir] < 0))) continue;
                    EXPECT_DOUBLE_EQ(At(host, x, 0, z, dir), expected_lo_tag(1));
                }
        }
        for (int k = 0; k < 5; ++k) {
            const int dir = Lattice::missingYHi[k];
            for (int z = 0; z < grid.local_nz; ++z)
                for (int x = 0; x < grid.local_nx; ++x) {
                    if (halo->split_[0] &&
                        ((x == 0 && Lattice::ex[dir] > 0) ||
                         (x == grid.local_nx - 1 && Lattice::ex[dir] < 0))) continue;
                    EXPECT_DOUBLE_EQ(At(host, x, grid.local_ny - 1, z, dir),
                                     expected_hi_tag(1));
                }
        }
    }
    if (halo->split_[2]) {
        for (int k = 0; k < 5; ++k) {
            const int dir = Lattice::missingZLo[k];
            for (int y = 0; y < grid.local_ny; ++y)
                for (int x = 0; x < grid.local_nx; ++x) {
                    if (halo->split_[0] &&
                        ((x == 0 && Lattice::ex[dir] > 0) ||
                         (x == grid.local_nx - 1 && Lattice::ex[dir] < 0))) continue;
                    if (halo->split_[1] &&
                        ((y == 0 && Lattice::ey[dir] > 0) ||
                         (y == grid.local_ny - 1 && Lattice::ey[dir] < 0))) continue;
                    EXPECT_DOUBLE_EQ(At(host, x, y, 0, dir), expected_lo_tag(2));
                }
        }
        for (int k = 0; k < 5; ++k) {
            const int dir = Lattice::missingZHi[k];
            for (int y = 0; y < grid.local_ny; ++y)
                for (int x = 0; x < grid.local_nx; ++x) {
                    if (halo->split_[0] &&
                        ((x == 0 && Lattice::ex[dir] > 0) ||
                         (x == grid.local_nx - 1 && Lattice::ex[dir] < 0))) continue;
                    if (halo->split_[1] &&
                        ((y == 0 && Lattice::ey[dir] > 0) ||
                         (y == grid.local_ny - 1 && Lattice::ey[dir] < 0))) continue;
                    EXPECT_DOUBLE_EQ(At(host, x, y, grid.local_nz - 1, dir),
                                     expected_hi_tag(2));
                }
        }
    }
}

// Non-boundary owned cells and non-crossing dirs at boundary cells must
// stay untouched — a wrong dir set (e.g. all 15 pops) would clobber these.
TEST_F(ExchangeLbmNvshmem, NonCrossingSlotsPreservedAtBoundary) {
    if (mpi->world_size == 1) GTEST_SKIP() << "np == 1 exchange is a no-op";

    // Pre-fill every owned cell at every dir with a canary + fill ±X ghosts
    // with the exchange tag in ONE cudaMemcpy so nothing gets clobbered.
    const std::size_t hv = static_cast<size_t>(grid.HaloVolume());
    const std::size_t n  = hv * Lattice::ndir;
    const double canary = 999.0;
    const double tag    = 42.0;
    std::vector<double> host(n, 0.0);
    auto put = [&](int x, int y, int z, int dir, double v) {
        host[dir * hv + grid.halo_idx(x, y, z)] = v;
    };
    for (int z = 0; z < grid.local_nz; ++z)
        for (int y = 0; y < grid.local_ny; ++y)
            for (int x = 0; x < grid.local_nx; ++x)
                for (int i = 0; i < Lattice::ndir; ++i)
                    put(x, y, z, i, canary);
    for (int z = 0; z < grid.local_nz; ++z)
        for (int y = 0; y < grid.local_ny; ++y)
            for (int k = 0; k < 5; ++k) {
                put(-1,             y, z, Lattice::missingXHi[k], tag);
                put(grid.local_nx,  y, z, Lattice::missingXLo[k], tag);
            }
    cudaMemcpy(d_fields->d_f, host.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    halo->ExchangeLBM(*d_fields);
    cudaDeviceSynchronize();
    auto out = CopyToHost(d_fields->d_f);

    // Interior owned cells (not on ±X boundary): untouched.
    for (int z = 0; z < grid.local_nz; ++z)
        for (int y = 0; y < grid.local_ny; ++y)
            for (int x = 1; x < grid.local_nx - 1; ++x)
                for (int i = 0; i < Lattice::ndir; ++i)
                    EXPECT_DOUBLE_EQ(At(out, x, y, z, i), canary)
                        << "interior owned canary lost at ("
                        << x << "," << y << "," << z << "," << i << ")";

    // At -X boundary (x=0): only dirs missingXLo overwritten; other dirs
    // keep the canary. Same at +X boundary for dirs missingXHi.
    for (int z = 0; z < grid.local_nz; ++z) {
        for (int y = 0; y < grid.local_ny; ++y) {
            std::array<bool, Lattice::ndir> lo_crossed{};
            std::array<bool, Lattice::ndir> hi_crossed{};
            for (int k = 0; k < 5; ++k) {
                lo_crossed[Lattice::missingXLo[k]] = true;
                hi_crossed[Lattice::missingXHi[k]] = true;
            }
            for (int i = 0; i < Lattice::ndir; ++i) {
                if (!lo_crossed[i]) {
                    EXPECT_DOUBLE_EQ(At(out, 0, y, z, i), canary)
                        << "-X boundary non-crossing canary lost at "
                        << "(0," << y << "," << z << ") dir=" << i;
                }
                if (!hi_crossed[i]) {
                    EXPECT_DOUBLE_EQ(At(out, grid.local_nx - 1, y, z, i), canary)
                        << "+X boundary non-crossing canary lost at "
                        << "(local_nx-1," << y << "," << z << ") dir=" << i;
                }
            }
        }
    }
}

}  // namespace
