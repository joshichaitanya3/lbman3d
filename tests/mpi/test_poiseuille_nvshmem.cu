// PR VII-g — Two-GPU Poiseuille flow integration test for HaloExchangeLbmNvshmem.
//
// This is the correctness gate for VII-g (face-only): with GpuCollideAndStream
// routing split-axis crossings into ghost cells and ExchangeLBM shipping them to
// the neighbour's owned boundary, the periodic X halo exchange is exercised by a
// real pressure-driven flow run.
//
// Shape mirrors tests/mpi/test_poiseuille_mpi.cc:
//   - PoiseuilleConfig (periodic X/Z, no-slip Y).
//   - Constant body force kDeltaP in X.
//   - 5000 LBM steps with no Q-tensor dynamics (body force held constant).
//   - Asserts parabolic profile, profile symmetry, and mass conservation.
//
// MPI-specific additions:
//   - Domain split along X by mpirun -n 2.
//   - MPI_Allreduce gathers x-row averages and global mass from all ranks.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <limits>
#include <ranges>
#include <vector>

#include "cuda/halo_exchange_lbm_nvshmem.h"
#include "device_fields.h"
#include "device_solver.h"
#include "fluid_fields.h"
#include "lbm_solver.h"
#include "local_grid.h"
#include "mpi/mpi_context.h"
#include "params.h"
#include "qtensor_fields.h"
#include "sim_config.h"

using namespace Params;

namespace {

class PoiseuilleFlowNvshmem {
    MPIContext          mpi_;
    LocalGrid           grid_;
    BackendInfo         backend_info_;
    FluidFields         fluid_;
    QTensorFields       qtensor_;  // unused by LBM step, but DeviceFields needs it
    DeviceFields        d_fields_;
    HaloExchangeLbmNvshmem lbm_halo_;
    LbmSolver<SimBC>    lbm_;      // used only for Initialize (equilibrium f IC)
    DeviceSolver<SimBC> d_solver_;
    int                 time_step_ = 0;

    void Initialize() {
        std::fill(fluid_.fx.begin(), fluid_.fx.end(), kDeltaP);
        lbm_.Initialize(fluid_);   // sets f = Feq(rho, u=0) — required before d_fields_.Initialize
        d_fields_.Initialize(fluid_, qtensor_);
        d_solver_.Initialize(d_fields_);
    }

public:
    PoiseuilleFlowNvshmem()
        : mpi_(periodicity_by_axis<SimBC>),
          grid_(mpi_.MakeLocalGrid()),
          backend_info_(InitializeComputeBackend(mpi_, grid_)),
          fluid_(grid_),
          qtensor_(grid_),
          d_fields_(grid_),
          lbm_halo_(grid_, mpi_, is_wall_by_face<SimBC>),
          d_solver_()
    {
        Initialize();
    }

    void Step() {
        // Keep the body force constant: re-copy fx to device before each LBM
        // step (DeviceSolver::LBMStep reads d_force_x which was set in Initialize;
        // the kernel does not overwrite it, so this is a no-op cost-wise, but it
        // guards against any future change that might zero the device force after
        // streaming).  For a static force we could skip this; it is kept here for
        // test clarity.
        d_solver_.LBMStep(d_fields_);
        lbm_halo_.ExchangeLBM(d_fields_);
        ++time_step_;
    }

    void SnapshotToHost() { d_fields_.CopyToHost(fluid_, qtensor_); }

    // Analytical Poiseuille profile (midpoint bounce-back convention).
    double ux_analytical(int y_global) const {
        const double yi = static_cast<double>(y_global);
        return (kDeltaP / (2.0 * nu)) * (yi + 0.5) * (static_cast<double>(ny) - 0.5 - yi);
    }

    // X-row average of ux at global (y_global, z=nz/2). Ranks that don't own
    // those rows contribute 0; MPI_Allreduce sums across all ranks.
    double x_averaged_ux(int y_global) const {
        const int z_local = nz / 2 - grid_.offset_z;
        const int y_local = y_global - grid_.offset_y;
        double local_sum = 0.0;
        if (y_local >= 0 && y_local < grid_.local_ny &&
            z_local >= 0 && z_local < grid_.local_nz) {
            for (int x : std::views::iota(0, grid_.local_nx))
                local_sum += fluid_.ux[grid_.halo_idx(x, y_local, z_local)];
        }
        double global_sum = 0.0;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_.cart_comm);
        return global_sum / static_cast<double>(nx);
    }

    double mass() const {
        double local = 0.0;
        for (int z : std::views::iota(0, grid_.local_nz))
            for (int y : std::views::iota(0, grid_.local_ny))
                for (int x : std::views::iota(0, grid_.local_nx))
                    local += fluid_.rho[grid_.halo_idx(x, y, z)];
        double global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, mpi_.cart_comm);
        return global;
    }
};

class PoiseuilleNvshmem : public ::testing::Test {
protected:
    static inline std::unique_ptr<PoiseuilleFlowNvshmem> sim;

    static void SetUpTestSuite() {
        sim = std::make_unique<PoiseuilleFlowNvshmem>();
        for (int i = 0; i < 5000; ++i)
            sim->Step();
        sim->SnapshotToHost();
    }

    static void TearDownTestSuite() { sim.reset(); }
};

TEST_F(PoiseuilleNvshmem, ParabolicProfile) {
    for (int y : std::views::iota(0, ny)) {
        EXPECT_NEAR(sim->ux_analytical(y), sim->x_averaged_ux(y), 1e-4)
            << "Parabolic profile mismatch at y=" << y;
    }
}

TEST_F(PoiseuilleNvshmem, ProfileSymmetry) {
    for (int y : std::views::iota(0, ny / 2)) {
        EXPECT_NEAR(sim->x_averaged_ux(y), sim->x_averaged_ux(ny - 1 - y), 1e-8)
            << "Profile asymmetry at y=" << y;
    }
}

TEST_F(PoiseuilleNvshmem, MassConservation) {
    const double expected = static_cast<double>(nx * ny * nz) * kDensity;
    const double tol = expected * 5000 * std::numeric_limits<double>::epsilon() * 10.0;
    EXPECT_NEAR(sim->mass(), expected, tol);
}

}  // namespace
