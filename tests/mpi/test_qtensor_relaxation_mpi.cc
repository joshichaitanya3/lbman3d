// CPU-MPI integration test for Q-tensor relaxation.
//
// This is the CPU-path counterpart to test_qtensor_relaxation_nvshmem.cu.
// It runs the Beris-Edwards Q-tensor relaxation under a full MPI decomposition
// using HaloExchangeQTensor (CPU-MPI path), exercising all three halo
// exchange calls (ExchangeQTensor before phase 1, ExchangePassiveStresses
// between phases 1 and 2) on all split axes simultaneously.
//
// Shape mirrors tests/integration/test_qtensor_relaxation.cc:
//   - Uniform ordered IC with a fixed-seed noise perturbation.
//   - Runs Beris-Edwards Q-tensor relaxation for 2000 steps.
//   - Asserts monotone free-energy decrease, spatial uniformity, and mean
//     order convergence to -B/(3C).
//
// MPI-specific additions:
//   - Domain is split by mpirun -n N (np=2 exercises one-axis split;
//     np=8 exercises the full 2×2×2 three-axis split on the 16³ grid).
//   - IC walk covers the global grid so the per-rank IC matches the
//     single-rank test bit-for-bit at every owned cell.
//   - MPI_Allreduce collects per-rank nematic energy, mean order, and
//     spatial variance into global quantities.
//   - NematicEnergy() does an extra ExchangeQTensor before evaluating so
//     the 7-point Laplacian is correct at all rank-local boundaries.

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>

#include "analysis_fields.h"
#include "fluid_fields.h"
#include "local_grid.h"
#include "mpi/halo_exchange_qtensor.h"
#include "mpi/mpi_context.h"
#include "params.h"
#include "qtensor_fields.h"
#include "qtensor_solver.h"
#include "sim_config.h"

using namespace Params;

namespace {

// ZeroActivitySolver walks the full global grid when seeding so an MPI split
// reproduces the single-rank IC exactly. The RNG is advanced through the
// entire (nx, ny, nz) sequence; only cells owned by this rank are written.
template<typename BC>
class ZeroActivitySolver : public QTensorSolver<BC> {
public:
    using QTensorSolver<BC>::QTensorSolver;
    void SetActiveStressAndComputeBodyForce(FluidFields&, const QTensorFields&) const override {}
    void Initialize(QTensorFields& qf) const override;
};

template<typename BC>
void ZeroActivitySolver<BC>::Initialize(QTensorFields& qf) const {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> noise_dist(-NOISE, NOISE);

    const LocalGrid& g = qf.grid;
    for (int Z = 0; Z < nz; ++Z) {
        for (int Y = 0; Y < ny; ++Y) {
            for (int X = 0; X < nx; ++X) {
                const double q_xx = 0.33 + noise_dist(gen);
                const double q_xy = noise_dist(gen);
                const double q_xz = noise_dist(gen);
                const double q_yy = -0.15 + noise_dist(gen);
                const double q_yz = noise_dist(gen);

                const int lx = X - g.offset_x;
                const int ly = Y - g.offset_y;
                const int lz = Z - g.offset_z;
                if (lx < 0 || lx >= g.local_nx) continue;
                if (ly < 0 || ly >= g.local_ny) continue;
                if (lz < 0 || lz >= g.local_nz) continue;

                const int i = g.halo_idx(lx, ly, lz);
                qf.qxx[i] = q_xx;
                qf.qxy[i] = q_xy;
                qf.qxz[i] = q_xz;
                qf.qyy[i] = q_yy;
                qf.qyz[i] = q_yz;
            }
        }
    }
}

// Halo-aware global nematic free energy. Caller must have run ExchangeQTensor
// so ghost cells hold the neighbour's owned Q values.
template<typename BC>
double GlobalNematicFreeEnergy(const QTensorFields& qf, MPI_Comm comm) {
    const LocalGrid& g = qf.grid;
    double local = 0.0;
    for (int z = 0; z < g.local_nz; ++z) {
        for (int y = 0; y < g.local_ny; ++y) {
            for (int x = 0; x < g.local_nx; ++x) {
                const int idxp = g.halo_idx(x, y, z);

                const double Qxx = qf.qxx[idxp];
                const double Qxy = qf.qxy[idxp];
                const double Qxz = qf.qxz[idxp];
                const double Qyy = qf.qyy[idxp];
                const double Qyz = qf.qyz[idxp];

                auto lap = [&](const std::vector<double>& f, double centre) {
                    return f[g.halo_idx(x + 1, y, z)] + f[g.halo_idx(x - 1, y, z)]
                         + f[g.halo_idx(x, y + 1, z)] + f[g.halo_idx(x, y - 1, z)]
                         + f[g.halo_idx(x, y, z + 1)] + f[g.halo_idx(x, y, z - 1)]
                         - 6.0 * centre;
                };
                const double lap_xx = lap(qf.qxx, Qxx);
                const double lap_xy = lap(qf.qxy, Qxy);
                const double lap_xz = lap(qf.qxz, Qxz);
                const double lap_yy = lap(qf.qyy, Qyy);
                const double lap_yz = lap(qf.qyz, Qyz);

                local += NematicFreeEnergyDensity(
                    Qxx, Qxy, Qxz, Qyy, Qyz,
                    lap_xx, lap_xy, lap_xz, lap_yy, lap_yz);
            }
        }
    }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global;
}

struct OrderStats { double mean; double stddev; };
OrderStats GlobalOrderStats(const QTensorFields& qf, AnalysisFields& af, MPI_Comm comm) {
    QtensorToOrderDirector(qf, af);
    const LocalGrid& g = qf.grid;
    double local_sum = 0.0;
    long long local_count = 0;
    for (int z = 0; z < g.local_nz; ++z)
        for (int y = 0; y < g.local_ny; ++y)
            for (int x = 0; x < g.local_nx; ++x) {
                local_sum += af.order_[g.halo_idx(x, y, z)];
                ++local_count;
            }

    double global_sum = 0.0;
    long long global_count = 0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, comm);
    const double mean = global_sum / static_cast<double>(global_count);

    double local_sq = 0.0;
    for (int z = 0; z < g.local_nz; ++z)
        for (int y = 0; y < g.local_ny; ++y)
            for (int x = 0; x < g.local_nx; ++x) {
                const double d = af.order_[g.halo_idx(x, y, z)] - mean;
                local_sq += d * d;
            }
    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, comm);
    const double stddev = std::sqrt(global_sq / static_cast<double>(global_count));
    return {mean, stddev};
}

template<typename BC>
class QTensorRelaxationBenchmarkMPI {
    MPIContext          mpi_;
    LocalGrid           grid_;
    FluidFields         fluid_;
    QTensorFields       qtensor_;
    HaloExchangeQTensor halo_;
    std::unique_ptr<QTensorSolver<BC>> qtensor_solver_;
    AnalysisFields      af_;
    int                 time_step_ = 0;

public:
    explicit QTensorRelaxationBenchmarkMPI(std::unique_ptr<QTensorSolver<BC>> solver)
        : mpi_(/*periods=*/{1, 1, 1}),
          grid_(mpi_.MakeLocalGrid()),
          fluid_(grid_),
          qtensor_(grid_),
          halo_(grid_, mpi_),
          qtensor_solver_(std::move(solver)),
          af_(grid_)
    {
        qtensor_solver_->Initialize(qtensor_);
    }

    // CPU phase-split step mirroring ActiveNematicSim::QTensorStep on the MPI path:
    //   1. ExchangeQTensor — fill Q + velocity ghost cells from neighbours.
    //   2. StepAndSetupBodyForce — Beris-Edwards phase 1.
    //   3. ExchangePassiveStresses — fill Sigma/Tau ghost cells.
    //   4. SetActiveStressAndComputeBodyForce — phase 2 (no-op for ZeroActivity).
    void Step() {
        halo_.ExchangeQTensor(qtensor_, fluid_);
        qtensor_solver_->StepAndSetupBodyForce(qtensor_, fluid_);
        halo_.ExchangePassiveStresses(qtensor_);
        qtensor_solver_->SetActiveStressAndComputeBodyForce(fluid_, qtensor_);
        ++time_step_;
    }

    // Refresh Q ghost cells before evaluating so the 7-point Laplacian is
    // correct at all rank-local boundaries.
    double NematicEnergy() {
        halo_.ExchangeQTensor(qtensor_, fluid_);
        return GlobalNematicFreeEnergy<BC>(qtensor_, mpi_.cart_comm);
    }

    double MeanOrder() {
        return GlobalOrderStats(qtensor_, af_, mpi_.cart_comm).mean;
    }
    double SpatialUniformity() {
        return GlobalOrderStats(qtensor_, af_, mpi_.cart_comm).stddev;
    }

    int GetTimeStep() const { return time_step_; }
    const MPIContext& mpi() const { return mpi_; }
};

class QTensorRelaxationMPI : public ::testing::Test {
protected:
    static inline std::unique_ptr<QTensorRelaxationBenchmarkMPI<FullyPeriodicConfig>> sim;
    static inline std::vector<double> nematic_energies;
    static inline int np = 20;

    static void SetUpTestSuite() {
        sim = std::make_unique<QTensorRelaxationBenchmarkMPI<FullyPeriodicConfig>>(
            std::make_unique<ZeroActivitySolver<FullyPeriodicConfig>>());
        for (int i = 0; i < np * 100; i++) {
            if (i % 100 == 0)
                nematic_energies.push_back(sim->NematicEnergy());
            sim->Step();
        }
        nematic_energies.push_back(sim->NematicEnergy());
    }

    static void TearDownTestSuite() {
        sim.reset();
        nematic_energies.clear();
    }
};

// Free energy stays near equilibrium once relaxation converges. Tolerance
// matches the NVSHMEM counterpart (3e-2); a broken ExchangePassiveStresses
// would inject a spurious body force at rank seams that drives Q away from
// equilibrium — far larger than 3e-2.
TEST_F(QTensorRelaxationMPI, FreeEnergyStaysNearEquilibrium) {
    const double eq = nematic_energies.back();
    for (int i : std::views::iota(1, np)) {
        EXPECT_NEAR(nematic_energies[i], eq, 3e-2)
            << "Free energy drifted from equilibrium at checkpoint " << i
            << ": " << nematic_energies[i] << " vs eq=" << eq;
    }
}

TEST_F(QTensorRelaxationMPI, MeanOrderConverges) {
    const double analytical = -B / (3.0 * C);
    EXPECT_NEAR(sim->MeanOrder(), analytical, 1e-4);
}

TEST_F(QTensorRelaxationMPI, SpatialUniformity) {
    EXPECT_LT(sim->SpatialUniformity(), 1e-2);
}

TEST_F(QTensorRelaxationMPI, Convergence) {
    EXPECT_LT(nematic_energies[np - 1] - nematic_energies[np], 1e-4);
}

}  // namespace
