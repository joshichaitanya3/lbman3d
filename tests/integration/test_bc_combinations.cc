#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <ranges>
#include "params.h"
#include "boundary_handler.h"
#include "fluid_fields.h"
#include "qtensor_fields.h"
#include "device_fields.h"
#include "device_solver.h"
#include "lbm_solver.h"
#include "qtensor_solver.h"
#include "local_grid.h"

using namespace Params;

// Each scenario below exercises a single BC mechanism in isolation — either
// pure LBM (velocity BCs) or pure Q-tensor relaxation (Q-tensor wall BCs) —
// rather than the fully coupled active-nematic loop, so a failure points
// straight at the mechanism under test instead of the coupling between them.

// ─────────────────────────────────────────────────────────────────────────────
// Pure-LBM benchmark: FluidFields + LbmSolver<BC> only, no Q-tensor
// dynamics. Mirrors test_poiseuille.cc's PoiseuilleFlowBenchmark; qtensor_
// is otherwise-unused state that DeviceFields::Initialize needs as an
// argument on the GPU path.
// ─────────────────────────────────────────────────────────────────────────────
template<typename BC>
class LbmOnlyBenchmark {
    LocalGrid        grid_;
    FluidFields      fluid_;
    QTensorFields    qtensor_;
    LbmSolver<BC>    lbm_;
    DeviceFields     d_fields_;
    DeviceSolver<BC> d_solver_;

public:
    LbmOnlyBenchmark() :
        grid_(LocalGrid::SingleRank()),
        fluid_(grid_),
        qtensor_(grid_),
        d_fields_(grid_)
    { Reinitialize(); }

    FluidFields& Fluid() { return fluid_; }

    // Re-derives f = f_eq from the current rho/ux/uy/uz. Call after directly
    // mutating Fluid() (e.g. to seed an initial velocity or a constant body
    // force) so the distributions stay consistent with the macroscopic state.
    void Reinitialize() {
        lbm_.Initialize(fluid_);
        #ifdef SIM_WITH_CUDA
        d_fields_.Initialize(fluid_, qtensor_);
        d_solver_.Initialize(d_fields_);
        #endif
    }

    void Step() {
        #ifdef SIM_WITH_CUDA
        d_solver_.LBMStep(d_fields_);
        #else
        lbm_.LatticeBoltzmannStep(fluid_);
        #endif
    }

    double TotalRho() const {
        double total = 0.0;
        for (double r : fluid_.rho) total += r;
        return total;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// ZeroActivitySolver: same subclass as the A2 (qt_relaxation) test — disables
// the active-stress contribution to the body force and seeds a moderately-
// ordered, fixed-seed IC so pure Beris-Edwards relaxation is deterministic.
// ─────────────────────────────────────────────────────────────────────────────
template<typename BC>
class ZeroActivitySolver : public QTensorSolver<BC> {
public:
    using QTensorSolver<BC>::QTensorSolver;

    void SetActiveStressAndComputeBodyForce(FluidFields&, const QTensorFields&) const override {}

    void Initialize(QTensorFields& qf) const override {
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> noise_dist(-NOISE, NOISE);
        LocalGrid& g = qf.grid;
        for (int z : std::views::iota(0, g.local_nz)) {
            for (int y : std::views::iota(0, g.local_ny)) {
                for (int x : std::views::iota(0, g.local_nx)) {
                    const int idxp = g.halo_idx(x, y, z);
                    qf.qxx[idxp] = 0.33 + noise_dist(gen);
                    qf.qxy[idxp] = noise_dist(gen);
                    qf.qxz[idxp] = noise_dist(gen);
                    qf.qyy[idxp] = -0.15 + noise_dist(gen);
                    qf.qyz[idxp] = noise_dist(gen);
                }
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pure Q-tensor benchmark: QTensorFields + QTensorSolver<BC> only, velocity
// stays zero throughout (never touched). Mirrors A2's QTensorRelaxationBenchmark,
// trimmed to what the anchoring-convergence test needs.
// ─────────────────────────────────────────────────────────────────────────────
template<typename BC>
class QOnlyBenchmark {
    FluidFields   fluid_;
    QTensorFields qtensor_;
    std::unique_ptr<QTensorSolver<BC>> qtensor_solver_;

public:
    explicit QOnlyBenchmark(std::unique_ptr<QTensorSolver<BC>> solver)
        : qtensor_solver_(std::move(solver))
    {
        qtensor_solver_->Initialize(qtensor_);
    }

    void Step() { qtensor_solver_->Step(qtensor_, fluid_); }

    const QTensorFields& Qtensor() const { return qtensor_; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 1: FullyPeriodicConfig — mass conservation.
// ─────────────────────────────────────────────────────────────────────────────

TEST(BCCombinations, FullyPeriodicMassConservation) {
    LbmOnlyBenchmark<FullyPeriodicConfig> sim;
    const double mass0 = sim.TotalRho();
    for (int step = 0; step < 500; ++step) sim.Step();
    const double mass1 = sim.TotalRho();

    // Worst-case bound: N cells * T steps * machine epsilon for linear
    // accumulation of floating-point roundoff through BGK collision and
    // streaming (same bound as test_poiseuille.cc's MassConservation).
    const double tol = static_cast<double>(nx * ny * nz) * 500 * std::numeric_limits<double>::epsilon();
    EXPECT_NEAR(mass1, mass0, tol);
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 2: ChannelConfig (NoSlip on X and Y) — mass conservation, and a
// small initial velocity perturbation damping to ~0 at the NoSlip walls.
// ─────────────────────────────────────────────────────────────────────────────

TEST(BCCombinations, ChannelNoSlipWallsDamping) {
    LbmOnlyBenchmark<ChannelConfig> sim;
    // ux=0 at t=0 would trivially satisfy the wall condition; seed a small
    // uniform perturbation so the test actually exercises the damping.
    std::fill(sim.Fluid().ux.begin(), sim.Fluid().ux.end(), 0.01);
    sim.Reinitialize();

    const double mass0 = sim.TotalRho();
    for (int step = 0; step < 500; ++step) sim.Step();
    const double mass1 = sim.TotalRho();

    const double tol = static_cast<double>(nx * ny * nz) * 500 * std::numeric_limits<double>::epsilon();
    EXPECT_NEAR(mass1, mass0, tol);

    const LocalGrid& g = sim.Fluid().grid;
    for (int z : std::views::iota(0, nz)) {
        for (int x : std::views::iota(0, nx)) {
            EXPECT_NEAR(sim.Fluid().ux[g.halo_idx(x, 0, z)], 0.0, 1e-6);
            EXPECT_NEAR(sim.Fluid().ux[g.halo_idx(x, ny - 1, z)], 0.0, 1e-6);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 3: SpecularReflection on Z (custom BC; Periodic X/Y). A uniform
// tangential body force should keep ux exactly uniform across z (free-slip
// walls apply zero shear, so no boundary layer forms, unlike NoSlip), while
// the normal component uz stays 0 at both walls (no-penetration).
// ─────────────────────────────────────────────────────────────────────────────

struct SpecularZConfig {
    using XLo = WallSpec<Periodic, Periodic>;
    using XHi = WallSpec<Periodic, Periodic>;
    using YLo = WallSpec<Periodic, Periodic>;
    using YHi = WallSpec<Periodic, Periodic>;
    using ZLo = WallSpec<Neumann, SpecularReflection>;
    using ZHi = WallSpec<Neumann, SpecularReflection>;
    static constexpr std::string_view name = "SpecularZ";
};

TEST(BCCombinations, SpecularZFreeSlip) {
    LbmOnlyBenchmark<SpecularZConfig> sim;
    std::fill(sim.Fluid().fx.begin(), sim.Fluid().fx.end(), 1e-5);
    sim.Reinitialize();

    for (int step = 0; step < 500; ++step) sim.Step();

    const LocalGrid& g = sim.Fluid().grid;
    const int x0 = nx / 2, y0 = ny / 2;
    const double ux_mid = sim.Fluid().ux[g.halo_idx(x0, y0, nz / 2)];
    for (int z : std::views::iota(0, nz)) {
        EXPECT_NEAR(sim.Fluid().ux[g.halo_idx(x0, y0, z)], ux_mid, 1e-9);
    }

    for (int y : std::views::iota(0, ny)) {
        for (int x : std::views::iota(0, nx)) {
            EXPECT_NEAR(sim.Fluid().uz[g.halo_idx(x, y, 0)], 0.0, 1e-9);
            EXPECT_NEAR(sim.Fluid().uz[g.halo_idx(x, y, nz - 1)], 0.0, 1e-9);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 4: Anchoring<0.5, pi/4, 0> on Z faces (custom BC; Periodic X/Y).
// Q at the z=0 wall nodes should converge to the prescribed anchoring target.
// ─────────────────────────────────────────────────────────────────────────────

struct AnchoredZConfig {
    using XLo = WallSpec<Periodic, Periodic>;
    using XHi = WallSpec<Periodic, Periodic>;
    using YLo = WallSpec<Periodic, Periodic>;
    using YHi = WallSpec<Periodic, Periodic>;
    using ZLo = WallSpec<Anchoring<0.5, std::numbers::pi / 4.0, 0.0>, NoSlip>;
    using ZHi = WallSpec<Anchoring<0.5, std::numbers::pi / 4.0, 0.0>, NoSlip>;
    static constexpr std::string_view name = "AnchoredZ";
};

TEST(BCCombinations, AnchoringConvergesAtWall) {
    QOnlyBenchmark<AnchoredZConfig> sim{std::make_unique<ZeroActivitySolver<AnchoredZConfig>>()};
    for (int step = 0; step < 500; ++step) sim.Step();

    using ZLoQ = AnchoredZConfig::ZLo::QBC;
    const SymTrLessTensor5 target = AnchoredQ<ZLoQ>();
    const QTensorFields& qf = sim.Qtensor();

    for (int y : std::views::iota(0, ny)) {
        for (int x : std::views::iota(0, nx)) {
            const int i = qf.grid.halo_idx(x, y, 0);
            EXPECT_NEAR(qf.qxx[i], target.xx, 0.01);
            EXPECT_NEAR(qf.qxy[i], target.xy, 0.01);
            EXPECT_NEAR(qf.qxz[i], target.xz, 0.01);
            EXPECT_NEAR(qf.qyy[i], target.yy, 0.01);
            EXPECT_NEAR(qf.qyz[i], target.yz, 0.01);
        }
    }
}
