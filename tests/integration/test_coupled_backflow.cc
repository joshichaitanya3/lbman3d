#include <gtest/gtest.h>
#include "params.h"
#include "sim_config.h"
#include "offsets.h"
#include "fluid_fields.h"
#include "qtensor_fields.h"
#include "device_fields.h"
#include "device_solver.h"
#include "lbm_solver.h"
#include "qtensor_solver.h"
#include "analysis_fields.h"
#include "local_grid.h"

#include <vector>
#include <numeric>
#include <random>
#include <ranges>
#include <cmath>

/*
 * A7: coupled Q <-> flow loop.
 *
 * Every other integration test drives exactly one solver: test_poiseuille runs
 * LbmSolver alone, test_qtensor_relaxation runs QTensorSolver alone with zero
 * velocity, and test_bc_combinations zeroes the body force via
 * ZeroActivitySolver. Nothing exercised the round trip
 *
 *     Q -> body force (backflow + passive stress) -> flow -> advection/co-rotation of Q
 *
 * which is where the two solvers actually couple. This test closes that gap.
 *
 * Physics under test: with ALPHA = 0 (no activity) and MU = 0 (no friction),
 * the passive system is a closed energy budget. Elastic distortion in Q drives
 * flow through the backflow force -H:grad Q; viscous and rotational dissipation
 * drain both reservoirs.
 *
 * The observable that actually discriminates a backflow sign error is the
 * KINETIC energy, not the total. Measured on this scenario, the nematic free
 * energy changes by O(2.4) over the run while the kinetic energy is O(6e-6) --
 * five orders of magnitude smaller -- so a total-energy bound cannot resolve
 * injection at the backflow scale, and indeed passes with the sign error
 * present. What does discriminate: with the correct sign the flow decays
 * (5.56e-6 -> 1.30e-6), while with the wrong sign backflow keeps pumping and
 * the flow is sustained, rising to 7.00e-6 and ending near 6.6e-6.
 *
 * So the total-energy assertions here are coupled-loop coverage, and
 * KineticEnergyDoesNotGrowAfterTransient / FlowDecaysInPassiveSystem are the
 * regression guards. The pointwise algebra is guarded separately and exactly by
 * tests/unit/test_backflow_contraction.cc.
 *
 * Runs at the production DT = 0.05 (see tests/params/coupled/params.h).
 *
 * Note on NaN checking: this file deliberately does NOT use
 * EXPECT_TRUE(std::isfinite(x)). The Release build sets -Ofast, which implies
 * -ffinite-math-only and folds std::isnan/std::isfinite to a constant, so such
 * an assertion silently passes on a diverged run. EXPECT_NEAR compares
 * fabs(a-b) <= tol, which is false for NaN, so it still fails correctly.
 */

using namespace Params;

namespace {

// Passive nematic: keep the default QTensorSolver activity model (ALPHA = 0
// makes the active stress vanish) but seed a reproducible distorted IC so the
// elastic energy has somewhere to go.
template<typename BC>
class DistortedICSolver : public QTensorSolver<BC> {
public:
    using QTensorSolver<BC>::QTensorSolver;
    void Initialize(QTensorFields& qf) const override {
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> noise_dist(-NOISE, NOISE);
        for (int z : std::views::iota(0, nz)) {
            for (int y : std::views::iota(0, ny)) {
                for (int x : std::views::iota(0, nx)) {
                    qf.qxx[qf.grid.halo_idx(x, y, z)] = 0.33 + noise_dist(gen);
                    qf.qxy[qf.grid.halo_idx(x, y, z)] = noise_dist(gen);
                    qf.qxz[qf.grid.halo_idx(x, y, z)] = noise_dist(gen);
                    qf.qyy[qf.grid.halo_idx(x, y, z)] = -0.15 + noise_dist(gen);
                    qf.qyz[qf.grid.halo_idx(x, y, z)] = noise_dist(gen);
                }
            }
        }
    }
};

template<typename BC>
class CoupledSim {
    LocalGrid     grid_;
    FluidFields   fluid_;
    QTensorFields qtensor_;
    LbmSolver<BC> lbm_;
    std::unique_ptr<QTensorSolver<BC>> qtensor_solver_;
    DeviceFields     d_fields_;
    DeviceSolver<BC> d_solver_;
    int time_step_ = 0;

public:
    explicit CoupledSim(std::unique_ptr<QTensorSolver<BC>> solver)
        :
            grid_(LocalGrid::SingleRank()),
            fluid_(grid_),
            qtensor_(grid_),
            qtensor_solver_(std::move(solver)),
            d_fields_(grid_)
    {
        lbm_.Initialize(fluid_);
        qtensor_solver_->Initialize(qtensor_);
        #ifdef SIM_WITH_CUDA
        d_fields_.Initialize(fluid_, qtensor_);
        d_solver_.Initialize(d_fields_);
        #endif
    }

    // Same order as ActiveNematicSim::Step(): Q update + body force, then LBM.
    void Step() {
        #ifdef SIM_WITH_CUDA
        d_solver_.QTensorStep(d_fields_);
        d_solver_.LBMStep(d_fields_);
        #else
        qtensor_solver_->Step(qtensor_, fluid_);
        lbm_.LatticeBoltzmannStep(fluid_);
        #endif
        ++time_step_;
    }

    void SyncToHost() {
        #ifdef SIM_WITH_CUDA
        d_fields_.CopyToHost(fluid_, qtensor_);
        #endif
    }

    double TotalMass() {
        SyncToHost();
        double mass = 0.0;
        for (int z : std::views::iota(0, nz)) {
            for (int y : std::views::iota(0, ny)) {
                for (int x : std::views::iota(0, nx)) {
                    mass += fluid_.rho[grid_.halo_idx(x, y, z)];
                }
            }
        }
        return mass;
    }

    double KineticEnergy() {
        SyncToHost();
        double ke = 0.0;
        for (int z : std::views::iota(0, nz)) {
            for (int y : std::views::iota(0, ny)) {
                for (int x : std::views::iota(0, nx)) {
                    ke += 0.5 * fluid_.rho[grid_.halo_idx(x, y, z)] * (fluid_.ux[grid_.halo_idx(x, y, z)]*fluid_.ux[grid_.halo_idx(x, y, z)]
                                       + fluid_.uy[grid_.halo_idx(x, y, z)]*fluid_.uy[grid_.halo_idx(x, y, z)]
                                       + fluid_.uz[grid_.halo_idx(x, y, z)]*fluid_.uz[grid_.halo_idx(x, y, z)]);
                }
            }
        }
        
        return ke;
    }

    double NematicEnergy() {
        SyncToHost();
        return TotalNematicFreeEnergy<BC>(qtensor_);
    }

    double TotalEnergy() { return NematicEnergy() + KineticEnergy(); }

    double MaxSpeed() {
        SyncToHost();
        double m = 0.0;
        for (int z : std::views::iota(0, nz)) {
            for (int y : std::views::iota(0, ny)) {
                for (int x : std::views::iota(0, nx)) {
                    const double s = std::sqrt(fluid_.ux[grid_.halo_idx(x, y, z)]*fluid_.ux[grid_.halo_idx(x, y, z)]
                                     + fluid_.uy[grid_.halo_idx(x, y, z)]*fluid_.uy[grid_.halo_idx(x, y, z)]
                                     + fluid_.uz[grid_.halo_idx(x, y, z)]*fluid_.uz[grid_.halo_idx(x, y, z)]);
                    if (s > m) m = s;
                }
            }
        }
        return m;
    }

    int GetTimeStep() const { return time_step_; }
};

constexpr int kSteps = 2000;
constexpr int kCheckEvery = 100;
constexpr int kCheckpoints = kSteps / kCheckEvery;

} // namespace

template class CoupledSim<FullyPeriodicConfig>;
template class DistortedICSolver<FullyPeriodicConfig>;

class CoupledBackflow : public ::testing::Test {
protected:
    static inline CoupledSim<FullyPeriodicConfig> sim{
        std::make_unique<DistortedICSolver<FullyPeriodicConfig>>()};

    static inline std::vector<double> total_energy;
    static inline std::vector<double> kinetic_energy;
    static inline std::vector<double> mass;
    static inline double initial_mass = 0.0;
    static inline double max_speed = 0.0;

    static void SetUpTestSuite() {
        initial_mass = sim.TotalMass();
        for (int i = 0; i < kSteps; ++i) {
            if (i % kCheckEvery == 0) {
                total_energy.push_back(sim.TotalEnergy());
                kinetic_energy.push_back(sim.KineticEnergy());
                mass.push_back(sim.TotalMass());
            }
            sim.Step();
            max_speed = std::max(max_speed, sim.MaxSpeed());
        }
        total_energy.push_back(sim.TotalEnergy());
        kinetic_energy.push_back(sim.KineticEnergy());
        mass.push_back(sim.TotalMass());
    }
};

// Mass is conserved exactly by push-style streaming into a separate buffer;
// this also fails on NaN, since EXPECT_NEAR's comparison is false for NaN.
TEST_F(CoupledBackflow, MassConservation) {
    // kDensity, not RHO: FluidFields seeds rho with Params::kDensity
    // (src/fluid_fields.cc:11) and LbmSolver::Initialize only *reads* ff.rho
    // (src/lbm_solver.tpp:38), so RHO never reaches the density field.
    const double expected = nx * ny * nz * kDensity;
    ASSERT_NEAR(initial_mass, expected, 1e-9)
        << "initial density should be kDensity everywhere";
    for (std::size_t i = 0; i < mass.size(); ++i) {
        EXPECT_NEAR(mass[i] / initial_mass, 1.0, 1e-10)
            << "mass drift at checkpoint " << i;
    }
}

// The backflow force must actually do work on the fluid, otherwise the coupling
// is not being exercised and the energy assertions below are vacuous.
TEST_F(CoupledBackflow, BackflowDrivesFlow) {
    ASSERT_NEAR(kinetic_energy.front(), 0.0, 1e-30)
        << "fluid should start at rest";
    EXPECT_GT(*std::max_element(kinetic_energy.begin(), kinetic_energy.end()), 1e-16)
        << "backflow never transferred energy into the flow; coupling untested";
}

// Decisive regression guard for the backflow contraction sign.
//
// Past the startup transient, a passive nematic (ALPHA = 0, MU = 0) with no
// external forcing cannot gain kinetic energy: the elastic reservoir is
// draining and viscosity dissipates what reaches the flow. A backflow force
// with the wrong sign on its cross terms instead pumps energy into the fluid,
// so the kinetic energy climbs above its post-transient value and stays there.
//
// Measured on this scenario: with the sign error, KE rises from 6.45e-6 at
// checkpoint 1 to 7.00e-6 by checkpoint 6 and remains ~6.6e-6 at the end.
// With the correct sign it decays monotonically, 5.56e-6 -> 1.30e-6.
TEST_F(CoupledBackflow, KineticEnergyDoesNotGrowAfterTransient) {
    ASSERT_GE(kinetic_energy.size(), 3u);
    const double ke_ref = kinetic_energy[1];   // one checkpoint past startup
    for (std::size_t i = 1; i < kinetic_energy.size(); ++i) {
        EXPECT_LE(kinetic_energy[i], ke_ref * (1.0 + 1e-9))
            << "kinetic energy grew past its post-transient value at checkpoint "
            << i << " (ref = " << ke_ref << ", KE = " << kinetic_energy[i]
            << "); backflow is injecting energy into a passive system";
    }
}

// The flow must actually decay, not merely stay bounded. The 0.5 factor is
// calibrated to sit well clear of both implementations: the correct sign gives
// a ratio of 0.23 here, the sign error gives 1.02.
TEST_F(CoupledBackflow, FlowDecaysInPassiveSystem) {
    const double ratio = kinetic_energy.back() / kinetic_energy[1];
    EXPECT_LT(ratio, 0.5)
        << "kinetic energy was sustained rather than decaying (ratio " << ratio << ")";
}

// Valid physics for the coupled passive system, but NOT sensitive to the
// backflow sign error: measured, the nematic free energy changes by O(2.4) over
// this run while the kinetic energy is O(6e-6), so injection at the backflow
// scale is five orders of magnitude below the resolution of this assertion.
// Kept as coupled-loop coverage; the guards above are what catch the sign bug.
TEST_F(CoupledBackflow, TotalEnergyDoesNotGrow) {
    const double e0 = total_energy.front();
    for (std::size_t i = 1; i < total_energy.size(); ++i) {
        EXPECT_LE(total_energy[i], e0 + 1e-9)
            << "total energy exceeded its initial value at checkpoint " << i
            << " (E0 = " << e0 << ", E = " << total_energy[i] << ")";
    }
}

// Over the whole run the system should have relaxed, not merely stayed bounded.
TEST_F(CoupledBackflow, RelaxesOverRun) {
    EXPECT_LT(total_energy.back(), total_energy.front())
        << "passive system did not relax over " << kSteps << " steps";
}

// A diverging run shows up as a velocity far outside the low-Mach LBM regime.
TEST_F(CoupledBackflow, StaysInLowMachRegime) {
    EXPECT_LT(max_speed, 0.1)
        << "max speed " << max_speed << " left the low-Mach regime";
}
