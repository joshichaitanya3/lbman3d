#include <gtest/gtest.h>
#include <limits>
#include "params.h"
#include "sim_config.h"
#include "offsets.h"
#include "fluid_fields.h"
#include "qtensor_fields.h"
#include "device_fields.h"
#include "device_solver.h"
#include "lbm_solver.h"
#include "local_grid.h"

using namespace Params;

// Uses LbmSolver to simulate Poiseuille flow in 3D.
template<typename BC>
class PoiseuilleFlowBenchmark {
    LocalGrid        grid_;
    FluidFields      fluid_;
    QTensorFields    qtensor_;
    LbmSolver<BC>    lbm_;
    DeviceFields     d_fields_;
    DeviceSolver<BC> d_solver_;
    int              time_step_ = 0;

    void Initialize() {
        std::fill(fluid_.fx.begin(), fluid_.fx.end(), kDeltaP);
        lbm_.Initialize(fluid_);
        #ifdef SIM_WITH_CUDA
        d_fields_.Initialize(fluid_, qtensor_);
        d_solver_.Initialize(d_fields_);
        #endif // SIM_WITH_CUDA
    }
public:
    // Default: constant-alpha active nematic.
    // Supply a QTensorSolver subclass to override the activity model.
    PoiseuilleFlowBenchmark() :
        grid_(LocalGrid::SingleRank()),
        fluid_(grid_),
        qtensor_(grid_),
        d_fields_(grid_)
    {
        Initialize();
    }

    void LBMStep() {
        #ifdef SIM_WITH_CUDA
        d_solver_.LBMStep(d_fields_);
        #else
        lbm_.LatticeBoltzmannStep(fluid_);
        #endif
    }
    // LBM step.
    void Step() {
        LBMStep();
        ++time_step_;
    }

    double ux_analytical(int y) {
        double yi = static_cast<double>(y);
        return (kDeltaP / (2 * nu)) * (yi + 0.5) * (static_cast<double>(ny) - 0.5 - yi);
    }

    double x_averaged_ux(int y) {
        const int z = nz/2;
        double avg = 0.0;
        for (int x : std::views::iota(0, nx)) {
            avg += fluid_.ux[grid_.halo_idx(x, y, z)];
        }
        avg /= static_cast<double>(nx);
        return avg;
    }

    double mass() {
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

};

struct PoiseuilleBC {
    using XLo = WallSpec<Periodic, Periodic>;
    using XHi = WallSpec<Periodic, Periodic>;
    using YLo = WallSpec<Neumann, NoSlip>;
    using YHi = WallSpec<Neumann, NoSlip>;
    using ZLo = WallSpec<Periodic, Periodic>;
    using ZHi = WallSpec<Periodic, Periodic>;
    static constexpr std::string_view name = "Poiseuille";
};

template struct PoiseuilleFlowBenchmark<PoiseuilleBC>;

class PoiseuilleFlow : public ::testing::Test {
    protected:
    static inline PoiseuilleFlowBenchmark<PoiseuilleBC> sim{};
    // Runs ONCE before all tests in this fixture
    static void SetUpTestSuite() {
        for (int i = 0; i < 5000; i++)
            sim.Step();
    }

};


TEST_F(PoiseuilleFlow, ParabolicProfile) {
    for (int y : std::views::iota(0, ny)) {
        double val1 = sim.ux_analytical(y);
        double val2 = sim.x_averaged_ux(y);
        EXPECT_NEAR(val1, val2, 1e-4);
    }
}

TEST_F(PoiseuilleFlow, ProfileSymmetry) {
    for (int y : std::views::iota(0, ny/2)) {
        double val1 = sim.x_averaged_ux(y);
        double val2 = sim.x_averaged_ux(ny-1-y);
        EXPECT_NEAR(val1, val2, 1e-8);
    }
}

TEST_F(PoiseuilleFlow, MassConservation) {
    double mass = sim.mass();
    double expected_mass = static_cast<double>(nx * ny * nz) * kDensity;
    // Worst-case bound: N cells * T steps * machine epsilon for linear accumulation
    // of floating-point roundoff through BGK collision and streaming.
    const double tol = static_cast<double>(nx * ny * nz) * 5000 * std::numeric_limits<double>::epsilon();
    EXPECT_NEAR(mass, expected_mass, tol);
}
