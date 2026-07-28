#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include "params.h"
#include "sim_config.h"
#include "offsets.h"
#include "fluid_fields.h"
#include "qtensor_fields.h"
#include "device_fields.h"
#include "device_solver.h"
#include "lbm_solver.h"
#include "local_grid.h"
#include "mpi/mpi_context.h"
#include "mpi/halo_exchange_lbm.h"

using namespace Params;

// Uses LbmSolver to simulate Poiseuille flow in 3D.
template<typename BC>
class PoiseuilleFlowBenchmark {
    MPIContext       mpi_;
    LocalGrid        grid_;
    HaloExchangeLBM  halo_;
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
        mpi_(periodicity_by_axis<BC>),
        grid_(mpi_.MakeLocalGrid()),
        halo_(grid_, mpi_, is_wall_by_face<BC>),
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
        halo_.ExchangeLBM(fluid_);
        #endif
    }
    // LBM step.
    void Step() {
        LBMStep();
        ++time_step_;
    }

    // y_global is a domain-global y coordinate in [0, ny).
    double ux_analytical(int y_global) {
        double yi = static_cast<double>(y_global);
        return (kDeltaP / (2 * nu)) * (yi + 0.5) * (static_cast<double>(ny) - 0.5 - yi);
    }

    // y_global is a domain-global y coordinate in [0, ny). Ranks that don't
    // own the (y_global, nz/2) row contribute 0 to the Allreduce sum.
    double x_averaged_ux(int y_global) {
        const int z_local = nz/2 - grid_.offset_z;
        const int y_local = y_global - grid_.offset_y;
        double sum = 0.0;
        if (y_local >= 0 && y_local < grid_.local_ny &&
            z_local >= 0 && z_local < grid_.local_nz) {
            for (int x : std::views::iota(0, grid_.local_nx)) {
                sum += fluid_.ux[grid_.halo_idx(x, y_local, z_local)];
            }
        }
        double global_sum;
        MPIContext::SumDoubles(&sum, &global_sum);
        return global_sum / static_cast<double>(nx);
    }

    double mass() {
        double mass = 0.0;
        for (int z : std::views::iota(0, grid_.local_nz)) {
            for (int y : std::views::iota(0, grid_.local_ny)) {
                for (int x : std::views::iota(0, grid_.local_nx)) {
                    mass += fluid_.rho[grid_.halo_idx(x, y, z)];
                }
            }
        }

        double global_mass;
        MPIContext::SumDoubles(&mass, &global_mass);
        return global_mass;
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
    // Heap-allocated (not a static inline value) so construction happens in
    // SetUpTestSuite — after main()'s MPI_Init. A static inline member would be
    // dynamically initialised before main(), and its MPIContext ctor would call
    // MPI_Init a second time.
    static std::unique_ptr<PoiseuilleFlowBenchmark<PoiseuilleBC>> sim;
    // Runs ONCE before all tests in this fixture
    static void SetUpTestSuite() {
        sim = std::make_unique<PoiseuilleFlowBenchmark<PoiseuilleBC>>();
        for (int i = 0; i < 5000; i++)
            sim->Step();
    }
    static void TearDownTestSuite() {
        sim.reset();
    }
};

std::unique_ptr<PoiseuilleFlowBenchmark<PoiseuilleBC>> PoiseuilleFlow::sim;


TEST_F(PoiseuilleFlow, ParabolicProfile) {
    for (int y : std::views::iota(0, ny)) {
        double val1 = sim->ux_analytical(y);
        double val2 = sim->x_averaged_ux(y);
        EXPECT_NEAR(val1, val2, 1e-4);
    }
}

TEST_F(PoiseuilleFlow, ProfileSymmetry) {
    for (int y : std::views::iota(0, ny/2)) {
        double val1 = sim->x_averaged_ux(y);
        double val2 = sim->x_averaged_ux(ny-1-y);
        EXPECT_NEAR(val1, val2, 1e-8);
    }
}

TEST_F(PoiseuilleFlow, MassConservation) {
    double mass = sim->mass();
    double expected_mass = static_cast<double>(nx * ny * nz) * kDensity;
    // Worst-case bound: N cells * T steps * machine epsilon for linear accumulation
    // of floating-point roundoff through BGK collision and streaming.
    const double tol = static_cast<double>(nx * ny * nz) * 5000 * std::numeric_limits<double>::epsilon();
    EXPECT_NEAR(mass, expected_mass, tol);
}
