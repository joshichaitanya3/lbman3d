#include <gtest/gtest.h>
#include <limits>
#include "params.h"
#include "sim_config.h"
#include "offsets.h"
#include "fluid_fields.h"
#include "qtensor_fields.h"
#include "device_fields.h"
#include "device_solver.h"
#include "qtensor_solver.h"
#include "analysis_fields.h"
#include "local_grid.h"

#include <vector>
#include <algorithm>
#include <math.h>

using namespace Params;


template<typename BC>
class ZeroActivitySolver : public QTensorSolver<BC> {
    public:
        using QTensorSolver<BC>::QTensorSolver;   // inherit ctor
        void SetActiveStressAndComputeBodyForce(FluidFields&, const QTensorFields&) const override;
        void Initialize(QTensorFields&) const override;
};

template<typename BC>
void ZeroActivitySolver<BC>::SetActiveStressAndComputeBodyForce(FluidFields&, const QTensorFields&) const {
    return;
}

template<typename BC>
void ZeroActivitySolver<BC>::Initialize(QTensorFields& qf) const {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> noise_dist(-NOISE, NOISE);
    for (int z : std::views::iota(0, nz)) {
        for (int y : std::views::iota(0, ny)) {
            for (int x : std::views::iota(0, nx)) {
                qf.qxx[idx(x, y, z)] = 0.33 + noise_dist(gen);
                qf.qxy[idx(x, y, z)] = noise_dist(gen);
                qf.qxz[idx(x, y, z)] = noise_dist(gen);
                qf.qyy[idx(x, y, z)] = -0.15 + noise_dist(gen);
                qf.qyz[idx(x, y, z)] = noise_dist(gen);
            }
        }
    }
}

template<typename BC>
class QTensorRelaxationBenchmark {
    LocalGrid         grid_;
    FluidFields       fluid_;
    QTensorFields     qtensor_;
    DeviceFields      d_fields_;
    std::unique_ptr<QTensorSolver<BC>> qtensor_solver_;
    AnalysisFields    af_;
    DeviceSolver<BC>  d_solver_;
    int               time_step_ = 0;

    void Initialize() {
        qtensor_solver_->Initialize(qtensor_);
        #ifdef SIM_WITH_CUDA
        d_fields_.Initialize(fluid_, qtensor_);
        d_solver_.Initialize(d_fields_);
        #endif // SIM_WITH_CUDA
    }
public:
    // Default: constant-alpha active nematic.
    // Supply a QTensorSolver subclass to override the activity model.
    public:
    // Default: constant-alpha active nematic.
    // Supply a QTensorSolver subclass to override the activity model.
    explicit QTensorRelaxationBenchmark(std::unique_ptr<QTensorSolver<BC>> solver = nullptr) :
        grid_(LocalGrid::SingleRank()),
        fluid_(grid_),
        qtensor_(grid_),
        d_fields_(grid_),
        qtensor_solver_(solver ? std::move(solver)
                                 : std::make_unique<QTensorSolver<BC>>())
    {
        Initialize();
    }


    void QTensorStep() {
        #ifdef SIM_WITH_CUDA
        d_solver_.QTensorStep(d_fields_);
        #else
        qtensor_solver_->Step(qtensor_, fluid_);
        #endif
    }

    void Step() {
        QTensorStep();
        ++time_step_;
    }

    double NematicEnergy() {
        return TotalNematicFreeEnergy<BC>(qtensor_);
    }

    double MeanOrder() {
        QtensorToOrderDirector(qtensor_, af_);
        return std::accumulate(af_.order_.begin(), af_.order_.end(),0.0) / static_cast<double>(af_.order_.size());
    }
    double SpatialUniformity() {
        QtensorToOrderDirector(qtensor_, af_);

        double mean_order = MeanOrder();
        std::vector<double> diff_sq;
        for (int i : std::views::iota(0, nx*ny*nz)) {
            double diff = af_.order_[i] - mean_order;
            diff_sq.push_back(diff*diff);
        }
        double variance = std::accumulate(diff_sq.begin(), diff_sq.end(),0.0) / static_cast<double>(diff_sq.size());

        double stddev = std::sqrt(variance);
        return stddev;
    }
    int GetTimeStep() const { return time_step_; }
};

template struct QTensorRelaxationBenchmark<FullyPeriodicConfig>;
template struct ZeroActivitySolver<FullyPeriodicConfig>;

class QTensorRelaxation : public ::testing::Test {
    protected:
    static inline QTensorRelaxationBenchmark<FullyPeriodicConfig> sim{std::make_unique<ZeroActivitySolver<FullyPeriodicConfig>>()};
    // Runs ONCE before all tests in this fixture
    static inline std::vector<double> nematic_energies;
    static inline int np = 20;
    static void SetUpTestSuite() {
        for (int i = 0; i < np*100; i++) {
            if (i % 100 == 0) {
                nematic_energies.push_back(sim.NematicEnergy());
            }
            sim.Step();
        }
    }

};


TEST_F(QTensorRelaxation, MonotoneFreeEnergyDecrease) {
    for (int i : std::views::iota(0, np-1)) {
        double val1 = nematic_energies[i];
        double val2 = nematic_energies[i+1];
        EXPECT_TRUE(val2 < (val1 + 1e-10));
    }
}

TEST_F(QTensorRelaxation, MeanOrder) {

    double analytical_expectation = -B/(3.0 * C);
    EXPECT_NEAR(sim.MeanOrder(), analytical_expectation, 1e-6);
}

TEST_F(QTensorRelaxation, SpatialUniformity) {
    
    EXPECT_LT(sim.SpatialUniformity(), 1e-2);
}

TEST_F(QTensorRelaxation, Convergence) {
    
    EXPECT_LT((nematic_energies[np-2] - nematic_energies[np-1]), 1e-6);
}
