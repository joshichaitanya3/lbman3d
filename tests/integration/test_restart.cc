#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <random>
#include <ranges>
#include <string>
#include <vector>

#include <params.h>
#include <sim_config.h>
#include "analysis_fields.h"
#include "boundary.h"
#include "fluid_fields.h"
#include "lbm_solver.h"
#include "local_grid.h"
#include "mpi/mpi_context.h"
#include "qtensor_fields.h"
#include "qtensor_solver.h"
#include "sim_config_attrs.h"
#include "vtkhdf_reader.h"
#include "vtkhdf_writer.h"

/*
 * Restart round-trip: exercise the code path ActiveNematicSim::LoadRestart uses
 * (equilibrium f from moments, uniaxial Q from order + director) and verify
 * two properties:
 *
 *   1. The reconstruction at t = kSnapshotStep is faithful:
 *        - rho, ux, uy, uz round-trip through VTKHDF bit-exactly (up to
 *          HDF5 double precision).
 *        - Q round-trips through (order, director) to well within the
 *          biaxial-projection error, because the seeded IC is nearly
 *          uniaxial and Beris-Edwards keeps it that way over the short
 *          pre-snapshot run.
 *
 *   2. Continuing the reconstructed state under identical dynamics stays
 *      well-behaved: mass conserved, velocities bounded, no NaN.
 *
 * NOTE we do NOT require the reference and restart trajectories to converge
 * pointwise after kPostRestartSteps. The equilibrium reconstruction drops
 * f^{neq}, which is a legitimate small perturbation. For this passive system
 * (ALPHA=0, MU=0) the base flow is itself relaxing to zero, so the
 * perturbation is O(base state) and the trajectories separate — that's a
 * property of the setup, not a bug. For an active/driven scenario the same
 * O(TAUF) transient would appear as a fast return to steady state, which is
 * the physically relevant use case for restart. The pointwise faithfulness
 * of the *reconstruction itself* is the right invariant to guard here.
 */

using namespace Params;

namespace {

// Same seed / IC as test_coupled_backflow.cc's DistortedICSolver.
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

using BC = FullyPeriodicConfig;

struct MiniSim {
    LocalGrid grid{LocalGrid::SingleRank()};
    FluidFields fluid{grid};
    QTensorFields qtensor{grid};
    LbmSolver<BC> lbm;
    std::unique_ptr<QTensorSolver<BC>> solver =
        std::make_unique<DistortedICSolver<BC>>();

    void Initialize() {
        lbm.Initialize(fluid);
        solver->Initialize(qtensor);
    }
    void Step() {
        solver->Step(qtensor, fluid);
        lbm.LatticeBoltzmannStep(fluid);
    }
    void Advance(int n) { for (int i = 0; i < n; ++i) Step(); }
};

std::filesystem::path RestartTempFile() {
    return std::filesystem::temp_directory_path() /
           (std::string("lbm_restart_") + std::to_string(::getpid()) + ".vtkhdf");
}

// Mirrors SimIO::ExportVTKHDF's field selection at the level LoadRestart reads.
// Kept off the SimIO link path per tests/CLAUDE.md.
void WriteRestartSnapshot(const std::filesystem::path& path,
                          const FluidFields& fluid,
                          const QTensorFields& qtensor,
                          const MPIContext& ctx) {
    AnalysisFields af(fluid.grid);
    QtensorToOrderDirector(qtensor, af);

    ImageDataWriter w(path.string(), ctx);
    SimConfigAttr::StampSimConfigAttributes<BC>(w.root());
    w.WriteScalarField("rho",   fluid.rho.data(), fluid.grid);
    w.WriteScalarField("ux",    fluid.ux.data(),  fluid.grid);
    w.WriteScalarField("uy",    fluid.uy.data(),  fluid.grid);
    w.WriteScalarField("uz",    fluid.uz.data(),  fluid.grid);
    w.WriteScalarField("order", af.order_.data(), fluid.grid);
    w.WriteVectorField("director", af.director_.data(), fluid.grid, 3);
}

// Mirrors ActiveNematicSim::LoadRestart's reconstruction.
void LoadRestartSnapshot(const std::filesystem::path& path,
                         MiniSim& sim,
                         const MPIContext& ctx) {
    ImageDataReader r(path.string(), ctx);
    r.ReadScalarField("rho", sim.fluid.rho.data(), sim.grid);
    r.ReadScalarField("ux",  sim.fluid.ux.data(),  sim.grid);
    r.ReadScalarField("uy",  sim.fluid.uy.data(),  sim.grid);
    r.ReadScalarField("uz",  sim.fluid.uz.data(),  sim.grid);
    sim.lbm.Initialize(sim.fluid);

    AnalysisFields af(sim.grid);
    r.ReadScalarField("order",    af.order_.data(),    sim.grid);
    r.ReadVectorField("director", af.director_.data(), sim.grid, 3);
    OrderDirectorToQtensor(af, sim.qtensor);
}

constexpr int kSnapshotStep     = 100;
constexpr int kPostRestartSteps = 100;

// rho, ux, uy, uz round-trip HDF5-double-precision.
constexpr double kMomentAbsTol = 1e-12;
// Q is compared against the reference *after the same uniaxial projection*,
// so this tolerance measures only what the HDF5 round-trip and the second
// application of the projection do — not the physical projection loss,
// which is up to O(NOISE) and legitimate. In practice both sides differ only
// by float-arithmetic reassociation in the QtensorToOrderDirector /
// OrderDirectorToQtensor pipeline, well below 1e-10.
constexpr double kQAbsTol = 1e-10;

} // namespace

template class DistortedICSolver<BC>;

class RestartRoundtrip : public ::testing::Test {
protected:
    static inline MPIContext ctx{periodicity_by_axis<BC>};
    static inline std::filesystem::path snapshot_path;

    static inline std::unique_ptr<MiniSim> ref_sim;      // continuous 0 → kSnapshotStep + kPostRestartSteps
    static inline std::unique_ptr<MiniSim> restart_sim;  // load at kSnapshotStep, advance kPostRestartSteps

    // Snapshots of both sims right after reconstruction (t = kSnapshotStep):
    // used to verify the reconstruction itself, before dynamics diverge them.
    static inline std::vector<double> ref_rho_snap, ref_ux_snap, ref_uy_snap, ref_uz_snap;
    static inline std::vector<double> ref_qxx_snap, ref_qxy_snap, ref_qxz_snap, ref_qyy_snap, ref_qyz_snap;
    static inline std::vector<double> load_rho_snap, load_ux_snap, load_uy_snap, load_uz_snap;
    static inline std::vector<double> load_qxx_snap, load_qxy_snap, load_qxz_snap, load_qyy_snap, load_qyz_snap;

    static void SetUpTestSuite() {
        snapshot_path = RestartTempFile();

        // Reference run — save state at kSnapshotStep before continuing.
        ref_sim = std::make_unique<MiniSim>();
        ref_sim->Initialize();
        ref_sim->Advance(kSnapshotStep);
        WriteRestartSnapshot(snapshot_path, ref_sim->fluid, ref_sim->qtensor, ctx);

        // Snapshot the reference state for comparison against the restart's
        // in-memory state. For Q we apply the *same* uniaxial projection the
        // exporter did, so the two Q arrays only differ by what HDF5 and the
        // second projection do — not by the physical biaxial-info loss, which
        // would obscure any real plumbing bug.
        QTensorFields ref_qtensor_projected(ref_sim->grid);
        {
            AnalysisFields tmp(ref_sim->grid);
            QtensorToOrderDirector(ref_sim->qtensor, tmp);
            OrderDirectorToQtensor(tmp, ref_qtensor_projected);
        }
        ref_rho_snap = ref_sim->fluid.rho;
        ref_ux_snap  = ref_sim->fluid.ux;
        ref_uy_snap  = ref_sim->fluid.uy;
        ref_uz_snap  = ref_sim->fluid.uz;
        ref_qxx_snap = ref_qtensor_projected.qxx;
        ref_qxy_snap = ref_qtensor_projected.qxy;
        ref_qxz_snap = ref_qtensor_projected.qxz;
        ref_qyy_snap = ref_qtensor_projected.qyy;
        ref_qyz_snap = ref_qtensor_projected.qyz;

        ref_sim->Advance(kPostRestartSteps);

        // Fresh sim; init overwritten by the load.
        restart_sim = std::make_unique<MiniSim>();
        restart_sim->Initialize();
        LoadRestartSnapshot(snapshot_path, *restart_sim, ctx);
        load_rho_snap = restart_sim->fluid.rho;
        load_ux_snap  = restart_sim->fluid.ux;
        load_uy_snap  = restart_sim->fluid.uy;
        load_uz_snap  = restart_sim->fluid.uz;
        load_qxx_snap = restart_sim->qtensor.qxx;
        load_qxy_snap = restart_sim->qtensor.qxy;
        load_qxz_snap = restart_sim->qtensor.qxz;
        load_qyy_snap = restart_sim->qtensor.qyy;
        load_qyz_snap = restart_sim->qtensor.qyz;
        restart_sim->Advance(kPostRestartSteps);
    }

    static void TearDownTestSuite() {
        std::error_code ec;
        std::filesystem::remove(snapshot_path, ec);
        ref_sim.reset();
        restart_sim.reset();
    }
};

// rho / ux / uy / uz are written and read as doubles; the reconstruction must
// round-trip them essentially bit-exactly — an off-by-one hyperslab or a wrong
// dtype would break this.
TEST_F(RestartRoundtrip, MomentsRoundTripAtSnapshot) {
    const auto& g = ref_sim->grid;
    for (int z : std::views::iota(0, nz)) {
        for (int y : std::views::iota(0, ny)) {
            for (int x : std::views::iota(0, nx)) {
                const int i = g.halo_idx(x, y, z);
                EXPECT_NEAR(load_rho_snap[i], ref_rho_snap[i], kMomentAbsTol) << "rho at " << x << "," << y << "," << z;
                EXPECT_NEAR(load_ux_snap[i],  ref_ux_snap[i],  kMomentAbsTol) << "ux at "  << x << "," << y << "," << z;
                EXPECT_NEAR(load_uy_snap[i],  ref_uy_snap[i],  kMomentAbsTol) << "uy at "  << x << "," << y << "," << z;
                EXPECT_NEAR(load_uz_snap[i],  ref_uz_snap[i],  kMomentAbsTol) << "uz at "  << x << "," << y << "," << z;
            }
        }
    }
}

// Q → (S, n̂) → uniaxial inverse. The seed keeps Q close to uniaxial, so a
// correctly wired reconstruction agrees to within the biaxial projection
// error. The main things this catches are: mis-scaled uniaxial factor,
// wrong director layout (dirIdx vs halo_idx), and off-by-one hyperslab in
// the vector-field read.
TEST_F(RestartRoundtrip, QReconstructionMatchesAtSnapshot) {
    const auto& g = ref_sim->grid;
    for (int z : std::views::iota(0, nz)) {
        for (int y : std::views::iota(0, ny)) {
            for (int x : std::views::iota(0, nx)) {
                const int i = g.halo_idx(x, y, z);
                EXPECT_NEAR(load_qxx_snap[i], ref_qxx_snap[i], kQAbsTol) << "qxx at " << x << "," << y << "," << z;
                EXPECT_NEAR(load_qxy_snap[i], ref_qxy_snap[i], kQAbsTol) << "qxy at " << x << "," << y << "," << z;
                EXPECT_NEAR(load_qxz_snap[i], ref_qxz_snap[i], kQAbsTol) << "qxz at " << x << "," << y << "," << z;
                EXPECT_NEAR(load_qyy_snap[i], ref_qyy_snap[i], kQAbsTol) << "qyy at " << x << "," << y << "," << z;
                EXPECT_NEAR(load_qyz_snap[i], ref_qyz_snap[i], kQAbsTol) << "qyz at " << x << "," << y << "," << z;
            }
        }
    }
}

// Continuing the reconstructed state must not blow up or drain mass — those
// are the failure modes if halo populations were reused stale or the
// reconstruction wrote into the wrong slots.
TEST_F(RestartRoundtrip, ContinuationStaysWellBehaved) {
    const auto& g = restart_sim->grid;
    double mass_after = 0.0;
    double max_speed_sq = 0.0;
    for (int z : std::views::iota(0, nz)) {
        for (int y : std::views::iota(0, ny)) {
            for (int x : std::views::iota(0, nx)) {
                const int i = g.halo_idx(x, y, z);
                mass_after += restart_sim->fluid.rho[i];
                const double u2 = restart_sim->fluid.ux[i]*restart_sim->fluid.ux[i]
                                + restart_sim->fluid.uy[i]*restart_sim->fluid.uy[i]
                                + restart_sim->fluid.uz[i]*restart_sim->fluid.uz[i];
                if (u2 > max_speed_sq) max_speed_sq = u2;
            }
        }
    }
    const double expected_mass = static_cast<double>(nx) * ny * nz * kDensity;
    EXPECT_NEAR(mass_after / expected_mass, 1.0, 1e-10)
        << "mass drifted during post-restart continuation";
    EXPECT_LT(std::sqrt(max_speed_sq), 0.1)
        << "post-restart flow left the low-Mach regime — likely a divergence";
}
