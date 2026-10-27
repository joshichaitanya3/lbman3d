// tests/mpi/test_fd_stencils_mpi.cc
//
// FD stencil helpers across MPI seams. The boundary-aware helpers take
// rank-local coordinates, so their wrap/clamp/wall decisions must be made on
// the matching global coordinate — otherwise a split axis reads the wrong
// cell at a periodic seam, sees a wall at an interior seam, and misses the
// physical wall on the high-side rank.
//
// Every rank fills Q, u, Sigma and tau on its owned cells from smooth analytic
// functions of GLOBAL coordinates, runs the two Q-side halo exchanges, then
// evaluates each helper at every owned cell. The reference is the same helper
// on a LocalGrid::SingleRank() copy of the full field (cheap at 8³), which is
// unsplit by construction. Same inputs, same arithmetic → EXPECT_DOUBLE_EQ.
//
// tests/params/unit is a cube (8×8×8), so np=2 splits x only and np=8 is a
// true 2×2×2 split.

#include <mpi.h>
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <numbers>
#include <vector>
#include "params.h"
#include "boundary.h"
#include "boundary_handler.h"
#include "analysis_fields.h"
#include "fluid_fields.h"
#include "qtensor_fields.h"
#include "mpi/mpi_context.h"
#include "mpi/halo_exchange_qtensor.h"

namespace {

// Walled x with distinct strong-anchoring targets on the two faces, so
// QGhost's Dirichlet branch is exercised (not just the Neumann clamp) and a
// lo/hi mix-up would show.
struct AnchoredXWallConfig {
    using XLo = WallSpec<Anchoring<0.5, 0.3, 0.7>, NoSlip>;
    using XHi = WallSpec<Anchoring<0.4, 1.1, 0.2>, NoSlip>;
    using YLo = WallSpec<Periodic, Periodic>;
    using YHi = WallSpec<Periodic, Periodic>;
    using ZLo = WallSpec<Periodic, Periodic>;
    using ZHi = WallSpec<Periodic, Periodic>;
};

// Field k at global (gx, gy, gz). Periodic in all three axes, with a
// per-field phase so no two fields (or a field and its mirror) coincide.
double Analytic(int k, int gx, int gy, int gz) {
    constexpr double two_pi = 2.0 * std::numbers::pi;
    const double a = 0.37 * k + 0.11;
    return std::sin(two_pi * gx / Params::nx + a)
         * std::cos(two_pi * gy / Params::ny + 1.3 * a)
         + 0.5 * std::sin(two_pi * gz / Params::nz + 2.1 * a)
         + 0.05 * k;
}

// All 16 scalar fields the helpers under test read, in a fixed order.
std::array<std::vector<double>*, 16> AllFields(QTensorFields& qf, FluidFields& ff) {
    return {&qf.qxx, &qf.qxy, &qf.qxz, &qf.qyy, &qf.qyz,
            &ff.ux, &ff.uy, &ff.uz,
            &qf.Sigma_xx, &qf.Sigma_xy, &qf.Sigma_xz, &qf.Sigma_yy, &qf.Sigma_yz,
            &qf.Tau_xy, &qf.Tau_xz, &qf.Tau_yz};
}

void FillOwned(QTensorFields& qf, FluidFields& ff, const LocalGrid& g) {
    const auto fields = AllFields(qf, ff);
    for (int k = 0; k < static_cast<int>(fields.size()); ++k)
        for (int z = 0; z < g.local_nz; ++z)
            for (int y = 0; y < g.local_ny; ++y)
                for (int x = 0; x < g.local_nx; ++x)
                    (*fields[k])[g.halo_idx(x, y, z)] =
                        Analytic(k, g.offset_x + x, g.offset_y + y, g.offset_z + z);
}

void ExpectDerivsEq(const QDerivs& a, const QDerivs& b, const char* comp) {
    EXPECT_DOUBLE_EQ(a.dx,  b.dx)  << comp;
    EXPECT_DOUBLE_EQ(a.dy,  b.dy)  << comp;
    EXPECT_DOUBLE_EQ(a.dz,  b.dz)  << comp;
    EXPECT_DOUBLE_EQ(a.lap, b.lap) << comp;
    EXPECT_DOUBLE_EQ(a.d2x, b.d2x) << comp;
    EXPECT_DOUBLE_EQ(a.d2y, b.d2y) << comp;
    EXPECT_DOUBLE_EQ(a.d2z, b.d2z) << comp;
}

template<typename BC>
void CheckStencilsAgainstSingleRank() {
    MPIContext mpi(periodicity_by_axis<BC>);
    const LocalGrid g = mpi.MakeLocalGrid();
    HaloExchangeQTensor halo(g, mpi);

    QTensorFields qf(g);
    FluidFields   ff(g);
    FillOwned(qf, ff, g);
    halo.ExchangeQTensor(qf, ff);
    halo.ExchangePassiveStresses(qf);

    const LocalGrid gr = LocalGrid::SingleRank();
    QTensorFields qf_ref(gr);
    FluidFields   ff_ref(gr);
    FillOwned(qf_ref, ff_ref, gr);

    for (int z = 0; z < g.local_nz; ++z) {
        for (int y = 0; y < g.local_ny; ++y) {
            for (int x = 0; x < g.local_nx; ++x) {
                const int gx = g.offset_x + x, gy = g.offset_y + y, gz = g.offset_z + z;
                SCOPED_TRACE(::testing::Message()
                    << "rank " << mpi.world_rank << " local (" << x << "," << y << "," << z
                    << ") global (" << gx << "," << gy << "," << gz << ")");

                ExpectDerivsEq(QGradientAndLaplacian<QComp::XX, BC>(qf.qxx.data(), x, y, z, g),
                               QGradientAndLaplacian<QComp::XX, BC>(qf_ref.qxx.data(), gx, gy, gz, gr), "Qxx");
                ExpectDerivsEq(QGradientAndLaplacian<QComp::XY, BC>(qf.qxy.data(), x, y, z, g),
                               QGradientAndLaplacian<QComp::XY, BC>(qf_ref.qxy.data(), gx, gy, gz, gr), "Qxy");
                ExpectDerivsEq(QGradientAndLaplacian<QComp::XZ, BC>(qf.qxz.data(), x, y, z, g),
                               QGradientAndLaplacian<QComp::XZ, BC>(qf_ref.qxz.data(), gx, gy, gz, gr), "Qxz");
                ExpectDerivsEq(QGradientAndLaplacian<QComp::YY, BC>(qf.qyy.data(), x, y, z, g),
                               QGradientAndLaplacian<QComp::YY, BC>(qf_ref.qyy.data(), gx, gy, gz, gr), "Qyy");
                ExpectDerivsEq(QGradientAndLaplacian<QComp::YZ, BC>(qf.qyz.data(), x, y, z, g),
                               QGradientAndLaplacian<QComp::YZ, BC>(qf_ref.qyz.data(), gx, gy, gz, gr), "Qyz");

                const GradTensor v = VelocityGradientTensor<BC>(
                    ff.ux.data(), ff.uy.data(), ff.uz.data(), x, y, z, g);
                const GradTensor v_ref = VelocityGradientTensor<BC>(
                    ff_ref.ux.data(), ff_ref.uy.data(), ff_ref.uz.data(), gx, gy, gz, gr);
                EXPECT_DOUBLE_EQ(v.ux_x, v_ref.ux_x);
                EXPECT_DOUBLE_EQ(v.ux_y, v_ref.ux_y);
                EXPECT_DOUBLE_EQ(v.ux_z, v_ref.ux_z);
                EXPECT_DOUBLE_EQ(v.uy_x, v_ref.uy_x);
                EXPECT_DOUBLE_EQ(v.uy_y, v_ref.uy_y);
                EXPECT_DOUBLE_EQ(v.uy_z, v_ref.uy_z);
                EXPECT_DOUBLE_EQ(v.uz_x, v_ref.uz_x);
                EXPECT_DOUBLE_EQ(v.uz_y, v_ref.uz_y);

                const Vec3 f = PassiveStressDivergence<BC>(
                    qf.Sigma_xx.data(), qf.Sigma_xy.data(), qf.Sigma_xz.data(),
                    qf.Sigma_yy.data(), qf.Sigma_yz.data(),
                    qf.Tau_xy.data(), qf.Tau_xz.data(), qf.Tau_yz.data(), x, y, z, g);
                const Vec3 f_ref = PassiveStressDivergence<BC>(
                    qf_ref.Sigma_xx.data(), qf_ref.Sigma_xy.data(), qf_ref.Sigma_xz.data(),
                    qf_ref.Sigma_yy.data(), qf_ref.Sigma_yz.data(),
                    qf_ref.Tau_xy.data(), qf_ref.Tau_xz.data(), qf_ref.Tau_yz.data(), gx, gy, gz, gr);
                EXPECT_DOUBLE_EQ(f.x, f_ref.x);
                EXPECT_DOUBLE_EQ(f.y, f_ref.y);
                EXPECT_DOUBLE_EQ(f.z, f_ref.z);
            }
        }
    }

    // Domain sum: same per-cell terms, different summation order.
    double local_energy = TotalNematicFreeEnergy<BC>(qf);
    double global_energy = 0.0;
    MPI_Allreduce(&local_energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM, mpi.cart_comm);
    const double ref_energy = TotalNematicFreeEnergy<BC>(qf_ref);
    EXPECT_NEAR(global_energy, ref_energy, 1e-12 * std::abs(ref_energy));
}

} // namespace

TEST(FdStencilsMpi, FullyPeriodicMatchesSingleRank) {
    CheckStencilsAgainstSingleRank<FullyPeriodicConfig>();
}

TEST(FdStencilsMpi, AnchoredXWallMatchesSingleRank) {
    CheckStencilsAgainstSingleRank<AnchoredXWallConfig>();
}
