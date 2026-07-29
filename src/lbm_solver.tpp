#include <params.h>
#include "boundary.h"
#include "lattice_stencil.h"

#include <iostream>
#include <ranges>
#include <cmath>
#include "physics_helpers.h"
#include "local_grid.h"

using namespace Params;

template<typename BC>
LbmSolver<BC>::LbmSolver() {
    constexpr bool x_sensitive =
        std::is_same_v<typename BC::XLo::UBC, SpecularReflection> || is_moving_wall_v<typename BC::XLo::UBC> ||
        std::is_same_v<typename BC::XHi::UBC, SpecularReflection> || is_moving_wall_v<typename BC::XHi::UBC>;
    constexpr bool y_sensitive =
        std::is_same_v<typename BC::YLo::UBC, SpecularReflection> || is_moving_wall_v<typename BC::YLo::UBC> ||
        std::is_same_v<typename BC::YHi::UBC, SpecularReflection> || is_moving_wall_v<typename BC::YHi::UBC>;
    constexpr bool z_sensitive =
        std::is_same_v<typename BC::ZLo::UBC, SpecularReflection> || is_moving_wall_v<typename BC::ZLo::UBC> ||
        std::is_same_v<typename BC::ZHi::UBC, SpecularReflection> || is_moving_wall_v<typename BC::ZHi::UBC>;

    if constexpr ((x_sensitive && y_sensitive) || (x_sensitive && z_sensitive) || (y_sensitive && z_sensitive)) {
        std::cerr << "Warning [LbmSolver]: BCConfig has two or more intersecting SpecularReflection or\n"
                     "  MovingWall boundaries on different axes. Per-wall boundary handling is applied\n"
                     "  independently and does not compose correctly at shared edges and corners:\n"
                     "  mass conservation is violated at those locations. An explicit HandleCorner\n"
                     "  override is required for correct behavior.\n";
    }
}

template<typename BC>
void LbmSolver<BC>::Initialize(FluidFields& ff) const {

    const LocalGrid& g = ff.grid;

    #pragma omp parallel for default(shared) num_threads(kNumOMPThreads)
    for (int z : std::views::iota(0, g.local_nz)) {
        for (int y : std::views::iota(0, g.local_ny)) {
            for (int x : std::views::iota(0, g.local_nx)) {
                const int idxp = g.halo_idx(x, y, z);
                const double rhop = ff.rho[idxp];
                const double uxp  =  ff.ux[idxp];
                const double uyp  =  ff.uy[idxp];
                const double uzp  =  ff.uz[idxp];
                Vec3 up{uxp, uyp, uzp};
                const double u2 = up.Dot(up);
                for (int i : std::views::iota(0, Lattice::ndir)) {
                const int idxfp = g.halo_idx(x, y, z, i);
                    Vec3 e_i{
                        static_cast<double>(Lattice::ex[i]),
                        static_cast<double>(Lattice::ey[i]),
                        static_cast<double>(Lattice::ez[i])
                    };
                    ff.f[idxfp] = Feq({rhop, up}, up.Dot(e_i), u2, Lattice::w[i]);
                }
            }
        }
    }
}

template<typename BC>
void LbmSolver<BC>::LatticeBoltzmannStep(FluidFields& ff) const {
    // Single-pass LBM step: moments → f_eq + forcing → collision → stream + BC.
    //
    // Boundary conditions are applied per-point, inline with streaming.
    // Non-periodic walls use mid-point bounce-back (NoSlip / MovingWall) or
    // specular reflection (SpecularReflection). Periodic walls are no-ops:
    // UXoff/UYoff/UZoff wrap destinations in-domain, so HandleBoundaryPoint
    // is never reached for periodic axes.
    //
    // Bounce-back: when direction i at (x,y,z) streams out of domain, f★ is
    // reflected into the opposite direction at the source node:
    //   f_new[opp[i]] = f★  +  6·ρ·w[opp[i]]·(e[opp[i]]·u_wall)
    // where u_wall = (0,0,0) for NoSlip.
    //
    // Specular reflection (free-slip; reverses only the wall-normal component):
    //   Z-walls: f_new[specZ[i]] = f★   (reflect ez, keep ex, ey)
    //   Y-walls: f_new[specY[i]] = f★   (reflect ey, keep ex, ez)
    //   X-walls: f_new[specX[i]] = f★   (reflect ex, keep ey, ez)
    //
    // Corner handling: when direction i hits multiple walls (dx<0 AND dy<0, etc.),
    // each wall check fires independently; Z-wall writes take precedence at conflicts.
    //   Bounce-back corners: correct — all walls compute opp[i], same destination.
    //   SpecularReflection or MovingWall at intersecting walls: incorrect —
    //   independent single-axis reflections do not compose at shared edges/corners,
    //   violating mass conservation. An explicit HandleCorner override is needed (TODO).

    const LocalGrid& g = ff.grid;

    // Plan A: an axis with dims[axis]==1 (this rank owns the whole ring) is
    // exactly local_n == global_n -- no separate MPI-topology state needed.
    // Loop-invariant, so hoisted out of the point/direction loops once.
    const bool x_split = (g.local_nx != nx);
    const bool y_split = (g.local_ny != ny);
    const bool z_split = (g.local_nz != nz);

    #pragma omp parallel for default(shared) num_threads(kNumOMPThreads)
    for (int z : std::views::iota(0, g.local_nz)) {
        for (int y : std::views::iota(0, g.local_ny)) {
            for (int x : std::views::iota(0, g.local_nx)) {

                const int idxp = g.halo_idx(x, y, z);

                // ── Compute Moments ──────────────────────────────────────────
                // Accumulate ρ, ρu from f; velocity uses the half-step
                // force correction (Guo forcing scheme).
                Vec3 force{
                    ff.fx[idxp],
                    ff.fy[idxp],
                    ff.fz[idxp]
                };
                Moments m = ComputeMoments(
                    ff.f.data(),
                    {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)},
                    force,
                    Lattice::ex,
                    Lattice::ey,
                    Lattice::ez,
                    g
                );
                
                
                ff.rho[idxp] = m.rho;
                ff.ux[idxp]  = m.u.x;
                ff.uy[idxp]  = m.u.y;
                ff.uz[idxp]  = m.u.z;
                const double uF =  m.u.Dot(force);
                const double u2 = m.u.Dot(m.u);
                for (int i : std::views::iota(0, Lattice::ndir)) {
                    Vec3 e_i{
                        static_cast<double>(Lattice::ex[i]),
                        static_cast<double>(Lattice::ey[i]),
                        static_cast<double>(Lattice::ez[i])
                    };
                    auto [feq, forcing_term] = ComputeFeqAndForcing(m, u2, uF, force, e_i, Lattice::w[i]);

                    // ── Collision (BGK) ───────────────────────────────────────
                    const double f_star = PointwiseBGKCollide(ff.f[g.halo_idx(x, y, z, i)], feq, forcing_term);

                    // ── Stream + Apply Boundary Conditions ───────────────────
                    const int dx_global = StreamXoff<BC>(g.offset_x + x, Lattice::ex[i]);
                    const int dy_global = StreamYoff<BC>(g.offset_y + y, Lattice::ey[i]);
                    const int dz_global = StreamZoff<BC>(g.offset_z + z, Lattice::ez[i]);
                    const int dx = x + Lattice::ex[i];
                    const int dy = y + Lattice::ey[i];
                    const int dz = z + Lattice::ez[i];
                    // dx_global/dy_global/dz_global are already fully-resolved global
                    // coordinates (StreamXoff folded in offset_x, wall clamping, and
                    // periodic wrap), so bounds-check them directly against
                    // Params::nx/ny/nz -- no LocalGrid offset math needed here.
                    const bool x_crosses_wall = (dx_global < 0) || (dx_global >= nx);
                    const bool y_crosses_wall = (dy_global < 0) || (dy_global >= ny);
                    const bool z_crosses_wall = (dz_global < 0) || (dz_global >= nz);
                    // Effective write coordinate per axis. On an unsplit axis, local
                    // IS global (offset==0, local_n==n), so dx_global already *is*
                    // the correctly wrapped/interior local index -- use it directly
                    // and never touch the halo for that axis. On a split axis, keep
                    // the raw (possibly halo-range) local dx: a rank seam or a split
                    // periodic wrap both resolve via MPI exchange, never locally.
                    const int dx_eff = x_split ? dx : dx_global;
                    const int dy_eff = y_split ? dy : dy_global;
                    const int dz_eff = z_split ? dz : dz_global;
                    if (!x_crosses_wall && !y_crosses_wall && !z_crosses_wall) {
                        ff.f_new[g.halo_idx(dx_eff, dy_eff, dz_eff, i)] = f_star;
                    }
                    // If we indeed are at a physical boundary, then we need to handle
                    // different cases.
                    else {
                        double* f_new = ff.f_new.data();

                        if (dx_global < 0) {
                            HandleBoundaryPoint<typename BC::XLo>(x, y, z, i, Lattice::specX[i], f_star, m.rho, f_new, Lattice::ex, Lattice::ey, Lattice::ez, Lattice::w, Lattice::opp, g);
                        }
                        else if (dx_global >= nx) {
                            HandleBoundaryPoint<typename BC::XHi>(x, y, z, i, Lattice::specX[i], f_star, m.rho, f_new, Lattice::ex, Lattice::ey, Lattice::ez, Lattice::w, Lattice::opp, g);
                        }
                        if (dy_global < 0) {
                            HandleBoundaryPoint<typename BC::YLo>(x, y, z, i, Lattice::specY[i], f_star, m.rho, f_new, Lattice::ex, Lattice::ey, Lattice::ez, Lattice::w, Lattice::opp, g);
                        }
                        else if (dy_global >= ny) {
                            HandleBoundaryPoint<typename BC::YHi>(x, y, z, i, Lattice::specY[i], f_star, m.rho, f_new, Lattice::ex, Lattice::ey, Lattice::ez, Lattice::w, Lattice::opp, g);
                        }
                        if (dz_global < 0) {
                            HandleBoundaryPoint<typename BC::ZLo>(x, y, z, i, Lattice::specZ[i], f_star, m.rho, f_new, Lattice::ex, Lattice::ey, Lattice::ez, Lattice::w, Lattice::opp, g);
                        }
                        else if (dz_global >= nz) {
                            HandleBoundaryPoint<typename BC::ZHi>(x, y, z, i, Lattice::specZ[i], f_star, m.rho, f_new, Lattice::ex, Lattice::ey, Lattice::ez, Lattice::w, Lattice::opp, g);
                        }
                    }
                }
            }
        }
    }

    ff.SwapFandFnew();
}
