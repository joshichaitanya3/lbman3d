#include "params.h"
#include "boundary.h"
#include "lattice_stencil.h"

#include <iostream>
#include <ranges>
#include <cmath>
#include "physics_helpers.h"

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
    for (int z : std::views::iota(0, nz)) {
        for (int y : std::views::iota(0, ny)) {
            for (int x : std::views::iota(0, nx)) {
                const double rhop = ff.rho[idx(x, y, z)];
                const double uxp  = ff.ux[idx(x, y, z)];
                const double uyp  = ff.uy[idx(x, y, z)];
                const double uzp  = ff.uz[idx(x, y, z)];
                Vec3 up{uxp, uyp, uzp};
                const double u2 = up.Dot(up);
                for (int i : std::views::iota(0, Lattice::ndir)) {

                    Vec3 e_i{
                        static_cast<double>(Lattice::ex[i]),
                        static_cast<double>(Lattice::ey[i]),
                        static_cast<double>(Lattice::ez[i])
                    };
                    ff.f[idx(x, y, z, i)] = Feq({rhop, up}, e_i, u2, Lattice::w[i]);
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

    #pragma omp parallel for default(shared) num_threads(numprocs)
    for (int z : std::views::iota(0, nz)) {
        for (int y : std::views::iota(0, ny)) {
            for (int x : std::views::iota(0, nx)) {

                // ── Compute Moments ──────────────────────────────────────────
                // Accumulate ρ, ρu from f; velocity uses the half-step
                // force correction (Guo forcing scheme).
                Vec3 force{
                    ff.fx[idx(x, y, z)],
                    ff.fy[idx(x, y, z)],
                    ff.fz[idx(x, y, z)]
                };
                Moments m = ComputeMoments(
                    ff.f.data(),
                    {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)},
                    force,
                    Lattice::ex,
                    Lattice::ey,
                    Lattice::ez
                );
                
                
                ff.rho[idx(x, y, z)] = m.rho;
                ff.ux[idx(x, y, z)]  = m.u.x;
                ff.uy[idx(x, y, z)]  = m.u.y;
                ff.uz[idx(x, y, z)]  = m.u.z;
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
                    const double f_star = PointwiseBGKCollide(ff.f[idx(x, y, z, i)], feq, forcing_term);

                    // ── Stream + Apply Boundary Conditions ───────────────────
                    const int dx = StreamXoff<BC>(x, Lattice::ex[i]);
                    const int dy = StreamYoff<BC>(y, Lattice::ey[i]);
                    const int dz = StreamZoff<BC>(z, Lattice::ez[i]);
                    if (InDomain(dx, dy, dz)) {
                        ff.f_new[idx(dx, dy, dz, i)] = f_star;
                    }
                    else {
                        double* f_new = ff.f_new.data();

                        if (dx < 0) {
                            HandleBoundaryPoint<typename BC::XLo>(x, y, z, i, Lattice::specX[i], f_star, m.rho, f_new, Lattice::ex, Lattice::ey, Lattice::ez, Lattice::w, Lattice::opp);
                        }
                        else if (dx >= nx) {
                            HandleBoundaryPoint<typename BC::XHi>(x, y, z, i, Lattice::specX[i], f_star, m.rho, f_new, Lattice::ex, Lattice::ey, Lattice::ez, Lattice::w, Lattice::opp);
                        }
                        if (dy < 0) {
                            HandleBoundaryPoint<typename BC::YLo>(x, y, z, i, Lattice::specY[i], f_star, m.rho, f_new, Lattice::ex, Lattice::ey, Lattice::ez, Lattice::w, Lattice::opp);
                        }
                        else if (dy >= ny) {
                            HandleBoundaryPoint<typename BC::YHi>(x, y, z, i, Lattice::specY[i], f_star, m.rho, f_new, Lattice::ex, Lattice::ey, Lattice::ez, Lattice::w, Lattice::opp);
                        }
                        if (dz < 0) {
                            HandleBoundaryPoint<typename BC::ZLo>(x, y, z, i, Lattice::specZ[i], f_star, m.rho, f_new, Lattice::ex, Lattice::ey, Lattice::ez, Lattice::w, Lattice::opp);
                        }
                        else if (dz >= nz) {
                            HandleBoundaryPoint<typename BC::ZHi>(x, y, z, i, Lattice::specZ[i], f_star, m.rho, f_new, Lattice::ex, Lattice::ey, Lattice::ez, Lattice::w, Lattice::opp);
                        }
                    }
                }
            }
        }
    }

    ff.SwapFandFnew();
}
