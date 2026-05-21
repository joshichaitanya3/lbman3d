#include "params.h"
#include "boundary.h"

#include <ranges>
#include <cmath>

using namespace Params;

template<typename BC>
LbmSolver<BC>::LbmSolver(Grid<BC> grid) : grid_(std::move(grid)) {}

template<typename BC>
bool LbmSolver<BC>::InDomain(int x, int y, int z) const {
    return (x >= 0) && (x < nx) && (y >= 0) && (y < ny) && (z >= 0) && (z < nz);
}

template<typename BC>
double LbmSolver<BC>::Feq(double rhop, double uxp, double uyp, double uzp, int i) const {
    const double u2 = uxp * uxp + uyp * uyp + uzp * uzp;	//Velocity squared
    const double u_dot_e = uxp * FluidFields::ex[i] + uyp * FluidFields::ey[i] + uzp * FluidFields::ez[i];
    return (FluidFields::w[i] * rhop * (1.0 + kCs2Inv * u_dot_e + khalfCs4Inv * u_dot_e * u_dot_e - khalfCs2Inv * u2));
        
}

template<typename BC>
void LbmSolver<BC>::Initialize(FluidFields& ff) const {
    for (int z : std::views::iota(0, nz)) {
        for (int y : std::views::iota(0, ny)) {
            for (int x : std::views::iota(0, nx)) {
                const double rhop = ff.rho[z, y, x];
                const double uxp  = ff.ux[z, y, x];
                const double uyp  = ff.uy[z, y, x];
                const double uzp  = ff.uz[z, y, x];
                for (int i : std::views::iota(0, ndir))
                    ff.f[z, y, x, i] = Feq(rhop, uxp, uyp, uzp, i);
            }
        }
    }
}

template<typename BC>
void LbmSolver<BC>::ResetFeq(FluidFields& ff) const {
    #pragma omp parallel for default(shared) num_threads(numprocs)
    for (int z : std::views::iota(0, nz)) {
        for (int y : std::views::iota(0, ny)) {
            for (int x : std::views::iota(0, nx)) {
                const double rhop = ff.rho[z, y, x];
                const double uxp  = ff.ux[z, y, x];
                const double uyp  = ff.uy[z, y, x];
                const double uzp  = ff.uz[z, y, x];
                for (int i : std::views::iota(0, ndir))
                    ff.f_eq[z, y, x, i] = Feq(rhop, uxp, uyp, uzp, i);
            }
        }
    }
}

template<typename BC>
void LbmSolver<BC>::ComputeForcingTerms(FluidFields& ff) const {
    #pragma omp parallel for default(shared) num_threads(numprocs) schedule(static)
    for (int z : std::views::iota(0, nz)) {
        for (int y : std::views::iota(0, ny)) {
            for (int x : std::views::iota(0, nx)) {
                const double forceX = ff.fx[z, y, x];
                const double forceY = ff.fy[z, y, x];
                const double forceZ = ff.fz[z, y, x];
                const double uxp    = ff.ux[z, y, x];
                const double uyp    = ff.uy[z, y, x];
                const double uzp    = ff.uz[z, y, x];
                const double uF     = uxp * forceX + uyp * forceY + uzp * forceZ;
                for (int i : std::views::iota(0, ndir)) {
                    const double ue = uxp * FluidFields::ex[i] + uyp * FluidFields::ey[i] + uzp * FluidFields::ez[i];
                    const double eF = FluidFields::ex[i] * forceX + FluidFields::ey[i] * forceY + FluidFields::ez[i] * forceZ;
                    ff.forcing[z, y, x, i] = omega_forcing * FluidFields::w[i]
                    * (3.0 * eF - 3.0 * uF + 9.0 * ue * eF);
                }
            }
        }
    }
}

template<typename BC>
void LbmSolver<BC>::ComputeMoments(FluidFields& ff) const {
    double arho, aux, auy, auz;
    #pragma omp parallel for default(shared) reduction(+:arho, aux, auy, auz) \
        num_threads(numprocs) schedule(static)
    for (int z : std::views::iota(0, nz)) {
        for (int y : std::views::iota(0, ny)) {
            for (int x : std::views::iota(0, nx)) {
                arho = 0; aux = 0; auy = 0, auz = 0;
                for (int i = 0; i < ndir; ++i) {
                    const double fi = ff.f[z, y, x, i];
                    arho += fi;
                    aux  += FluidFields::ex[i] * fi;
                    auy  += FluidFields::ey[i] * fi;
                    auz  += FluidFields::ez[i] * fi;
                }
                const double inv_r = 1.0 / arho;
                ff.rho[z, y, x] = arho;
                ff.ux[z, y, x]  = (aux + 0.5 * ff.fx[z, y, x] * DT) * inv_r;
                ff.uy[z, y, x]  = (auy + 0.5 * ff.fy[z, y, x] * DT) * inv_r;
                ff.uz[z, y, x]  = (auz + 0.5 * ff.fz[z, y, x] * DT) * inv_r;
            }
        }
    }
}

template<typename BC>
void LbmSolver<BC>::Collide(FluidFields& ff) const {
    const int n = nx * ny * nz * ndir;
    #pragma omp parallel for schedule(static) default(shared) num_threads(numprocs)
    for (int k = 0; k < n; ++k)
        ff.f_new_data[k] = omega * ff.f_data[k]
                         + omega_prime * ff.f_eq_data[k]
                         + DT * ff.forcing_data[k];
}

template<typename BC>
void LbmSolver<BC>::Stream(FluidFields& ff) const {
    // Interior: plain ±1 offsets, no BC dispatch needed
    #pragma omp parallel for schedule(static) default(shared) num_threads(numprocs)
    for (int z = 1; z < nz - 1; ++z)
        for (int y = 1; y < ny - 1; ++y)
            for (int x = 1; x < nx - 1; ++x)
                for (int i = 0; i < ndir; ++i)
                    ff.f[z + FluidFields::ez[i], y + FluidFields::ey[i], x + FluidFields::ex[i], i] = ff.f_new[z, y, x, i];

    // Boundary rows/columns: use UXoff/UYoff; out-of-domain destinations are
    // silently dropped (InDomain check), then rebuilt by HandleBoundaries.
    auto stream_cell = [&](int x, int y, int z) {
        for (int i = 0; i < ndir; ++i) {
            const int dx = UXoff(x, FluidFields::ex[i]);
            const int dy = UYoff(y, FluidFields::ey[i]);
            const int dz = UZoff(z, FluidFields::ez[i]);
            if (InDomain(dx, dy, dz))
                ff.f[dz, dy, dx, i] = ff.f_new[z, y, x, i];
        }
    };
    for (int y : std::views::iota(0, ny)) {
        for (int x : std::views::iota(0, nx)) {
            stream_cell(x, y, 0);
            stream_cell(x, y, nz-1);
        }
    }
    for (int z : std::views::iota(1, nz-1)) {
        for (int x : std::views::iota(0, nx)) {
            stream_cell(x, 0, z);
            stream_cell(x, ny-1, z);
        }
    }
    for (int z : std::views::iota(1, nz-1)) {
        for (int y : std::views::iota(1, ny-1)) {
            stream_cell(0, y, z);
            stream_cell(nx-1, y, z);
        }
    }

}

/* ─────────────────────────────────────────────────────────────────────────────
 * Per-wall boundary condition application
 *
 * All non-periodic walls use mid-point bounce-back (NoSlip / MovingWall) or
 * specular reflection (SpecularReflection).  Periodic walls are no-ops.
 *
 * Bounce-back formula (places wall halfway between last fluid node and ghost):
 *   f[m] = f_new[opp[m]]  +  6 · ρ · w[m] · (e[m] · u_wall)
 * where u_wall = (0,0,0) for NoSlip.
 *
 * Specular reflection (free-slip: reverses only the wall-normal component):
 *   Z-walls: f[m] = f_new[specZ[m]]   (reflect ez, keep ex, ey)
 *   Y-walls: f[m] = f_new[specY[m]]   (reflect ey, keep ex, ez)
 *   X-walls: f[m] = f_new[specX[m]]   (reflect ex, keep ey, ez)
 *
 * Corner handling:
 *   Each wall loop runs over its FULL extent (corners included), so corner
 *   cells receive contributions from both adjacent wall handlers.  X-wall
 *   handlers are called first, then Y-wall; Z-wall handlers run last and 
 *   take precedence for any overlapping direction.
 *   This correctly handles all common cases: closed box, lid-driven cavity,
 *   and channel flow, since the shear velocity is by convention applied to the Z-walls.
 *   Intersecting moving walls (e.g. left wall moving up
 *   AND top wall moving right) would need an explicit HandleCorner override.
 * ─────────────────────────────────────────────────────────────────────────────
 */

/* ─────────────────────────────────────────────────────────────────────────────
 * Z direction
 * ─────────────────────────────────────────────────────────────────────────────
 */

template<typename BC>
template<typename WallSpec>
void LbmSolver<BC>::HandleWallZHi(FluidFields& ff) const {
    using U = typename WallSpec::UBC;
    if constexpr (std::is_same_v<U, Periodic>) return;

    for (int y : std::views::iota(0, ny)) {
        for (int x : std::views::iota(0, nx)) {
            constexpr int z = nz - 1;
            if constexpr (std::is_same_v<U, SpecularReflection>) {
                for (int m : FluidFields::missingZHi)
                    ff.f[z, y, x, m] = ff.f_new[z, y, x, FluidFields::specZ[m]];
            } else {
                const double rho = ff.rho[z, y, x];
                const double Ux = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Ux; else return 0.0; }();
                const double Uy = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Uy; else return 0.0; }();
                    const double Uz = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Uz; else return 0.0; }();
                for (int m : FluidFields::missingZHi)
                    ff.f[z, y, x, m] = ff.f_new[z, y, x, FluidFields::opp[m]]
                                + kCs2InvTimes2 * rho * FluidFields::w[m]
                                    * (FluidFields::ex[m] * Ux + FluidFields::ey[m] * Uy + FluidFields::ez[m] * Uz);
            }
        }
    }
}

template<typename BC>
template<typename WallSpec>
void LbmSolver<BC>::HandleWallZLo(FluidFields& ff) const {
    using U = typename WallSpec::UBC;
    if constexpr (std::is_same_v<U, Periodic>) return;

    for (int y : std::views::iota(0, ny)) {
        for (int x : std::views::iota(0, nx)) {
            constexpr int z = 0;
            if constexpr (std::is_same_v<U, SpecularReflection>) {
                for (int m : FluidFields::missingZLo)
                    ff.f[z, y, x, m] = ff.f_new[z, y, x, FluidFields::specZ[m]];
            } else {
                const double rho = ff.rho[z, y, x];
                const double Ux = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Ux; else return 0.0; }();
                const double Uy = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Uy; else return 0.0; }();
                    const double Uz = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Uz; else return 0.0; }();
                for (int m : FluidFields::missingZLo)
                    ff.f[z, y, x, m] = ff.f_new[z, y, x, FluidFields::opp[m]]
                                + kCs2InvTimes2 * rho * FluidFields::w[m]
                                    * (FluidFields::ex[m] * Ux + FluidFields::ey[m] * Uy + FluidFields::ez[m] * Uz);
            }
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Y direction
 * ─────────────────────────────────────────────────────────────────────────────
 */

template<typename BC>
template<typename WallSpec>
void LbmSolver<BC>::HandleWallYHi(FluidFields& ff) const {
    using U = typename WallSpec::UBC;
    if constexpr (std::is_same_v<U, Periodic>) return;

    for (int z : std::views::iota(0, nz)) {
        for (int x : std::views::iota(0, nx)) {
            constexpr int y = ny - 1;
            if constexpr (std::is_same_v<U, SpecularReflection>) {
                for (int m : FluidFields::missingYHi)
                    ff.f[z, y, x, m] = ff.f_new[z, y, x, FluidFields::specY[m]];
            } else {
                const double rho = ff.rho[z, y, x];
                const double Ux = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Ux; else return 0.0; }();
                const double Uy = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Uy; else return 0.0; }();
                    const double Uz = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Uz; else return 0.0; }();
                for (int m : FluidFields::missingYHi)
                    ff.f[z, y, x, m] = ff.f_new[z, y, x, FluidFields::opp[m]]
                                + kCs2InvTimes2 * rho * FluidFields::w[m]
                                    * (FluidFields::ex[m] * Ux + FluidFields::ey[m] * Uy + FluidFields::ez[m] * Uz);
            }
        }
    }
}

template<typename BC>
template<typename WallSpec>
void LbmSolver<BC>::HandleWallYLo(FluidFields& ff) const {
    using U = typename WallSpec::UBC;
    if constexpr (std::is_same_v<U, Periodic>) return;

    for (int z : std::views::iota(0, nz)) {
        for (int x : std::views::iota(0, nx)) {
            constexpr int y = 0;
            if constexpr (std::is_same_v<U, SpecularReflection>) {
                for (int m : FluidFields::missingYLo)
                    ff.f[z, y, x, m] = ff.f_new[z, y, x, FluidFields::specY[m]];
            } else {
                const double rho = ff.rho[z, y, x];
                const double Ux = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Ux; else return 0.0; }();
                const double Uy = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Uy; else return 0.0; }();
                    const double Uz = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Uz; else return 0.0; }();
                for (int m : FluidFields::missingYLo)
                    ff.f[z, y, x, m] = ff.f_new[z, y, x, FluidFields::opp[m]]
                                + kCs2InvTimes2 * rho * FluidFields::w[m]
                                    * (FluidFields::ex[m] * Ux + FluidFields::ey[m] * Uy + FluidFields::ez[m] * Uz);
            }
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────────────
 * X direction
 * ─────────────────────────────────────────────────────────────────────────────
 */

template<typename BC>
template<typename WallSpec>
void LbmSolver<BC>::HandleWallXHi(FluidFields& ff) const {
    using U = typename WallSpec::UBC;
    if constexpr (std::is_same_v<U, Periodic>) return;

    for (int z : std::views::iota(0, nz)) {
        for (int y : std::views::iota(0, ny)) {
            constexpr int x = nx - 1;
            if constexpr (std::is_same_v<U, SpecularReflection>) {
                for (int m : FluidFields::missingXHi)
                    ff.f[z, y, x, m] = ff.f_new[z, y, x, FluidFields::specX[m]];
            } else {
                const double rho = ff.rho[z, y, x];
                const double Ux = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Ux; else return 0.0; }();
                const double Uy = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Uy; else return 0.0; }();
                    const double Uz = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Uz; else return 0.0; }();
                for (int m : FluidFields::missingXHi)
                    ff.f[z, y, x, m] = ff.f_new[z, y, x, FluidFields::opp[m]]
                                + kCs2InvTimes2 * rho * FluidFields::w[m]
                                    * (FluidFields::ex[m] * Ux + FluidFields::ey[m] * Uy + FluidFields::ez[m] * Uz);
            }
        }
    }
}

template<typename BC>
template<typename WallSpec>
void LbmSolver<BC>::HandleWallXLo(FluidFields& ff) const {
    using U = typename WallSpec::UBC;
    if constexpr (std::is_same_v<U, Periodic>) return;

    for (int z : std::views::iota(0, nz)) {
        for (int y : std::views::iota(0, ny)) {
            constexpr int x = 0;
            if constexpr (std::is_same_v<U, SpecularReflection>) {
                for (int m : FluidFields::missingXLo)
                    ff.f[z, y, x, m] = ff.f_new[z, y, x, FluidFields::specX[m]];
            } else {
                const double rho = ff.rho[z, y, x];
                const double Ux = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Ux; else return 0.0; }();
                const double Uy = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Uy; else return 0.0; }();
                    const double Uz = []() -> double {
                    if constexpr (is_moving_wall_v<U>) return U::Uz; else return 0.0; }();
                for (int m : FluidFields::missingXLo)
                    ff.f[z, y, x, m] = ff.f_new[z, y, x, FluidFields::opp[m]]
                                + kCs2InvTimes2 * rho * FluidFields::w[m]
                                    * (FluidFields::ex[m] * Ux + FluidFields::ey[m] * Uy + FluidFields::ez[m] * Uz);
            }
        }
    }
}



// X walls first, then Y and Z walls last — Z-wall velocity corrections win at corners.
template<typename BC>
void LbmSolver<BC>::HandleBoundaries(FluidFields& ff) const {
    HandleWallXLo<typename BC::XLo>(ff);
    HandleWallXHi<typename BC::XHi>(ff);
    HandleWallYLo<typename BC::YLo>(ff);
    HandleWallYHi<typename BC::YHi>(ff);
    HandleWallZLo<typename BC::ZLo>(ff);
    HandleWallZHi<typename BC::ZHi>(ff);
}

template<typename BC>
void LbmSolver<BC>::LatticeBoltzmannStep(FluidFields& ff) const {
    ResetFeq(ff);
    ComputeForcingTerms(ff);
    Collide(ff);
    Stream(ff);
    HandleBoundaries(ff);  // no-op for fully-periodic BC (compile-time dispatch)
    ComputeMoments(ff);
}
