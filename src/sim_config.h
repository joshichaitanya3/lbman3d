#ifndef LBM_AN_SIM_CONFIG_H_
#define LBM_AN_SIM_CONFIG_H_

#include "boundary.h"
#include "qtensor_types.h"

// ── Boundary condition ────────────────────────────────────────────────────────
// Choose a preset from boundary.h or define a custom config here:
//

// ── Slit geometry: the confined axis is X ─────────────────────────────────────
// The plate separation H is `nx`, with y and z the long periodic directions.
// This matches ChannelConfig, which also puts its confined axes first.
//
// The confined axis is the short one, and `idx` makes z the slowest-varying
// index. Keeping the short axis off z is what leaves the outer (z) loop — the
// one carrying `#pragma omp parallel for` in every solver phase — with enough
// iterations to spread across the thread pool.
struct SlitConfig {
    using XLo = WallSpec<Neumann, SpecularReflection>;
    using XHi = WallSpec<Neumann, SpecularReflection>;
    using YLo = WallSpec<Periodic, Periodic>;
    using YHi = WallSpec<Periodic, Periodic>;
    using ZLo = WallSpec<Periodic, Periodic>;
    using ZHi = WallSpec<Periodic, Periodic>;
    static constexpr std::string_view name = "SlitFreeSlip";
};

// Same slab geometry as SlitConfig but with NO-SLIP plates instead of free-slip.
// Free-slip walls exert no tangential stress, so with MU = 0 the system has no
// momentum sink at all and in-plane momentum is conserved and undamped. Measured:
// free-slip diverges (step 16000 at nu=2/3, step 11000 at nu=1.833 — viscosity
// does not help), while no-slip saturates at u_rms = 0.0155 through 12000 steps.
// Shendruk et al. 2018 Fig. 4 validate this variant: the 2D->3D transition is
// sharper but the critical activity number is unchanged, so it is a faithful
// alternative rather than a workaround.
struct SlitNoSlipConfig {
    using XLo = WallSpec<Neumann, NoSlip>;
    using XHi = WallSpec<Neumann, NoSlip>;
    using YLo = WallSpec<Periodic, Periodic>;
    using YHi = WallSpec<Periodic, Periodic>;
    using ZLo = WallSpec<Periodic, Periodic>;
    using ZHi = WallSpec<Periodic, Periodic>;
    static constexpr std::string_view name = "SlitNoSlip";
};

// This is the critical line. SimBC is what the main code will use
using SimBC = SlitNoSlipConfig;
// SlitNoSlipConfig with nx=20 for a benchmark run.

// ── Q advection scheme ────────────────────────────────────────────────────────
// Advection::Centred is second-order accurate but imposes the cell-Peclet
// constraint |u| <= 2*GAMMA*L (see the stability budget in params.h), which is
// what fails first at high activity or high LAMBDA. Advection::Upwind (first
// order) lifts that constraint at the cost of a numerical Q diffusivity
// |u|*DX/2 — at |u| = 0.05 that is 0.025, slightly ABOVE the physical
// GAMMA*L = 0.0204, so compare the two before trusting upwind results
// quantitatively.
//
// Measured envelope, ChannelConfig at 20x100x100, 3000 steps, 5 RNG seeds per
// cell (divergences out of 5):
//
//   LAMBDA     0.55   0.6   0.7   0.8   0.9   1.0
//   Centred     2/5   5/5   5/5    -     -    5/5
//   Upwind      0/5   0/5   0/5   4/5   5/5   5/5
//
// Upwinding moves the reliable ceiling from ~0.5 to 0.7 and removes the
// seed-dependence at 0.55, but does NOT reach LAMBDA = 1. That failure is not
// advection-limited: raising viscosity 7x does not help it either, and the
// growth is in the Q equation rather than the flow.
//
// LAMBDA near threshold is seed-dependent, so a single run does not establish a
// ceiling — repeat over seeds before concluding.
//
// BENCHMARK WARNING. The upwind correction is +(|u|/2) * d2Q added to dQ/dt,
// which is the same form as the elastic term GAMMA*L*lap(Q). Upwinding therefore
// enlarges the effective Frank constant,
//
//     L_eff = L + |u| / (2*GAMMA)
//
// and at GAMMA = 0.34, L = 0.06 that is not a small correction:
//
//     |u| = 0.0155 (measured steady state):  L_eff = 1.38 L  ->  A_act 17.5 -> 14.9
//     |u| = 0.05   (high-LAMBDA regime):     L_eff = 2.2  L  ->  A_act 17.5 -> 11.7
//
// So an upwinded run does NOT sit at its nominal activity number, and because
// L_eff depends on the local |u| it is not even a constant offset to correct
// for. Only the advective term is affected — H, the stresses and the body force
// are all still built from the centred Laplacian and gradients — which is
// exactly why the Q dynamics and the free energy stop referring to the same
// elastic constant. Going second order (linear-upwind/QUICK), whose leading
// error is dispersive rather than diffusive, is the way out.
inline constexpr Advection kQAdvection = Advection::Centred;

// ── Time loop ─────────────────────────────────────────────────────────────────
inline constexpr int kNumSteps     = 10001;
inline constexpr int kSaveInterval = 100;

#endif // LBM_AN_SIM_CONFIG_H_
