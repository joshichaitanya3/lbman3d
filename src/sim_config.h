#ifndef LBM_AN_SIM_CONFIG_H_
#define LBM_AN_SIM_CONFIG_H_

#include "boundary.h"

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

// ── Time loop ─────────────────────────────────────────────────────────────────
inline constexpr int kNumSteps     = 10001;
inline constexpr int kSaveInterval = 100;

#endif // LBM_AN_SIM_CONFIG_H_
