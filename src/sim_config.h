#ifndef LBM_AN_SIM_CONFIG_H_
#define LBM_AN_SIM_CONFIG_H_

#include "boundary.h"

// ── Boundary condition ────────────────────────────────────────────────────────
// Choose a preset from boundary.h or define a custom config here:
//

struct SlitConfig {
    using XLo = WallSpec<Periodic, Periodic>;
    using XHi = WallSpec<Periodic, Periodic>;
    using YLo = WallSpec<Periodic, Periodic>;
    using YHi = WallSpec<Periodic, Periodic>;
    using ZLo = WallSpec<Neumann, SpecularReflection>;
    using ZHi = WallSpec<Neumann, SpecularReflection>;
    static constexpr std::string_view name = "SlitFreeSlip";
};

// This is the critical line. SimBC is what the main code will use
using SimBC = SlitConfig;

// ── Time loop ─────────────────────────────────────────────────────────────────
inline constexpr int kNumSteps     = 200001;
inline constexpr int kSaveInterval = 2000;

#endif // LBM_AN_SIM_CONFIG_H_
