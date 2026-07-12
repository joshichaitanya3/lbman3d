#ifndef LBM_AN_OFFSETS_H_
#define LBM_AN_OFFSETS_H_

#include <type_traits>
#include "boundary.h"
#include "params.h"
#include "physics_helpers.h"

// ─────────────────────────────────────────────────────────────────────────────
// Index-offset policy for a Neumann-only (zero-gradient) finite-difference
// stencil, and per-axis QXoff dispatch: maps (coord, step) to an effective
// neighbour index for a BCConfig (see boundary.h). Stateless, host/device-
// shared free functions — n is always Params::nx/ny/nz (no runtime-
// configurable grid size exists anywhere else in the codebase), so there's
// no object to construct or pass around; a stateful Grid<BC> class used to
// live here for exactly that purpose before nx/ny/nz became constexpr.
//
// QXoff/QYoff/QZoff's only remaining caller is P's (nematic stress tensor)
// gradient in QTensorSolver::SetActiveStressAndComputeBodyForce: P has no
// prescribed wall target of its own to build a Dirichlet ghost from, so a
// Neumann-style clamp is the only available treatment there regardless of Q's
// own wall type. Everything else that used to live here — the streaming
// offset (StreamXoff/StreamYoff/StreamZoff), velocity's gradient ghost
// (VelocityGhost/VelocityGradientTensor), and Q's own gradient/Laplacian
// ghost (QGhost/QGradientAndLaplacian, correctly Dirichlet-aware for
// Anchoring<S,θ,φ> walls, not just Neumann) — has moved to
// boundary_handler.h: all three depend on which specific wall type is in
// play, so none of them collapse into a single reusable index the way this
// file's Neumann-only clamp does.
//
// QWallOffset: ghost-node index for a Neumann finite-difference stencil, for
// a single wall's BC tag.
//   Periodic → modulo wrap. Anything else → clamp (ghost = boundary value).
//
//   The clamp gives ∂/∂n = 0 at the midpoint between the boundary node and
//   its (absent) ghost. This midpoint is exactly where bounce-back places the
//   physical wall, so this and velocity's mid-point-convention ghost
//   (boundary_handler.h) are mutually consistent. A Zou/He velocity BC, which
//   places the wall AT the boundary node, would require a different stencil
//   here.
//
// QAxisOffset: picks the Lo or Hi wall's QWallOffset above based on which
// side of the axis (x, y, or z) the step (i, s) falls off, or passes the
// interior offset through unchanged. QXoff/QYoff/QZoff below are just
// QAxisOffset applied to one axis's WallSpec pair.
// ─────────────────────────────────────────────────────────────────────────────

template<typename BC>
inline CUDA_HOST_DEVICE constexpr int QWallOffset(int i, int s, int n) {
    if constexpr (std::is_same_v<BC, Periodic>)
        return ((i + s) % n + n) % n;
    else
        return (i + s < 0) ? 0 : (i + s >= n ? n - 1 : i + s);  // manual clamp: std::clamp isn't device-callable without --expt-relaxed-constexpr
}

template<typename LoBC, typename HiBC>
inline CUDA_HOST_DEVICE int QAxisOffset(int i, int s, int n) {
    if (i + s <  0) return QWallOffset<LoBC>(i, s, n);
    if (i + s >= n) return QWallOffset<HiBC>(i, s, n);
    return i + s;
}

// Q-tensor stencil offsets — always clamped/wrapped into [0, n).
template<typename BCConfig> inline CUDA_HOST_DEVICE int QXoff(int x, int s) {
    return QAxisOffset<typename BCConfig::XLo::QBC, typename BCConfig::XHi::QBC>(x, s, Params::nx);
}
template<typename BCConfig> inline CUDA_HOST_DEVICE int QYoff(int y, int s) {
    return QAxisOffset<typename BCConfig::YLo::QBC, typename BCConfig::YHi::QBC>(y, s, Params::ny);
}
template<typename BCConfig> inline CUDA_HOST_DEVICE int QZoff(int z, int s) {
    return QAxisOffset<typename BCConfig::ZLo::QBC, typename BCConfig::ZHi::QBC>(z, s, Params::nz);
}

#endif // LBM_AN_OFFSETS_H_
