#ifndef LBM_AN_OFFSETS_H_
#define LBM_AN_OFFSETS_H_

#include <type_traits>
#include "boundary.h"
#include "params.h"
#include "physics_helpers.h"

// ─────────────────────────────────────────────────────────────────────────────
// Index-offset policies, and per-axis QXoff/UXoff dispatch: maps (coord, step)
// to an effective neighbour index for a BCConfig (see boundary.h). Stateless,
// host/device-shared free functions — n is always Params::nx/ny/nz (no
// runtime-configurable grid size exists anywhere else in the codebase), so
// there's no object to construct or pass around; a stateful Grid<BC> class
// used to live here for exactly that purpose before nx/ny/nz became constexpr.
//
// QWallOffset: ghost-node index for the Q-tensor finite-difference stencil,
// for a single wall's BC tag.
//   Periodic  → modulo wrap.
//   Neumann / Anchoring → clamp (Q_ghost = Q_boundary).
//
//   The clamp gives ∂Q/∂n = 0 at the midpoint between the boundary node and
//   its (absent) ghost.  This midpoint is exactly where bounce-back places the
//   physical wall, so the Q and velocity BCs are mutually consistent.
//   A Zou/He velocity BC, which places the wall AT the boundary node, would
//   require a different Q stencil here.
//
// UWallOffset: streaming destination for LBM populations, for a single wall's
// BC tag.
//   Periodic → modulo wrap; NoSlip / MovingWall / SpecularReflection → raw
//   offset (out-of-domain destinations are dropped by InDomain in Stream;
//   HandleBoundaries then reconstructs the missing incoming populations).
//
// QAxisOffset / UAxisOffset: pick the Lo or Hi wall's *Offset above based on
// which side of the axis (x, y, or z) the step (i, s) falls off, or pass the
// interior offset through unchanged. QXoff/QYoff/QZoff/UXoff/UYoff/UZoff below
// are just QAxisOffset/UAxisOffset applied to one axis's WallSpec pair.
// ─────────────────────────────────────────────────────────────────────────────

template<typename BC>
inline CUDA_HOST_DEVICE constexpr int QWallOffset(int i, int s, int n) {
    if constexpr (std::is_same_v<BC, Periodic>)
        return ((i + s) % n + n) % n;
    else
        return (i + s < 0) ? 0 : (i + s >= n ? n - 1 : i + s);  // manual clamp: std::clamp isn't device-callable without --expt-relaxed-constexpr
}

template<typename BC>
inline CUDA_HOST_DEVICE constexpr int UWallOffset(int i, int s, int n) {
    if constexpr (std::is_same_v<BC, Periodic>)
        return ((i + s) % n + n) % n;
    else
        return i + s;  // raw: out-of-domain values dropped by InDomain in Stream
}

template<typename LoBC, typename HiBC>
inline CUDA_HOST_DEVICE int QAxisOffset(int i, int s, int n) {
    if (i + s <  0) return QWallOffset<LoBC>(i, s, n);
    if (i + s >= n) return QWallOffset<HiBC>(i, s, n);
    return i + s;
}

template<typename LoBC, typename HiBC>
inline CUDA_HOST_DEVICE int UAxisOffset(int i, int s, int n) {
    if (i + s <  0) return UWallOffset<LoBC>(i, s, n);
    if (i + s >= n) return UWallOffset<HiBC>(i, s, n);
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

// Velocity streaming offsets — may leave [0, n) for non-periodic BCs; callers
// must guard with InDomain (physics_helpers.h) before indexing with the result.
template<typename BCConfig> inline CUDA_HOST_DEVICE int UXoff(int x, int s) {
    return UAxisOffset<typename BCConfig::XLo::UBC, typename BCConfig::XHi::UBC>(x, s, Params::nx);
}
template<typename BCConfig> inline CUDA_HOST_DEVICE int UYoff(int y, int s) {
    return UAxisOffset<typename BCConfig::YLo::UBC, typename BCConfig::YHi::UBC>(y, s, Params::ny);
}
template<typename BCConfig> inline CUDA_HOST_DEVICE int UZoff(int z, int s) {
    return UAxisOffset<typename BCConfig::ZLo::UBC, typename BCConfig::ZHi::UBC>(z, s, Params::nz);
}

#endif // LBM_AN_OFFSETS_H_
