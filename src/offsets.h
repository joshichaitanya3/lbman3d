#ifndef LBM_AN_OFFSETS_H_
#define LBM_AN_OFFSETS_H_

#include <type_traits>
#include "boundary.h"
#include <params.h>
#include "physics_helpers.h"
#include "local_grid.h"

// ─────────────────────────────────────────────────────────────────────────────
// Index-offset policy for a Neumann-only (zero-gradient) finite-difference
// stencil, and per-axis QXoff dispatch: maps (coord, step) to an effective
// neighbour index for a BCConfig (see boundary.h). Stateless, host/device-
// shared free functions.
//
// Two coordinate frames meet here. QWallOffset/QAxisOffset work in GLOBAL
// coordinates against the global extent n = Params::nx/ny/nz — that is where
// periodic wraps and physical walls are defined. Every caller, though, holds
// RANK-LOCAL coordinates and indexes a halo-padded local buffer.
// ResolveAxisOffset bridges the two: it makes the BC decision on the global
// coordinate (offset + i) and returns a local index. On an unsplit axis
// (local_n == n, so offset == 0) local and global coincide and the result is
// QAxisOffset's, unchanged. On a split axis, a step that crosses a rank seam
// or a periodic wrap lands in the ghost layer (i + s), which the halo
// exchange has filled with the neighbour's owned value; only a step through
// a physical wall is clamped back onto the boundary node. This is the same
// rule LatticeBoltzmannStep applies to streaming (its x_split/dx_eff).
//
// QXoff/QYoff/QZoff's solver caller is the nematic stress gradient in
// QTensorSolver::SetActiveStressAndComputeBodyForce — that means BOTH parts of
// the stress: the symmetric-traceless Sigma (Sigma_xx..Sigma_yz) and the antisymmetric
// tau (Tau_xy/Tau_xz/Tau_yz). Neither has a prescribed wall target of its own to build a
// Dirichlet ghost from, so a Neumann-style clamp is the only available treatment
// there regardless of Q's own wall type.
//
// Known limitation, currently unquantified: at an Anchoring<S,θ,φ> wall the true
// normal stress gradient is nonzero (anchoring generates elastic stress), so the
// clamp is an O(1) boundary-layer error in ∇·Π at the wall node and its
// neighbour. It is unreachable in the shipped configs — none use a non-Neumann Q
// wall — and settling it needs a grid-refinement study on an anchored-wall
// quiescent case (a consistent discretization error shrinks under refinement; a
// boundary-condition error does not), not a blind change of stencil.
//
// Everything else that used to live here — the streaming
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
// ResolveAxisOffset applied to one axis's WallSpec pair.
//
// QWallOffset/QAxisOffset only distinguish Periodic from not-Periodic, so they
// serve velocity's UBC tags as well as Q's QBC tags; boundary_handler.h uses
// ResolveAxisOffset for both.
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

// Resolve (local i, step s) to a local/halo index along one axis. BC decisions
// are made on the global coordinate offset + i against the global extent n;
// the result is local. See the header comment for the split-axis rule.
template<typename LoBC, typename HiBC>
inline CUDA_HOST_DEVICE int ResolveAxisOffset(int i, int s, int offset, int local_n, int n) {
    const int gi = offset + i;
    const int gr = QAxisOffset<LoBC, HiBC>(gi, s, n);  // global wrap/clamp
    if (local_n == n) return gr;                       // unsplit: local == global
    const bool crosses_wall = (gi + s <  0 && !std::is_same_v<LoBC, Periodic>)
                           || (gi + s >= n && !std::is_same_v<HiBC, Periodic>);
    return crosses_wall ? gr - offset : i + s;         // split: seam or periodic wrap -> ghost
}

// Q-tensor stencil offsets for local coordinates on grid g. The result is an
// owned index or, on a split axis, a ghost index (-1 or local_n).
template<typename BCConfig> inline CUDA_HOST_DEVICE int QXoff(int x, int s, const LocalGrid& g) {
    return ResolveAxisOffset<typename BCConfig::XLo::QBC, typename BCConfig::XHi::QBC>(x, s, g.offset_x, g.local_nx, Params::nx);
}
template<typename BCConfig> inline CUDA_HOST_DEVICE int QYoff(int y, int s, const LocalGrid& g) {
    return ResolveAxisOffset<typename BCConfig::YLo::QBC, typename BCConfig::YHi::QBC>(y, s, g.offset_y, g.local_ny, Params::ny);
}
template<typename BCConfig> inline CUDA_HOST_DEVICE int QZoff(int z, int s, const LocalGrid& g) {
    return ResolveAxisOffset<typename BCConfig::ZLo::QBC, typename BCConfig::ZHi::QBC>(z, s, g.offset_z, g.local_nz, Params::nz);
}

#endif // LBM_AN_OFFSETS_H_
