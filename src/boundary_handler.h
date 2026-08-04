#ifndef LBM_AN_BOUNDARY_HANDLER_H_
#define LBM_AN_BOUNDARY_HANDLER_H_

#include <type_traits>
#include <cmath>
#include "boundary.h"
#include <params.h>
#include "physics_helpers.h"
#include "qtensor_types.h"
#include "offsets.h"
#include "local_grid.h"

using namespace Params;

// ─────────────────────────────────────────────────────────────────────────────
// Boundary-condition dispatch for both velocity and Q: stream-destination
// index offsets, wall-value extraction, ghost-VALUE construction for
// gradients/Laplacians (velocity's ∇u and Q's own ∇Q/∇²Q — NOT the nematic
// stress gradient, which stays on offsets.h's Neumann-only clamp; neither the
// symmetric part Sigma nor the antisymmetric part tau has a prescribed wall target
// to build a Dirichlet ghost from), and
// post-collision boundary reconstruction. Stateless, host/device-shared
// (CUDA_HOST_DEVICE) free functions/templates.
//
// This is the counterpart to offsets.h's QXoff/QYoff/QZoff, which remains
// the right tool for anything that only ever needs Neumann's zero-gradient
// clamp (the Sigma and Tau gradients in PassiveStressDivergence below, still using
// it as-is — see offsets.h for the Anchoring-wall limitation that carries).
// Velocity's ghost and Q's own
// Anchoring ghost both genuinely depend on the specific wall type (NoSlip
// vs MovingWall vs SpecularReflection; Neumann vs Anchoring<S,θ,φ>), so
// neither collapses into a single reusable index the way the Neumann-only
// clamp does — hence the value-level (not index-level) machinery below.
// ─────────────────────────────────────────────────────────────────────────────

// StreamXoff/StreamYoff/StreamZoff: destination index for one direction's
// stream from (x,y,z). Periodic -> modulo wrap (always in [0,n)); otherwise
// -> raw offset (may leave [0,n) — callers must guard with InDomain
// (physics_helpers.h) before indexing with the result, then fall back to
// HandleBoundaryPoint below to reconstruct the missing incoming population).
template<typename BC>
inline CUDA_HOST_DEVICE constexpr int StreamWallOffset(int i, int s, int n) {
    if constexpr (std::is_same_v<BC, Periodic>)
        return ((i + s) % n + n) % n;
    else
        return i + s;  // raw: out-of-domain values dropped by InDomain in Stream
}

template<typename LoBC, typename HiBC>
inline CUDA_HOST_DEVICE int StreamAxisOffset(int i, int s, int n) {
    if (i + s <  0) return StreamWallOffset<LoBC>(i, s, n);
    if (i + s >= n) return StreamWallOffset<HiBC>(i, s, n);
    return i + s;
}

template<typename BCConfig> inline CUDA_HOST_DEVICE int StreamXoff(int x, int s) {
    return StreamAxisOffset<typename BCConfig::XLo::UBC, typename BCConfig::XHi::UBC>(x, s, nx);
}
template<typename BCConfig> inline CUDA_HOST_DEVICE int StreamYoff(int y, int s) {
    return StreamAxisOffset<typename BCConfig::YLo::UBC, typename BCConfig::YHi::UBC>(y, s, ny);
}
template<typename BCConfig> inline CUDA_HOST_DEVICE int StreamZoff(int z, int s) {
    return StreamAxisOffset<typename BCConfig::ZLo::UBC, typename BCConfig::ZHi::UBC>(z, s, nz);
}

// Safe neighbor index for a raw array fetch feeding VelocityAxisGradient or
// QAxisGhostPair: idx(x,y,z) asserts its input is already in [0,n) (it wraps
// only *after* that check), so a raw i+s can't be handed to it directly on a
// non-periodic axis. Periodic -> real wraparound neighbor (always used
// as-is). Non-periodic -> clamps into [0,n) purely so idx() doesn't assert;
// the fetched value there is only ever read at the interior (clamp is then
// a no-op) or at a genuine wall (i==0 / i==n-1), where the caller discards
// it in favor of a properly-constructed ghost value anyway. BC-tag-generic
// (only checks Periodic-or-not), so shared verbatim between velocity's UBC
// and Q's QBC callers.
template<typename BC>
inline CUDA_HOST_DEVICE constexpr int SafeFetchOffset(int i, int s, int n) {
    if constexpr (std::is_same_v<BC, Periodic>)
        return ((i + s) % n + n) % n;
    else
        return (i + s < 0) ? 0 : (i + s >= n ? n - 1 : i + s);
}

template<typename LoBC, typename HiBC>
inline CUDA_HOST_DEVICE int SafeFetchAxisOffset(int i, int s, int n) {
    if (i + s <  0) return SafeFetchOffset<LoBC>(i, s, n);
    if (i + s >= n) return SafeFetchOffset<HiBC>(i, s, n);
    return i + s;
}

// Wall velocity extraction: 0 for NoSlip/SpecularReflection, the prescribed
// component for MovingWall<Ux,Uy,Uz>.
template<typename U> inline CUDA_HOST_DEVICE constexpr double wallVx() {
    if constexpr (is_moving_wall_v<U>) return U::Ux; else return 0.0;
}
template<typename U> inline CUDA_HOST_DEVICE constexpr double wallVy() {
    if constexpr (is_moving_wall_v<U>) return U::Uy; else return 0.0;
}
template<typename U> inline CUDA_HOST_DEVICE constexpr double wallVz() {
    if constexpr (is_moving_wall_v<U>) return U::Uz; else return 0.0;
}

enum class Axis { X, Y, Z };

template<Axis A, typename UBC>
inline CUDA_HOST_DEVICE constexpr double WallVelocity() {
    if constexpr (A == Axis::X) return wallVx<UBC>();
    else if constexpr (A == Axis::Y) return wallVy<UBC>();
    else return wallVz<UBC>();
}

// Ghost VALUE (not index) for velocity component A at a wall of type UBC.
// Mid-point bounce-back convention: the wall sits half a cell beyond the
// boundary node, so the ghost is chosen such that linear interpolation
// between (ghost, boundary) lands on the true wall condition at that
// midpoint: v_ghost = 2*wall_value - v_boundary.
//   NoSlip / MovingWall: Dirichlet (u = 0 or U_wall) for every component.
//   SpecularReflection: Dirichlet(0) (no-penetration) for the component
//     parallel to the wall normal; Neumann (zero-gradient, v_ghost =
//     v_boundary) for tangential components — is_normal disambiguates which.
template<Axis A, typename UBC>
inline CUDA_HOST_DEVICE double VelocityGhost(double v_boundary, bool is_normal) {
    constexpr double wall_value = WallVelocity<A, UBC>();
    if constexpr (std::is_same_v<UBC, SpecularReflection>) {
        return is_normal ? (2.0 * wall_value - v_boundary) : v_boundary;
    } else {
        // NoSlip (wall_value == 0) or MovingWall<Ux,Uy,Uz>
        return 2.0 * wall_value - v_boundary;
    }
}


struct NeighborPair { double minus, plus; };

template<Axis A, typename LoBC, typename HiBC>
inline CUDA_HOST_DEVICE NeighborPair VelocityAxisGhostPair(
    int i, int n, double v_minus, double v_center, double v_plus, bool is_normal
) {
    if constexpr (std::is_same_v<LoBC, Periodic>) {
        return NeighborPair{v_minus, v_plus};
    } else {
        const double vm = (i == 0)   ? VelocityGhost<A, LoBC>(v_center, is_normal) : v_minus;
        const double vp = (i == n-1) ? VelocityGhost<A, HiBC>(v_center, is_normal) : v_plus;
        return NeighborPair{vm, vp};
    }
}

// Central-difference gradient of velocity component A along an axis whose
// walls are LoBC/HiBC. v_minus/v_plus must already be fetched with a
// wraparound-safe index (e.g. idx(x-1,y,z), which always wraps modulo n —
// see offsets.h) so the fetch itself never goes out of bounds; on a
// Periodic axis those fetched values are the real neighbors and used
// directly, short-circuiting before VelocityGhost (which has no Periodic
// case) is ever consulted. On a non-Periodic axis, i==0/i==n-1 replaces the
// (physically meaningless, since idx() wrapped it) fetched value with the
// correct wall-aware ghost value instead.
template<Axis A, typename LoBC, typename HiBC>
inline CUDA_HOST_DEVICE double VelocityAxisGradient(
    int i, int n, double v_minus, double v_center, double v_plus, bool is_normal
) {
    NeighborPair pair = VelocityAxisGhostPair<A, LoBC, HiBC>(i, n, v_minus, v_center, v_plus, is_normal);
    return (pair.plus - pair.minus) / 2.0;
}


// Full (non-symmetric) velocity gradient tensor at (x,y,z): ∇u with
// components v_A_B = ∂(u_A)/∂B. Central differences with wall-aware ghost
// values (VelocityGhost/VelocityAxisGradient above) rather than Q's
// Neumann-only clamp (QXoff/QYoff/QZoff in offsets.h) — see offsets.h's
// header comment for why the two can't share one index-offset scheme.
// uz_z isn't stored; derive it via incompressibility (-(ux_x + uy_y)) at
// the call site, same as the device-side VelGradient in kernels.cu, whose
// GradTensor return type this shares (kept host/device-common since it's a
// plain data aggregate with no access-pattern-specific logic).
template<typename BCConfig>
inline CUDA_HOST_DEVICE GradTensor VelocityGradientTensor(
    const double* ux, const double* uy, const double* uz, int x, int y, int z, const LocalGrid& g
) {
    using XLoU = typename BCConfig::XLo::UBC;
    using XHiU = typename BCConfig::XHi::UBC;
    using YLoU = typename BCConfig::YLo::UBC;
    using YHiU = typename BCConfig::YHi::UBC;
    using ZLoU = typename BCConfig::ZLo::UBC;
    using ZHiU = typename BCConfig::ZHi::UBC;

    const int xm = SafeFetchAxisOffset<XLoU, XHiU>(x, -1, nx);
    const int xp = SafeFetchAxisOffset<XLoU, XHiU>(x, +1, nx);
    const int ym = SafeFetchAxisOffset<YLoU, YHiU>(y, -1, ny);
    const int yp = SafeFetchAxisOffset<YLoU, YHiU>(y, +1, ny);
    const int zm = SafeFetchAxisOffset<ZLoU, ZHiU>(z, -1, nz);
    const int zp = SafeFetchAxisOffset<ZLoU, ZHiU>(z, +1, nz);

    const double ux0 = ux[g.halo_idx(x, y, z)];
    const double uy0 = uy[g.halo_idx(x, y, z)];
    const double uz0 = uz[g.halo_idx(x, y, z)];

    return GradTensor{
        VelocityAxisGradient<Axis::X, XLoU, XHiU>(x, nx, ux[g.halo_idx(xm, y, z)], ux0, ux[g.halo_idx(xp, y, z)], true),
        VelocityAxisGradient<Axis::X, YLoU, YHiU>(y, ny, ux[g.halo_idx(x, ym, z)], ux0, ux[g.halo_idx(x, yp, z)], false),
        VelocityAxisGradient<Axis::X, ZLoU, ZHiU>(z, nz, ux[g.halo_idx(x, y, zm)], ux0, ux[g.halo_idx(x, y, zp)], false),

        VelocityAxisGradient<Axis::Y, XLoU, XHiU>(x, nx, uy[g.halo_idx(xm, y, z)], uy0, uy[g.halo_idx(xp, y, z)], false),
        VelocityAxisGradient<Axis::Y, YLoU, YHiU>(y, ny, uy[g.halo_idx(x, ym, z)], uy0, uy[g.halo_idx(x, yp, z)], true),
        VelocityAxisGradient<Axis::Y, ZLoU, ZHiU>(z, nz, uy[g.halo_idx(x, y, zm)], uy0, uy[g.halo_idx(x, y, zp)], false),

        VelocityAxisGradient<Axis::Z, XLoU, XHiU>(x, nx, uz[g.halo_idx(xm, y, z)], uz0, uz[g.halo_idx(xp, y, z)], false),
        VelocityAxisGradient<Axis::Z, YLoU, YHiU>(y, ny, uz[g.halo_idx(x, ym, z)], uz0, uz[g.halo_idx(x, yp, z)], false)
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Q-tensor gradient/Laplacian ghost construction. Mirrors VelocityGhost/
// VelocityAxisGradient above, but simpler: Anchoring<S,θ,φ> prescribes all 5
// independent Q components uniformly (qxx, qxy, qxz, qyy, qyz; qzz is
// implied via tracelessness) — no normal/tangential split like velocity's
// SpecularReflection needed. Supersedes the old HandleQBoundaryPoint, which
// used to overwrite q_new AT the boundary node with this same target value
// post-step (both the wrong location — the wall is a half-cell beyond the
// boundary node, same mid-point convention as everywhere else — and only a
// crude proxy for a real ghost, since it fed the *next* step's Neumann-style
// clamp rather than a proper Dirichlet ghost). With QGhost below wired into
// every gradient/Laplacian evaluation near a wall, the boundary node's own Q
// evolves through the ordinary FD update like any interior point, correctly
// feeling the wall through its ghost neighbor — no post-hoc overwrite needed.
// ─────────────────────────────────────────────────────────────────────────────

// Strong-anchoring target Q (order parameter S at director angle (θ,φ)),
// same formula boundary.h's Anchoring<S,θ,φ> documents. Only meaningful for
// QBC = Anchoring<...>; the true branch below never instantiates against a
// QBC without ::s/::theta/::phi (Neumann, Periodic) since if constexpr
// discards the untaken branch.
template<typename QBC>
inline CUDA_HOST_DEVICE SymTrLessTensor5 AnchoredQ() {
    if constexpr (is_anchoring_v<QBC>) {
        const double sin_sq_phi     = 0.5 * (1.0 - std::cos(2.0 * QBC::phi));
        const double cos_sq_phi     = 1.0 - sin_sq_phi;
        const double sin_phi_cos_phi = 0.5 * std::sin(2.0 * QBC::phi);
        const double sin_sq_th      = 0.5 * (1.0 - std::cos(2.0 * QBC::theta));
        const double sin_th_cos_th  = 0.5 * std::sin(2.0 * QBC::theta);
        const double S = QBC::s;
        return SymTrLessTensor5{
            S * (cos_sq_phi * sin_sq_th - 1.0/3.0),
            S * (sin_phi_cos_phi * sin_sq_th),
            S * (std::cos(QBC::phi) * sin_th_cos_th),
            S * (sin_sq_phi * sin_sq_th - 1.0/3.0),
            S * (std::sin(QBC::phi) * sin_th_cos_th)
        };
    } else {
        return SymTrLessTensor5{0.0, 0.0, 0.0, 0.0, 0.0};
    }
}

// Ghost VALUE for Q component C at a wall of type QBC. Mid-point convention,
// same as VelocityGhost: Q_ghost = 2*wall_target - Q_boundary.
//   Neumann: wall_target = Q_boundary itself (zero-gradient) -> Q_ghost = Q_boundary.
//   Anchoring<S,θ,φ>: wall_target = the prescribed strong-anchoring value.
template<QComp C, typename QBC>
inline CUDA_HOST_DEVICE double QGhost(double q_boundary) {
    if constexpr (is_anchoring_v<QBC>) {
        return 2.0 * QComponent<C>(AnchoredQ<QBC>()) - q_boundary;
    } else {
        return q_boundary; // Neumann clamp; Periodic never reaches here
    }
}

// Ghost-substituted (minus, plus) neighbor pair for Q component C along an
// axis whose walls are LoBC/HiBC. q_minus/q_plus must already be fetched
// with SafeFetchAxisOffset. Periodic short-circuits to the real fetched
// neighbors (QGhost has no Periodic case); otherwise i==0/i==n-1 replaces
// the (physically meaningless, since the fetch clamped) value with QGhost.
template<QComp C, typename LoBC, typename HiBC>
inline CUDA_HOST_DEVICE NeighborPair QAxisGhostPair(
    int i, int n, double q_minus, double q_center, double q_plus
) {
    if constexpr (std::is_same_v<LoBC, Periodic>) {
        return NeighborPair{q_minus, q_plus};
    } else {
        const double qm = (i == 0)   ? QGhost<C, LoBC>(q_center) : q_minus;
        const double qp = (i == n-1) ? QGhost<C, HiBC>(q_center) : q_plus;
        return NeighborPair{qm, qp};
    }
}

// Gradient (central difference) and Laplacian (7-point stencil) of Q
// component C at (x,y,z), wall-aware via QGhost/QAxisGhostPair above rather
// than offsets.h's Neumann-only QXoff/QYoff/QZoff clamp.
template<QComp C, typename BCConfig>
inline CUDA_HOST_DEVICE QDerivs QGradientAndLaplacian(const double* q, int x, int y, int z, const LocalGrid& g) {
    using XLoQ = typename BCConfig::XLo::QBC;
    using XHiQ = typename BCConfig::XHi::QBC;
    using YLoQ = typename BCConfig::YLo::QBC;
    using YHiQ = typename BCConfig::YHi::QBC;
    using ZLoQ = typename BCConfig::ZLo::QBC;
    using ZHiQ = typename BCConfig::ZHi::QBC;

    const int xm = SafeFetchAxisOffset<XLoQ, XHiQ>(x, -1, nx);
    const int xp = SafeFetchAxisOffset<XLoQ, XHiQ>(x, +1, nx);
    const int ym = SafeFetchAxisOffset<YLoQ, YHiQ>(y, -1, ny);
    const int yp = SafeFetchAxisOffset<YLoQ, YHiQ>(y, +1, ny);
    const int zm = SafeFetchAxisOffset<ZLoQ, ZHiQ>(z, -1, nz);
    const int zp = SafeFetchAxisOffset<ZLoQ, ZHiQ>(z, +1, nz);

    const double q0 = q[g.halo_idx(x, y, z)];

    const NeighborPair px = QAxisGhostPair<C, XLoQ, XHiQ>(x, nx, q[g.halo_idx(xm, y, z)], q0, q[g.halo_idx(xp, y, z)]);
    const NeighborPair py = QAxisGhostPair<C, YLoQ, YHiQ>(y, ny, q[g.halo_idx(x, ym, z)], q0, q[g.halo_idx(x, yp, z)]);
    const NeighborPair pz = QAxisGhostPair<C, ZLoQ, ZHiQ>(z, nz, q[g.halo_idx(x, y, zm)], q0, q[g.halo_idx(x, y, zp)]);

    return QDerivs{
        (px.plus - px.minus) / 2.0,
        (py.plus - py.minus) / 2.0,
        (pz.plus - pz.minus) / 2.0,
        (px.minus + px.plus + py.minus + py.plus + pz.minus + pz.plus) - 6.0 * q0
    };
}

// Divergence of the nematic stress under the row convention
//
//     f_alpha = d_beta Pi_alpha,beta
//
// Pi splits into a symmetric-traceless part Sigma (sigma_xx..sigma_yz, with
// Σ_zz = -(Σ_xx + Σ_yy)) and an antisymmetric part tau (tau_xy/tau_xz/tau_yz, upper
// triangle only). Since tau_beta,alpha = -tau_alpha,beta, the lower-triangle
// entries fy and fz reference carry the opposite sign to the upper-triangle ones
// fx references:
//
//     f_x = d_x Σ_xx          + d_y(Σ_xy + τ_xy) + d_z(Σ_xz + τ_xz)
//     f_y = d_x(Σ_xy - τ_xy)  + d_y Σ_yy         + d_z(Σ_yz + τ_yz)
//     f_z = d_x(Σ_xz - τ_xz)  + d_y(Σ_yz - τ_yz) + d_z Σ_zz
template<typename BCConfig>
inline CUDA_HOST_DEVICE Vec3 PassiveStressDivergence(
    const double* sigma_xx,
    const double* sigma_xy,
    const double* sigma_xz,
    const double* sigma_yy,
    const double* sigma_yz,
    const double* tau_xy,
    const double* tau_xz,
    const double* tau_yz,
    const int x,
    const int y,
    const int z,
    const LocalGrid& g
) {

    const int xm = QXoff<BCConfig>(x, -1);
    const int xp = QXoff<BCConfig>(x, +1);
    const int ym = QYoff<BCConfig>(y, -1);
    const int yp = QYoff<BCConfig>(y, +1);
    const int zm = QZoff<BCConfig>(z, -1);
    const int zp = QZoff<BCConfig>(z, +1);

    // Central differences of each stress component along each axis
    const double dx_Sigma_xx = (sigma_xx[g.halo_idx(xp, y, z)] - sigma_xx[g.halo_idx(xm, y, z)]) / 2.0;
    const double dx_Sigma_xy = (sigma_xy[g.halo_idx(xp, y, z)] - sigma_xy[g.halo_idx(xm, y, z)]) / 2.0;
    const double dx_Sigma_xz = (sigma_xz[g.halo_idx(xp, y, z)] - sigma_xz[g.halo_idx(xm, y, z)]) / 2.0;
    const double dx_Tau_xy = (tau_xy[g.halo_idx(xp, y, z)] - tau_xy[g.halo_idx(xm, y, z)]) / 2.0;
    const double dx_Tau_xz = (tau_xz[g.halo_idx(xp, y, z)] - tau_xz[g.halo_idx(xm, y, z)]) / 2.0;

    const double dy_Sigma_xy = (sigma_xy[g.halo_idx(x, yp, z)] - sigma_xy[g.halo_idx(x, ym, z)]) / 2.0;
    const double dy_Sigma_yy = (sigma_yy[g.halo_idx(x, yp, z)] - sigma_yy[g.halo_idx(x, ym, z)]) / 2.0;
    const double dy_Sigma_yz = (sigma_yz[g.halo_idx(x, yp, z)] - sigma_yz[g.halo_idx(x, ym, z)]) / 2.0;
    const double dy_Tau_xy = (tau_xy[g.halo_idx(x, yp, z)] - tau_xy[g.halo_idx(x, ym, z)]) / 2.0;
    const double dy_Tau_yz = (tau_yz[g.halo_idx(x, yp, z)] - tau_yz[g.halo_idx(x, ym, z)]) / 2.0;

    const double dz_Sigma_xz = (sigma_xz[g.halo_idx(x, y, zp)] - sigma_xz[g.halo_idx(x, y, zm)]) / 2.0;
    const double dz_Sigma_yz = (sigma_yz[g.halo_idx(x, y, zp)] - sigma_yz[g.halo_idx(x, y, zm)]) / 2.0;
    const double dz_Tau_xz = (tau_xz[g.halo_idx(x, y, zp)] - tau_xz[g.halo_idx(x, y, zm)]) / 2.0;
    const double dz_Tau_yz = (tau_yz[g.halo_idx(x, y, zp)] - tau_yz[g.halo_idx(x, y, zm)]) / 2.0;
    // Σ_zz = -(Σ_xx + Σ_yy); τ has no diagonal, so it does not enter here.
    const double dz_Sigma_zz = -((sigma_xx[g.halo_idx(x, y, zp)] - sigma_xx[g.halo_idx(x, y, zm)]) / 2.0
                          + (sigma_yy[g.halo_idx(x, y, zp)] - sigma_yy[g.halo_idx(x, y, zm)]) / 2.0);

    const double fx = dx_Sigma_xx
                    + (dy_Sigma_xy + dy_Tau_xy)
                    + (dz_Sigma_xz + dz_Tau_xz);

    const double fy = (dx_Sigma_xy - dx_Tau_xy)
                    + dy_Sigma_yy
                    + (dz_Sigma_yz + dz_Tau_yz);

    const double fz = (dx_Sigma_xz - dx_Tau_xz)
                    + (dy_Sigma_yz - dy_Tau_yz)
                    + dz_Sigma_zz;

    return {fx, fy, fz};
}
// Post-collision boundary handling for a single out-of-domain stream. Writes
// the reconstructed population to f_new at the SOURCE node (x,y,z), not the
// out-of-domain destination:
//   SpecularReflection : f_new[i_refl] = f_star
//   NoSlip / MovingWall: f_new[opp[i]] = f_star + 6·ρ·w[opp[i]]·(e[opp[i]]·u_wall)
// rhop is rho at (x,y,z) — pass the value the caller already has (e.g. from
// ComputeMoments) rather than a whole rho array, since only one point is
// ever needed here. ex/ey/ez/w/opp are the D3Q15 stencil arrays — Lattice::
// on host, the __constant__ d_ex etc. arrays on device (mirrors
// ComputeMoments's existing ex/ey/ez parameterization in physics_helpers.h).
template<typename WallSpec>
inline CUDA_HOST_DEVICE void HandleBoundaryPoint(
    int x, int y, int z, int i, int i_refl, double f_star, double rhop,
    double* f_new,
    const int* ex, const int* ey, const int* ez,
    const double* w, const int* opp, const LocalGrid& g
) {
    using U = typename WallSpec::UBC;
    if constexpr (std::is_same_v<U, Periodic>) {
        // Periodic axes never reach here: StreamXoff/StreamYoff/StreamZoff
        // wrap destinations in-domain. Guard against silent bounce-back if
        // that assumption breaks.
        return;
    }
    else if constexpr (std::is_same_v<U, SpecularReflection>) {
        f_new[g.halo_idx(x, y, z, i_refl)] = f_star;
    }
    else {
        constexpr Vec3 U_wall{wallVx<U>(), wallVy<U>(), wallVz<U>()};
        const int m = opp[i];
        const Vec3 e_m{static_cast<double>(ex[m]), static_cast<double>(ey[m]), static_cast<double>(ez[m])};
        f_new[g.halo_idx(x, y, z, m)] = f_star + kCs2InvTimes2 * rhop * w[m] * e_m.Dot(U_wall);
    }
}

#endif // LBM_AN_BOUNDARY_HANDLER_H_
