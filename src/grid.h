#ifndef LBM_AN_GRID_H_
#define LBM_AN_GRID_H_

#include <algorithm>
#include <string_view>
#include "boundary.h"

// ─────────────────────────────────────────────────────────────────────────────
// Index-offset policies — used by Grid internals only.
//
// QOffset: ghost-node index for the Q-tensor finite-difference stencil.
//   Periodic  → modulo wrap.
//   Neumann / Anchoring → clamp (Q_ghost = Q_boundary).
//
//   The clamp gives ∂Q/∂n = 0 at the midpoint between the boundary node and
//   its (absent) ghost.  This midpoint is exactly where bounce-back places the
//   physical wall, so the Q and velocity BCs are mutually consistent.
//   A Zou/He velocity BC, which places the wall AT the boundary node, would
//   require a different Q stencil here.
//
// UOffset: streaming destination for LBM populations.
//   Periodic → modulo wrap; NoSlip / MovingWall / SpecularReflection → raw
//   offset (out-of-domain destinations are dropped by InDomain in Stream;
//   HandleBoundaries then reconstructs the missing incoming populations).
// ─────────────────────────────────────────────────────────────────────────────

template<typename BC>
constexpr int QOffset(int i, int s, int n) {
    if constexpr (std::is_same_v<BC, Periodic>)
        return ((i + s) % n + n) % n;
    else
        return std::clamp(i + s, 0, n - 1);
}

template<typename BC>
constexpr int UOffset(int i, int s, int n) {
    if constexpr (std::is_same_v<BC, Periodic>)
        return ((i + s) % n + n) % n;
    else
        return i + s;  // raw: out-of-domain values dropped by InDomain in Stream
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid<BCConfig>: maps (coord, step) → effective neighbour index.
//
// Four methods, one per field/direction pair:
//   QXoff / QYoff  — Q-tensor stencil (FiniteDifferenceStep, ComputeActiveBodyForce)
//   UXoff / UYoff  — velocity streaming (Stream)
// ─────────────────────────────────────────────────────────────────────────────

template<typename BCConfig>
class Grid {
    int nx_, ny_, nz_;

    template<typename LoBC, typename HiBC>
    int Offset_Q(int i, int s, int n) const {
        if (i + s <  0) return QOffset<LoBC>(i, s, n);
        if (i + s >= n) return QOffset<HiBC>(i, s, n);
        return i + s;
    }

    template<typename LoBC, typename HiBC>
    int Offset_U(int i, int s, int n) const {
        if (i + s <  0) return UOffset<LoBC>(i, s, n);
        if (i + s >= n) return UOffset<HiBC>(i, s, n);
        return i + s;
    }

public:
    Grid(int nx, int ny, int nz) : nx_(nx), ny_(ny), nz_(nz) {}

    // Q-tensor stencil offsets
    int QXoff(int x, int s) const {
        return Offset_Q<typename BCConfig::XLo::QBC,
                        typename BCConfig::XHi::QBC>(x, s, nx_);
    }
    int QYoff(int y, int s) const {
        return Offset_Q<typename BCConfig::YLo::QBC,
                        typename BCConfig::YHi::QBC>(y, s, ny_);
    }
    int QZoff(int z, int s) const {
        return Offset_Q<typename BCConfig::ZLo::QBC,
                        typename BCConfig::ZHi::QBC>(z, s, nz_);
    }

    // Velocity streaming offsets
    int UXoff(int x, int s) const {
        return Offset_U<typename BCConfig::XLo::UBC,
                        typename BCConfig::XHi::UBC>(x, s, nx_);
    }
    int UYoff(int y, int s) const {
        return Offset_U<typename BCConfig::YLo::UBC,
                        typename BCConfig::YHi::UBC>(y, s, ny_);
    }
    int UZoff(int z, int s) const {
        return Offset_U<typename BCConfig::ZLo::UBC,
                        typename BCConfig::ZHi::UBC>(z, s, nz_);
    }

    static constexpr std::string_view GridType() { return BCConfig::name; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Convenience type alias — keeps existing code working unchanged
// ─────────────────────────────────────────────────────────────────────────────
using PeriodicGrid = Grid<FullyPeriodicConfig>;

#endif // LBM_AN_GRID_H_
