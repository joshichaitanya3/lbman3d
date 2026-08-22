#ifndef LBM_AN_ANALYSIS_DEFECT_PERIODICITY_H
#define LBM_AN_ANALYSIS_DEFECT_PERIODICITY_H

#include <array>
#include <type_traits>
#include <params.h>
#include "boundary.h"

// ─────────────────────────────────────────────────────────────────────────────
// BC-aware wrap and lookup helpers for defect-face arrays.
//
// The defect finder needs to (a) iterate plaquettes with per-axis-appropriate
// extents (n if periodic on that axis, n-1 otherwise), (b) read a director
// value at a vertex "next to" the current one where "next" may wrap under
// periodic BC or fall on a wall under Neumann, and (c) look up a face-defect
// flag at an index that may sit one step beyond the family's extent under
// periodic BC (a valid seam-crossing plaquette).
//
// These helpers keep the wrap logic in one place. The MPI-parallel version
// of defect detection will replace the modular arithmetic here with halo-
// buffer reads (via the same seam that boundary_handler.h uses for Q ghosts);
// nothing else in the finder will change. That's the reason these functions
// stay separate from the ones in offsets.h — offsets.h operates on the full
// domain via Params::n{x,y,z}, while these operate on face-family extents
// that additionally shrink by 1 on non-periodic axes.
// ─────────────────────────────────────────────────────────────────────────────

// True if the low-x face BC is Periodic (both low- and high-x must match; the
// physics rejects mixed Periodic/Neumann on a single axis).
template<typename BC> inline constexpr bool kPX = std::is_same_v<typename BC::XLo::QBC, Periodic>;
template<typename BC> inline constexpr bool kPY = std::is_same_v<typename BC::YLo::QBC, Periodic>;
template<typename BC> inline constexpr bool kPZ = std::is_same_v<typename BC::ZLo::QBC, Periodic>;

// Wrap a director-lookup index into [0, n) under periodic BC. Under Neumann
// the caller is guaranteed by loop bounds to stay in range, but we clamp
// defensively anyway.
inline int WrapDirIdx(int i, int n, bool periodic) {
    if (periodic) return ((i % n) + n) % n;
    if (i < 0) return 0;
    if (i >= n) return n - 1;
    return i;
}

// Wrap a face-family index into [0, extent). Returns -1 if the index sits
// outside the family (a plaquette past a walled boundary), signalling to the
// caller that no such face exists. Under periodic BC this never returns -1;
// under wall BC it does whenever i is out of the face-family's [0, extent).
inline int WrapFaceIdx(int i, int extent, bool periodic) {
    if (i >= 0 && i < extent) return i;
    if (!periodic) return -1;
    return ((i % extent) + extent) % extent;
}

#endif // LBM_AN_ANALYSIS_DEFECT_PERIODICITY_H
