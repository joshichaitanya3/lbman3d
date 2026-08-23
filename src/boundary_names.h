#ifndef LBM_AN_BOUNDARY_NAMES_H_
#define LBM_AN_BOUNDARY_NAMES_H_

#include <string>
#include <type_traits>
#include "boundary.h"
#include "format_compat.h"

// ─────────────────────────────────────────────────────────────────────────────
// Canonical string names for BC tags — the authoritative wire representation
// stamped onto VTKHDF frames so the analysis binary can validate that its
// compile-time SimBC matches the run's BC on a per-face-slot basis.
//
// SimBC::name (from boundary.h) is a human label only; two different configs
// can share a name and any `BCConfig<...>` custom collapses to "Custom".
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
struct BCName;   // primary undefined — specialize per tag or fail to compile

template<> struct BCName<Periodic>           { static std::string get() { return "Periodic"; } };
template<> struct BCName<Neumann>            { static std::string get() { return "Neumann"; } };
template<> struct BCName<NoSlip>             { static std::string get() { return "NoSlip"; } };
template<> struct BCName<SpecularReflection> { static std::string get() { return "SpecularReflection"; } };

// `{:.17g}` pins the double-to-string conversion so the wire string depends
// only on the double's bit pattern, not on which formatter (std::format vs
// {fmt}, or which version) built the binary. 
template<double S, double Th, double Ph>
struct BCName<Anchoring<S, Th, Ph>> {
    static std::string get() {
        return compat::format("Anchoring(S={:.17g},theta={:.17g},phi={:.17g})",
                              S, Th, Ph);
    }
};

template<double Ux, double Uy, double Uz>
struct BCName<MovingWall<Ux, Uy, Uz>> {
    static std::string get() {
        return compat::format("MovingWall(Ux={:.17g},Uy={:.17g},Uz={:.17g})",
                              Ux, Uy, Uz);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// QBC/UBC periodicity invariant.
//
// A single axis mixing Periodic on one slot and non-Periodic on the other
// silently violates assumptions in the finder / analyzer (DefectFields sizes
// off UBC-periodicity at boundary.h; the finder walks QBC-periodicity at
// defect_periodicity.h; these must agree). This static_assert enforces the
// invariant at compile time.
//
// A future concepts-based refactor of WallSpec/BCConfig would make the
// mixed state uncompilable structurally, at which point this static_assert
// becomes redundant and can be removed. Until then, it holds the line.
// ─────────────────────────────────────────────────────────────────────────────

template<typename Wall>
inline constexpr bool wall_periodicity_matches_v =
    std::is_same_v<typename Wall::QBC, Periodic> ==
    std::is_same_v<typename Wall::UBC, Periodic>;

template<typename BC>
inline constexpr bool bc_periodicity_consistent_v =
    wall_periodicity_matches_v<typename BC::XLo> &&
    wall_periodicity_matches_v<typename BC::XHi> &&
    wall_periodicity_matches_v<typename BC::YLo> &&
    wall_periodicity_matches_v<typename BC::YHi> &&
    wall_periodicity_matches_v<typename BC::ZLo> &&
    wall_periodicity_matches_v<typename BC::ZHi>;

#endif // LBM_AN_BOUNDARY_NAMES_H_
