#ifndef LBM_AN_BOUNDARY_H_
#define LBM_AN_BOUNDARY_H_

#include <array>
#include <string_view>
#include <type_traits>

// ─────────────────────────────────────────────────────────────────────────────
// Boundary condition type tags
//
// Q-tensor wall types (control the FD stencil and HandleQBoundary):
//   Periodic          — periodic stencil, no HandleQBoundary action
//   Neumann           — zero-flux stencil (clamp ghost node), no HandleQBoundary
//   Anchoring<S,θ,ϕ>    — zero-flux stencil + HandleQBoundary overwrites with
//                       strong anchoring value (Q prescribed by S and angles {θ,ϕ})
//
// Velocity wall types (control streaming and HandleBoundaries):
//   Periodic             — periodic streaming, no HandleBoundaries action
//   NoSlip               — bounce-back (full reversal); u_wall = 0
//   MovingWall<Ux,Uy,Uz> — bounce-back with imposed wall velocity (Ux, Uy, Uz)
//   SpecularReflection— specular (mirror) reflection; ∂u_tangential/∂n = 0,
//                       u_normal → −u_normal; no-penetration with free slip
//
// Bounce-back vs. specular:
//   NoSlip / MovingWall: all velocity components reverse → wall halfway between
//                        grid node and ghost point (standard mid-point bounce-back).
//   SpecularReflection:  only normal velocity component reverses; tangential is
//                        preserved → free-slip (Neumann) for velocity.
// ─────────────────────────────────────────────────────────────────────────────

struct Periodic           {};
struct Neumann            {};
struct NoSlip             {};
struct SpecularReflection {};

// Strong Q anchoring: order parameter S at director angle (θ, ϕ) (radians).
template<double S = 1.0, double Theta = 0.0, double Phi = 0.0>
struct Anchoring {
    static constexpr double s     = S;
    static constexpr double theta = Theta;
    static constexpr double phi   = Phi;
};

// Moving-lid wall: imposed velocity (Ux, Uy) in lattice units.
template<double Ux_ = 0.0, double Uy_ = 0.0, double Uz_ = 0.0>
struct MovingWall {
    static constexpr double Ux = Ux_;
    static constexpr double Uy = Uy_;
    static constexpr double Uz = Uz_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Type traits for compile-time dispatch
// ─────────────────────────────────────────────────────────────────────────────

template<typename T> struct is_moving_wall_t : std::false_type {};
template<double Ux, double Uy, double Uz>
struct is_moving_wall_t<MovingWall<Ux, Uy, Uz>> : std::true_type {};
template<typename T>
inline constexpr bool is_moving_wall_v = is_moving_wall_t<T>::value;

template<typename T> struct is_anchoring_t : std::false_type {};
template<double S, double A, double B>
struct is_anchoring_t<Anchoring<S, A, B>> : std::true_type {};
template<typename T>
inline constexpr bool is_anchoring_v = is_anchoring_t<T>::value;

// ─────────────────────────────────────────────────────────────────────────────
// Per-wall BC bundle: Q stencil policy + velocity BC for one face
// ─────────────────────────────────────────────────────────────────────────────

template<typename QBC_, typename UBC_>
struct WallSpec {
    using QBC = QBC_;
    using UBC = UBC_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Full domain BC configuration — four faces, each independently specified
// ─────────────────────────────────────────────────────────────────────────────

template<
    typename XLo_ = WallSpec<Neumann, NoSlip>,
    typename XHi_ = WallSpec<Neumann, NoSlip>,
    typename YLo_ = WallSpec<Neumann, NoSlip>,
    typename YHi_ = WallSpec<Neumann, NoSlip>,
    typename ZLo_ = WallSpec<Neumann, NoSlip>,
    typename ZHi_ = WallSpec<Neumann, NoSlip>
>
struct BCConfig {
    using XLo = XLo_;
    using XHi = XHi_;
    using YLo = YLo_;
    using YHi = YHi_;
    using ZLo = ZLo_;
    using ZHi = ZHi_;
    static constexpr std::string_view name = "Custom";
};

// ─────────────────────────────────────────────────────────────────────────────
// Named presets
// ─────────────────────────────────────────────────────────────────────────────

// All four walls periodic (Q and velocity).
struct FullyPeriodicConfig {
    using XLo = WallSpec<Periodic, Periodic>;
    using XHi = WallSpec<Periodic, Periodic>;
    using YLo = WallSpec<Periodic, Periodic>;
    using YHi = WallSpec<Periodic, Periodic>;
    using ZLo = WallSpec<Periodic, Periodic>;
    using ZHi = WallSpec<Periodic, Periodic>;
    static constexpr std::string_view name = "FullyPeriodic";
};

// Periodic in X; no-slip walls (free Q anchoring) in Y.
struct ChannelConfig {
    using XLo = WallSpec<Neumann, NoSlip>;
    using XHi = WallSpec<Neumann, NoSlip>;
    using YLo = WallSpec<Neumann, NoSlip>;
    using YHi = WallSpec<Neumann, NoSlip>;
    using ZLo = WallSpec<Periodic, Periodic>;
    using ZHi = WallSpec<Periodic, Periodic>;
    static constexpr std::string_view name = "Channel";
};

// ─────────────────────────────────────────────────────────────────────────────
// Backward-compat alias — existing code using PeriodicBC keeps working
// ─────────────────────────────────────────────────────────────────────────────
using PeriodicBC = FullyPeriodicConfig;

template <typename BCConfig>
inline constexpr std::array<int, 3> periodicity_by_axis = {
    static_cast<int>(std::is_same_v<typename BCConfig::XLo::UBC, Periodic>),
    static_cast<int>(std::is_same_v<typename BCConfig::YLo::UBC, Periodic>),
    static_cast<int>(std::is_same_v<typename BCConfig::ZLo::UBC, Periodic>),
};

// True on faces where the UBC is a physical wall (i.e., not Periodic). Order:
// XLo, XHi, YLo, YHi, ZLo, ZHi. Used by HaloExchangeLBM to skip crossing-dir
// unpacks at cells where a local wall bounce already wrote the same dir slot.
template <typename BCConfig>
inline constexpr std::array<bool, 6> is_wall_by_face = {
    !std::is_same_v<typename BCConfig::XLo::UBC, Periodic>,
    !std::is_same_v<typename BCConfig::XHi::UBC, Periodic>,
    !std::is_same_v<typename BCConfig::YLo::UBC, Periodic>,
    !std::is_same_v<typename BCConfig::YHi::UBC, Periodic>,
    !std::is_same_v<typename BCConfig::ZLo::UBC, Periodic>,
    !std::is_same_v<typename BCConfig::ZHi::UBC, Periodic>,
};

#endif // LBM_AN_BOUNDARY_H_
