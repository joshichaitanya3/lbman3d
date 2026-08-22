#ifndef LBM_AN_ANALYSIS_DEFECT_CURVE_ROUTINES_H
#define LBM_AN_ANALYSIS_DEFECT_CURVE_ROUTINES_H

#include <array>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include "disclination.h"
#include "physics_helpers.h"

// ─────────────────────────────────────────────────────────────────────────────
// Curve routines for disclination lines
//
// Two conventions matter throughout:
//
// 1. Loop lines have a duplicated closing point in `points`/`smooth_points`
//    (path[0] == path[last]). Cross-boundary loops arrive here with
//    smooth_points[last] == smooth_points[0] + period_offset (nonzero period-
//    winding along one or more axes).
//
// 2. Smoothing and spline fitting operate on **unwrapped** unique points —
//    the closing duplicate is dropped, minimum-image unwrap is applied so
//    consecutive points are geometrically continuous, and after the periodic
//    smoothing/spline pass we re-append the closing duplicate at
//    unique[0] + period_offset. This keeps the smoothing kernel oblivious to
//    the seam and lets FitArcLengthSpline's chord-length parametrization use
//    the true unwrapped distances (crucial: a raw jump across the seam would
//    contribute one lattice-diameter chord and destroy the spline).
// ─────────────────────────────────────────────────────────────────────────────

// Detect whether a raw traversal (wrapped image) closes on itself. The graph
// traversal writes the starting face at the end of a loop, so a physical
// closure is signalled by points[0] == points[last] within a small tolerance
// (a lattice spacing to allow floating-point noise). Cross-boundary loops
// also satisfy this test in wrapped coordinates.
inline bool IsLoop(const std::vector<double>& points) {
    const int l = static_cast<int>(points.size());
    if (l < 9) return false; // fewer than 3 points cannot be a loop
    return std::abs(points[0] - points[l-3]) < 0.25 &&
           std::abs(points[1] - points[l-2]) < 0.25 &&
           std::abs(points[2] - points[l-1]) < 0.25;
}

// Minimum-image unwrap along each periodic axis. `periods[a]` = 0 means axis
// `a` is walled and no unwrap is performed on that axis; otherwise it is the
// physical box size along axis a in the same units as `points`. Consecutive
// jumps larger than half the period are folded by adding/subtracting integer
// periods so the returned sequence is geometrically continuous.
inline std::vector<double> MinimumImageUnwrap(const std::vector<double>& points,
                                              const std::array<double, 3>& periods) {
    const int n = static_cast<int>(points.size() / 3);
    std::vector<double> out(points);
    if (n < 2) return out;

    for (int i = 1; i < n; ++i) {
        for (int a = 0; a < 3; ++a) {
            const double p = periods[a];
            if (p <= 0.0) continue;
            double d = out[3*i + a] - out[3*(i-1) + a];
            // Fold d into (-p/2, p/2].
            const double k = std::round(d / p);
            if (k != 0.0) {
                out[3*i + a] -= k * p;
            }
        }
    }
    return out;
}

// Compute the period-winding offset of a closed line: the offset the last
// (duplicated) unwrapped point picks up relative to the first, expressed as
// integer multiples of `periods`. Returns (0,0,0) for an in-box loop, and
// (±period) on the axes it winds across.
inline std::array<double, 3> WindingOffset(const std::vector<double>& raw_points,
                                           const std::vector<double>& unwrapped_points) {
    std::array<double, 3> off = {0.0, 0.0, 0.0};
    const int n = static_cast<int>(unwrapped_points.size() / 3);
    if (n < 2) return off;
    for (int a = 0; a < 3; ++a) {
        // raw last == raw first (loop convention), so any residual difference
        // in unwrapped coords is the winding offset.
        off[a] = unwrapped_points[3*(n-1) + a] - raw_points[3*0 + a];
    }
    return off;
}

// Uniform-weight moving-average smoothing. `is_loop` == true treats the input
// as cyclic under mod n; the caller is expected to have dropped any closing
// duplicate before calling and to reappend it (with any period offset) after.
// stencil must be odd and >= 3.
inline std::vector<double> Smoothen(const std::vector<double>& points,
                                    bool is_loop, int stencil = 3) {
    if (stencil % 2 == 0 || stencil < 3)
        throw std::invalid_argument("stencil must be an odd number >= 3");

    const int n = static_cast<int>(points.size() / 3);
    if (n < stencil) return points; // too short to smooth meaningfully

    const int half = stencil / 2;
    const double w = 1.0 / stencil;

    std::vector<double> out(points.size());

    for (int i = 0; i < n; ++i) {
        double sx = 0, sy = 0, sz = 0;
        for (int k = -half; k <= half; ++k) {
            int j = i + k;
            if (is_loop) j = ((j % n) + n) % n; // wrap
            else         j = std::clamp(j, 0, n - 1);
            sx += points[3*j    ];
            sy += points[3*j + 1];
            sz += points[3*j + 2];
        }
        out[3*i    ] = sx * w;
        out[3*i + 1] = sy * w;
        out[3*i + 2] = sz * w;
    }

    return out;
}

// ── Result type ─────────────────────────────────────────────────────────────
struct SplineResult {
    std::vector<double> positions;
    std::vector<double> tangents;
};

// ── Thomas algorithm ────────────────────────────────────────────────────────
static void SolveTriDiagonal(const std::vector<double>& lower,
                                    std::vector<double>  diag,
                              const std::vector<double>& upper,
                                    std::vector<double>& rhs) {
    const int n = static_cast<int>(diag.size());
    for (int i = 1; i < n; ++i) {
        double m = lower[i] / diag[i-1];
        diag[i] -= m * upper[i-1];
        rhs [i] -= m * rhs  [i-1];
    }
    rhs[n-1] /= diag[n-1];
    for (int i = n-2; i >= 0; --i)
        rhs[i] = (rhs[i] - upper[i] * rhs[i+1]) / diag[i];
}

// Natural or periodic cubic spline through (t[i], v[i]). Writes position and
// tangent samples at every knot into pos_out / tan_out.
static void CubicSpline1D(const std::vector<double>& t,
                          const std::vector<double>& v,
                          bool is_loop,
                          std::vector<double>& pos_out,
                          std::vector<double>& tan_out) {
    const int n = static_cast<int>(t.size());

    std::vector<double> h(n - 1);
    for (int i = 0; i < n - 1; ++i)
        h[i] = t[i+1] - t[i];

    std::vector<double> M(n, 0.0);

    if (!is_loop) {
        const int m = n - 2;
        if (m > 0) {
            std::vector<double> lo(m), diag(m), up(m), rhs(m);
            for (int i = 0; i < m; ++i) {
                int r = i + 1;
                lo  [i] = h[r-1];
                diag[i] = 2.0 * (h[r-1] + h[r]);
                up  [i] = h[r];
                rhs [i] = 6.0 * ((v[r+1]-v[r])/h[r] - (v[r]-v[r-1])/h[r-1]);
            }
            SolveTriDiagonal(lo, diag, up, rhs);
            for (int i = 0; i < m; ++i) M[i+1] = rhs[i];
        }
    } else {
        // Periodic: M[0] == M[n-1]. Solve on unique nodes m = n-1 via
        // Sherman-Morrison over the cyclic tridiagonal system.
        const int m = n - 1;
        std::vector<double> rhs(m);
        for (int i = 0; i < m; ++i) {
            int prev = (i - 1 + m) % m;
            int next = (i + 1    ) % m;
            rhs[i] = 6.0 * ((v[next]-v[i])/h[i] - (v[i]-v[prev])/h[prev]);
        }
        const double corner = h[m-1];
        const double gamma  = -2.0 * (h[m-1] + h[0]);
        std::vector<double> diag(m), lo(m), up(m);
        for (int i = 0; i < m; ++i) {
            lo  [i] = h[(i - 1 + m) % m];
            diag[i] = 2.0 * (h[(i-1+m)%m] + h[i]);
            up  [i] = h[i];
        }
        diag[0]   -= gamma;
        diag[m-1] -= corner * corner / gamma;
        std::vector<double> q   = rhs;
        std::vector<double> uvec(m, 0.0);
        uvec[0]   =  gamma;
        uvec[m-1] =  corner;
        SolveTriDiagonal(lo, diag, up, q);
        SolveTriDiagonal(lo, diag, up, uvec);
        double vTq = q[0] + (corner / gamma) * q[m-1];
        double vTu = uvec[0] + (corner / gamma) * uvec[m-1];
        for (int i = 0; i < m; ++i)
            M[i] = q[i] - (vTq / (1.0 + vTu)) * uvec[i];
        M[n-1] = M[0];
    }

    pos_out.resize(n);
    tan_out.resize(n);

    for (int i = 0; i < n - 1; ++i) {
        double hi = h[i];
        double dS = (v[i+1] - v[i]) / hi - hi / 6.0 * (2.0 * M[i] + M[i+1]);
        pos_out[i] = v[i];
        tan_out[i] = dS;
    }
    {
        int i = n - 2;
        double hi = h[i];
        double dS = (v[i+1] - v[i]) / hi + hi / 6.0 * (M[i] + 2.0 * M[i+1]);
        pos_out[n-1] = v[n-1];
        tan_out[n-1] = dS;
    }
}

// Fit a chord-length-parametrised cubic spline through the (unwrapped) input
// points and return positions + unit tangents at every input knot.
// `is_loop == true` expects the closing duplicate to be present in `points`,
// matching the SmoothenDisclination convention; the periodic spline is fit
// on the unique nodes and evaluated at the duplicate as well.
inline SplineResult FitArcLengthSpline(const std::vector<double>& points,
                                       bool is_loop = false) {
    const int nTotal = static_cast<int>(points.size() / 3);
    const int n      = is_loop ? nTotal - 1 : nTotal;

    if (n < 2)
        throw std::invalid_argument("need at least 2 unique points");

    std::vector<double> t(n);
    t[0] = 0.0;
    for (int i = 1; i < n; ++i) {
        double dx = points[3*i    ] - points[3*(i-1)    ];
        double dy = points[3*i + 1] - points[3*(i-1) + 1];
        double dz = points[3*i + 2] - points[3*(i-1) + 2];
        t[i] = t[i-1] + std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    if (is_loop) {
        double dx = points[3*n    ] - points[3*(n-1)    ];
        double dy = points[3*n + 1] - points[3*(n-1) + 1];
        double dz = points[3*n + 2] - points[3*(n-1) + 2];
        t.push_back(t[n-1] + std::sqrt(dx*dx + dy*dy + dz*dz));
    }

    std::vector<double> vx(t.size()), vy(t.size()), vz(t.size());
    for (int i = 0; i < nTotal; ++i) {
        vx[i] = points[3*i    ];
        vy[i] = points[3*i + 1];
        vz[i] = points[3*i + 2];
    }

    std::vector<double> sx, sy, sz, dtx, dty, dtz;
    CubicSpline1D(t, vx, is_loop, sx, dtx);
    CubicSpline1D(t, vy, is_loop, sy, dty);
    CubicSpline1D(t, vz, is_loop, sz, dtz);

    SplineResult result;
    result.positions.resize(3 * nTotal);
    result.tangents .resize(3 * nTotal);

    for (int i = 0; i < nTotal; ++i) {
        result.positions[3*i    ] = sx[i];
        result.positions[3*i + 1] = sy[i];
        result.positions[3*i + 2] = sz[i];
        double tx = dtx[i], ty = dty[i], tz = dtz[i];
        double len = std::sqrt(tx*tx + ty*ty + tz*tz);
        if (len > 1e-12) { tx /= len; ty /= len; tz /= len; }
        result.tangents[3*i    ] = tx;
        result.tangents[3*i + 1] = ty;
        result.tangents[3*i + 2] = tz;
    }
    return result;
}

#endif // LBM_AN_ANALYSIS_DEFECT_CURVE_ROUTINES_H
