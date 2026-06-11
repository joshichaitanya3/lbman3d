#ifndef LBM_AN_ANALYSIS_DEFECT_CURVE_ROUTINES_H
#define LBM_AN_ANALYSIS_DEFECT_CURVE_ROUTINES_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "disclination.h"

// -- Smoothing function

bool IsLoop(const std::vector<double> points) {
    bool is_loop = false;
    int l = points.size();
    if (l<9) { // Less than 3 points: cannot be a loop
        return is_loop;
    }
    if (abs(points[0]-points[l-3])<0.25 &&
        abs(points[1]-points[l-2])<0.25 &&
        abs(points[2]-points[l-1])<0.25
    ) {
        is_loop = true;
    }
    return is_loop;
}

std::vector<double> Smoothen(const std::vector<double>& points, bool is_loop, int stencil=3) {
    int l = points.size();

    if (stencil % 2 == 0 || stencil < 3)
        throw std::invalid_argument("stencil must be an odd number >= 3");

    const int n = static_cast<int>(points.size() / 3);  // number of 3-D points
    if (n < stencil)
        return points; // No need of smoothing for small loop

    const int half = stencil / 2; // e.g. 1 for 3-pt, 2 for 5-pt
    const double w  = 1.0 / stencil; // uniform weight

    std::vector<double> out(points.size());

    for (int i = 0; i < n; ++i) {
        double sx = 0, sy = 0, sz = 0;

        for (int k = -half; k <= half; ++k) {
            int j = i + k;

            if (is_loop) {
                j = ((j % n) + n) % n; // wrap-around (handles negative %)
            } else {
                j = std::clamp(j, 0, n - 1); // clamp / replicate-border
            }

            sx += points[3*j    ];
            sy += points[3*j + 1];
            sz += points[3*j + 2];
        }

        out[3*i    ] = sx * w;
        out[3*i + 1] = sy * w;
        out[3*i + 2] = sz * w;
    }

    if (is_loop) {
        // Restore the closing duplicate: last point == first point
        out[l-3] = out[0];
        out[l-2] = out[1];
        out[l-1] = out[2];
    }
    return out;
}


// ── Tiny 3-vector helper ─────────────────────────────────────────────────────
struct Vec3 { double x, y, z; };

// ── Result type ──────────────────────────────────────────────────────────────
struct SplineResult {
    std::vector<double> positions;  // smoothed {x,y,z, ...} at original t values
    std::vector<double> tangents;   // unit tangents  {tx,ty,tz, ...}
};

// ── Solve a tridiagonal system Ax = rhs in-place (Thomas algorithm) ──────────
static void SolveTriDiagonal(const std::vector<double>& lower,
                                    std::vector<double>  diag,   // copy — modified
                              const std::vector<double>& upper,
                                    std::vector<double>& rhs)
{
    const int n = static_cast<int>(diag.size());
    // Forward sweep
    for (int i = 1; i < n; ++i) {
        double m = lower[i] / diag[i-1];
        diag[i] -= m * upper[i-1];
        rhs [i] -= m * rhs  [i-1];
    }
    // Back substitution
    rhs[n-1] /= diag[n-1];
    for (int i = n-2; i >= 0; --i)
        rhs[i] = (rhs[i] - upper[i] * rhs[i+1]) / diag[i];
}

// ── Fit a natural cubic spline to (t, v) and return {S, dS/dt} at t[] ────────
//    'is_loop': if true, uses periodic boundary conditions
static void CubicSpline1D(const std::vector<double>& t,
                          const std::vector<double>& v,
                          bool is_loop,
                          std::vector<double>& pos_out,
                          std::vector<double>& tan_out)
{
    const int n = static_cast<int>(t.size());

    // Interval widths
    std::vector<double> h(n - 1);
    for (int i = 0; i < n - 1; ++i)
        h[i] = t[i+1] - t[i];

    // ── Build and solve for second derivatives (moments) M[] ─────────────
    std::vector<double> M(n, 0.0);

    if (!is_loop) {
        // Natural spline: M[0] = M[n-1] = 0 (already set)
        // Interior equations only
        const int m = n - 2;
        if (m > 0) {
            std::vector<double> lo(m), diag(m), up(m), rhs(m);
            for (int i = 0; i < m; ++i) {
                int r = i + 1;                            // interior point index
                lo  [i] = h[r-1];
                diag[i] = 2.0 * (h[r-1] + h[r]);
                up  [i] = h[r];
                rhs [i] = 6.0 * ((v[r+1]-v[r])/h[r] - (v[r]-v[r-1])/h[r-1]);
            }
            SolveTriDiagonal(lo, diag, up, rhs);
            for (int i = 0; i < m; ++i) M[i+1] = rhs[i];
        }
    } else {
        // Periodic spline: M[0] == M[n-1] (the duplicate endpoint)
        // Solve for M[0..n-2] with wrap-around — use Sherman-Morrison
        // For small n (<100) a direct dense solve is fine, but we keep it
        // tridiagonal + rank-1 via Sherman-Morrison for correctness.
        const int m = n - 1;                              // unique points

        // Build the cyclic tridiagonal RHS
        std::vector<double> rhs(m);
        for (int i = 0; i < m; ++i) {
            int prev = (i - 1 + m) % m;
            int next = (i + 1    ) % m;
            rhs[i] = 6.0 * ((v[next]-v[i])/h[i] - (v[i]-v[prev])/h[prev]);
        }

        // Sherman-Morrison: solve (T + u*vT) x = rhs
        // where T is the plain tridiagonal and u,v handle the corners
        const double corner = h[m-1];
        const double gamma  = -2.0 * (h[m-1] + h[0]);   // T[0,0] shift

        std::vector<double> diag(m), lo(m), up(m);
        for (int i = 0; i < m; ++i) {
            lo  [i] = h[(i - 1 + m) % m];
            diag[i] = 2.0 * (h[(i-1+m)%m] + h[i]);
            up  [i] = h[i];
        }
        // Modify corners for Sherman-Morrison
        diag[0]   -= gamma;
        diag[m-1] -= corner * corner / gamma;

        // Solve T' * q = rhs  and  T' * u_vec = e_u
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
        M[n-1] = M[0];                                   // periodic
    }

    // ── Evaluate position S(t) and derivative S'(t) at each knot ─────────
    pos_out.resize(n);
    tan_out.resize(n);

    // For each interior of segment [i, i+1], evaluate at t[i]
    for (int i = 0; i < n - 1; ++i) {
        double hi = h[i];
        // Cubic Hermite form derived from the moments:
        //   S (t) = A*v[i] + B*v[i+1] + C*M[i] + D*M[i+1]
        //   S'(t) at left endpoint (dt=0):
        double dS = (v[i+1] - v[i]) / hi
                  - hi / 6.0 * (2.0 * M[i] + M[i+1]);

        pos_out[i] = v[i];
        tan_out[i] = dS;
    }
    // Last point: evaluate using the last segment from the right
    {
        int i = n - 2;
        double hi = h[i];
        double dS = (v[i+1] - v[i]) / hi
                  + hi / 6.0 * (M[i] + 2.0 * M[i+1]);
        pos_out[n-1] = v[n-1];
        tan_out[n-1] = dS;
    }
}

// ── Main entry point ─────────────────────────────────────────────────────────
SplineResult FitArcLengthSpline(const std::vector<double>& points,
                                bool is_loop = false)
{
    const int nTotal = static_cast<int>(points.size() / 3);
    const int n      = is_loop ? nTotal - 1 : nTotal;    // unique points

    if (n < 2)
        throw std::invalid_argument("need at least 2 unique points");

    // ── 1. Chord-length parameter ─────────────────────────────────────────
    std::vector<double> t(n);
    t[0] = 0.0;
    for (int i = 1; i < n; ++i) {
        double dx = points[3*i    ] - points[3*(i-1)    ];
        double dy = points[3*i + 1] - points[3*(i-1) + 1];
        double dz = points[3*i + 2] - points[3*(i-1) + 2];
        t[i] = t[i-1] + std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    if (is_loop) {
        // Add the closing chord back to first point
        double dx = points[0] - points[3*(n-1)    ];
        double dy = points[1] - points[3*(n-1) + 1];
        double dz = points[2] - points[3*(n-1) + 2];
        t.push_back(t[n-1] + std::sqrt(dx*dx + dy*dy + dz*dz));
        // t now has n+1 entries matching nTotal points including duplicate
    }

    // Separate x, y, z
    std::vector<double> vx(t.size()), vy(t.size()), vz(t.size());
    for (int i = 0; i < nTotal; ++i) {          // include duplicate if is_loop
        vx[i] = points[3*i    ];
        vy[i] = points[3*i + 1];
        vz[i] = points[3*i + 2];
    }

    // ── 2. Fit spline per component ───────────────────────────────────────
    std::vector<double> sx, sy, sz, dtx, dty, dtz;
    CubicSpline1D(t, vx, is_loop, sx, dtx);
    CubicSpline1D(t, vy, is_loop, sy, dty);
    CubicSpline1D(t, vz, is_loop, sz, dtz);

    // ── 3. Pack results ───────────────────────────────────────────────────
    SplineResult result;
    result.positions.resize(3 * nTotal);
    result.tangents .resize(3 * nTotal);

    for (int i = 0; i < nTotal; ++i) {
        result.positions[3*i    ] = sx[i];
        result.positions[3*i + 1] = sy[i];
        result.positions[3*i + 2] = sz[i];

        // Normalise tangent to unit length
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
