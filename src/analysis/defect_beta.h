#ifndef LBM_AN_ANALYSIS_DEFECT_BETA_H
#define LBM_AN_ANALYSIS_DEFECT_BETA_H

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>
#include <params.h>
#include "qtensor_fields.h"
#include "analysis_fields.h"
#include "offsets.h"
#include "boundary.h"
#include "physics_helpers.h"
#include "defect_analysis_config.h"

using namespace Params;

// ─────────────────────────────────────────────────────────────────────────────
// β pipeline: at each smoothed disclination vertex, interpolate Q on a polar
// ring in the plane ⊥ T̂, sequentially sign-align the ring directors, fit the
// rotation axis Ω as the smallest-eigenvalue eigenvector of the director
// Gram matrix, and set β = arccos(|T̂·Ω̂|). Follows the Nematics3D pipeline
// (see the lab-knowledge concepts under disclination-geometry/), with the
// ring radius R driven by the defect core length ξ computed from Params.
// ─────────────────────────────────────────────────────────────────────────────

namespace DefectAnalysis {

// ξ, R, arc-distance all live in params.h so users can override them in a
// shadow params.h (the standard sweep/test mechanism); see the constants
// under `namespace DefectAnalysis` there.


// ── Trilinear Q interpolation, BC-aware ─────────────────────────────────────
// Physical position → lattice-unit position, wrapping / clamping under BC.
template<typename BC>
struct QSample { double xx, xy, xz, yy, yz; };

// Reduce a lattice-unit coordinate to (i0, f): the floor-integer part and the
// [0, 1) fractional. Wrapping into the axis's valid range happens per-neighbour
// via QXoff/QYoff/QZoff.
inline void SplitCoord(double u, int& i0, double& f) {
    double base = std::floor(u);
    i0 = static_cast<int>(base);
    f  = u - base;
}

template<typename BC>
inline QSample<BC> InterpolateQ(const QTensorFields& qf,
                                double px, double py, double pz) {
    const LocalGrid& g = qf.grid;

    // Physical → lattice units.
    const double u = px / DX;
    const double v = py / DY;
    const double w = pz / DZ;

    int i0, j0, k0;
    double fx, fy, fz;
    SplitCoord(u, i0, fx);
    SplitCoord(v, j0, fy);
    SplitCoord(w, k0, fz);

    // Fold i0 into a valid vertex index for THIS axis before offsetting;
    // QXoff/QYoff/QZoff then produce a valid neighbour under BC.
    const int i0w = QXoff<BC>(i0, 0);
    const int j0w = QYoff<BC>(j0, 0);
    const int k0w = QZoff<BC>(k0, 0);

    const int i1 = QXoff<BC>(i0w, 1);
    const int j1 = QYoff<BC>(j0w, 1);
    const int k1 = QZoff<BC>(k0w, 1);

    auto sample = [&](const std::vector<double>& q) -> double {
        const double c000 = q[g.halo_idx(i0w, j0w, k0w)];
        const double c100 = q[g.halo_idx(i1,  j0w, k0w)];
        const double c010 = q[g.halo_idx(i0w, j1,  k0w)];
        const double c110 = q[g.halo_idx(i1,  j1,  k0w)];
        const double c001 = q[g.halo_idx(i0w, j0w, k1 )];
        const double c101 = q[g.halo_idx(i1,  j0w, k1 )];
        const double c011 = q[g.halo_idx(i0w, j1,  k1 )];
        const double c111 = q[g.halo_idx(i1,  j1,  k1 )];
        const double c00 = c000 * (1.0 - fx) + c100 * fx;
        const double c01 = c001 * (1.0 - fx) + c101 * fx;
        const double c10 = c010 * (1.0 - fx) + c110 * fx;
        const double c11 = c011 * (1.0 - fx) + c111 * fx;
        const double c0 = c00 * (1.0 - fy) + c10 * fy;
        const double c1 = c01 * (1.0 - fy) + c11 * fy;
        return c0 * (1.0 - fz) + c1 * fz;
    };

    return {
        sample(qf.qxx),
        sample(qf.qxy),
        sample(qf.qxz),
        sample(qf.qyy),
        sample(qf.qyz),
    };
}


// ── Ring construction in the plane ⊥ T̂ ──────────────────────────────────────
struct Vec3d { double x, y, z; };

inline Vec3d Cross(Vec3d a, Vec3d b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline double DotV(Vec3d a, Vec3d b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline double NormV(Vec3d a) { return std::sqrt(DotV(a, a)); }
inline Vec3d NormalizeV(Vec3d a) {
    const double n = NormV(a);
    return n > 1e-14 ? Vec3d{a.x/n, a.y/n, a.z/n} : Vec3d{0, 0, 0};
}

// Build an orthonormal basis (u, v) in the plane perpendicular to unit t.
inline void PlaneBasis(Vec3d t, Vec3d& u, Vec3d& v) {
    // Pick a reference axis not (nearly) parallel to t.
    Vec3d ref = std::abs(t.x) < 0.9 ? Vec3d{1, 0, 0} : Vec3d{0, 1, 0};
    u = NormalizeV(Cross(t, ref));
    v = NormalizeV(Cross(t, u));
}


// ── Small 3x3 symmetric eigendecomposition (Jacobi) ─────────────────────────
// Returns eigenvalues in ascending order and eigenvectors as columns.
inline void JacobiEig3(double M[3][3], double eval[3], double evec[3][3]) {
    // Initialize evec = I.
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            evec[i][j] = (i == j) ? 1.0 : 0.0;

    // Copy M into A; we'll rotate A toward diagonal.
    double A[3][3] = {
        {M[0][0], M[0][1], M[0][2]},
        {M[1][0], M[1][1], M[1][2]},
        {M[2][0], M[2][1], M[2][2]}
    };

    for (int iter = 0; iter < 40; ++iter) {
        // Find largest off-diagonal element.
        int p = 0, q = 1;
        double max_off = std::abs(A[0][1]);
        if (std::abs(A[0][2]) > max_off) { p = 0; q = 2; max_off = std::abs(A[0][2]); }
        if (std::abs(A[1][2]) > max_off) { p = 1; q = 2; max_off = std::abs(A[1][2]); }
        if (max_off < 1e-14) break;

        double app = A[p][p], aqq = A[q][q], apq = A[p][q];
        double theta = 0.5 * (aqq - app) / apq;
        double t = (theta >= 0)
            ? 1.0 / (theta + std::sqrt(1.0 + theta * theta))
            : 1.0 / (theta - std::sqrt(1.0 + theta * theta));
        double c = 1.0 / std::sqrt(1.0 + t * t);
        double s = t * c;

        A[p][p] = app - t * apq;
        A[q][q] = aqq + t * apq;
        A[p][q] = A[q][p] = 0.0;
        for (int i = 0; i < 3; ++i) {
            if (i != p && i != q) {
                double aip = A[i][p], aiq = A[i][q];
                A[i][p] = A[p][i] = c * aip - s * aiq;
                A[i][q] = A[q][i] = s * aip + c * aiq;
            }
        }
        for (int i = 0; i < 3; ++i) {
            double vip = evec[i][p], viq = evec[i][q];
            evec[i][p] = c * vip - s * viq;
            evec[i][q] = s * vip + c * viq;
        }
    }

    // Sort eigenvalues ascending, permute eigenvectors accordingly.
    for (int i = 0; i < 3; ++i) eval[i] = A[i][i];
    for (int i = 0; i < 2; ++i) {
        int mn = i;
        for (int j = i+1; j < 3; ++j) if (eval[j] < eval[mn]) mn = j;
        if (mn != i) {
            std::swap(eval[i], eval[mn]);
            for (int r = 0; r < 3; ++r) std::swap(evec[r][i], evec[r][mn]);
        }
    }
}


// ── Director from 5-component Q ─────────────────────────────────────────────
inline Vec3d DirectorFromQ(double xx, double xy, double xz, double yy, double yz) {
    const auto od = QtensorToOrderDirectorPoint(xx, xy, xz, yy, yz);
    return {od.nx, od.ny, od.nz};
}


// ── Ω on a ring around one point on the line ────────────────────────────────
struct OmegaResult {
    Vec3d omega;
    double rotation_consistency; // in [0, 1]; low → fit is not a rotation
    bool valid;
};

template<typename BC>
inline OmegaResult OmegaOnRing(const QTensorFields& qf,
                               Vec3d center, Vec3d tangent, double R) {
    Vec3d u, v;
    PlaneBasis(tangent, u, v);

    // Arc-uniform sampling with a floor of 6 points so the fit is stable.
    int n_theta = static_cast<int>(std::round(2.0 * M_PI * R / kOmegaRingArcDist));
    if (n_theta < 6) n_theta = 6;

    std::vector<Vec3d> directors;
    directors.reserve(static_cast<size_t>(n_theta));

    for (int k = 0; k < n_theta; ++k) {
        const double theta = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(n_theta);
        const double ct = std::cos(theta), st = std::sin(theta);
        const double px = center.x + R * (ct * u.x + st * v.x);
        const double py = center.y + R * (ct * u.y + st * v.y);
        const double pz = center.z + R * (ct * u.z + st * v.z);
        const auto qs = InterpolateQ<BC>(qf, px, py, pz);
        directors.push_back(DirectorFromQ(qs.xx, qs.xy, qs.xz, qs.yy, qs.yz));
    }

    // Sequential sign alignment around the ring.
    for (size_t i = 1; i < directors.size(); ++i) {
        if (DotV(directors[i-1], directors[i]) < 0.0) {
            directors[i].x = -directors[i].x;
            directors[i].y = -directors[i].y;
            directors[i].z = -directors[i].z;
        }
    }

    // Gram matrix M = D^T D.
    double M[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    for (const Vec3d& d : directors) {
        M[0][0] += d.x*d.x; M[0][1] += d.x*d.y; M[0][2] += d.x*d.z;
        M[1][1] += d.y*d.y; M[1][2] += d.y*d.z;
        M[2][2] += d.z*d.z;
    }
    M[1][0] = M[0][1]; M[2][0] = M[0][2]; M[2][1] = M[1][2];

    double eval[3];
    double evec[3][3];
    JacobiEig3(M, eval, evec);
    // Smallest eigenvector is column 0.
    Vec3d omega{evec[0][0], evec[1][0], evec[2][0]};

    // Sign-fix via Σ d_k × d_{k+1}.
    Vec3d cross_sum{0, 0, 0};
    double abs_cross_sum = 0.0;
    for (size_t i = 0; i + 1 < directors.size(); ++i) {
        Vec3d c = Cross(directors[i], directors[i+1]);
        cross_sum = {cross_sum.x + c.x, cross_sum.y + c.y, cross_sum.z + c.z};
        abs_cross_sum += NormV(c);
    }
    if (DotV(omega, cross_sum) < 0.0) omega = {-omega.x, -omega.y, -omega.z};

    OmegaResult out;
    out.omega = omega;
    out.rotation_consistency = abs_cross_sum > 1e-12
        ? NormV(cross_sum) / abs_cross_sum
        : 0.0;
    out.valid = (eval[0] < eval[2]); // any nontrivial spectrum
    return out;
}


// ── β from tangent and Ω ────────────────────────────────────────────────────
// β = arccos(|T̂·Ω̂|), returned in RADIANS to match analysis-code conventions
// elsewhere (deg conversion is a viewer-side concern).
inline double BetaFromOmegaTangent(Vec3d tangent, Vec3d omega) {
    Vec3d tt = NormalizeV(tangent);
    Vec3d oo = NormalizeV(omega);
    double c = std::abs(DotV(tt, oo));
    if (!std::isfinite(c)) return std::nan("");
    if (c > 1.0) c = 1.0;
    if (c < 0.0) c = 0.0;
    return std::acos(c);
}

} // namespace DefectAnalysis

#endif // LBM_AN_ANALYSIS_DEFECT_BETA_H
