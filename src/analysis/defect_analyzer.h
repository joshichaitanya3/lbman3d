#ifndef LBM_AN_ANALYSIS_DEFECT_ANALYZER_H
#define LBM_AN_ANALYSIS_DEFECT_ANALYZER_H

#include <array>
#include <params.h>
#include "boundary.h"
#include "defect_curve_routines.h"
#include "defect_fields.h"
#include "defect_beta.h"
#include "defect_analysis_config.h"
#include "qtensor_fields.h"

// ─────────────────────────────────────────────────────────────────────────────
// DefectAnalyzer: turn raw wrapped defect polylines into smoothed unwrapped
// curves with per-vertex tangents and β. Templated on BC so the trilinear
// Q interpolation (used by β) knows how to wrap ring-samples that fall in
// a periodic-image cell or clamp against a wall.
// ─────────────────────────────────────────────────────────────────────────────

template<typename BC>
class DefectAnalyzer {

    // Physical periods along each axis (0 if walled).
    std::array<double, 3> Periods() const {
        return {
            std::is_same_v<typename BC::XLo::QBC, Periodic> ? Params::nx * Params::DX : 0.0,
            std::is_same_v<typename BC::YLo::QBC, Periodic> ? Params::ny * Params::DY : 0.0,
            std::is_same_v<typename BC::ZLo::QBC, Periodic> ? Params::nz * Params::DZ : 0.0,
        };
    }

    void CheckAndSetIfLoop(Disclination& d) {
        d.is_loop = IsLoop(d.points);
    }

    // Populate smooth_points (in UNWRAPPED coordinates), preserving the
    // closing-duplicate convention with the appropriate period offset for
    // cross-boundary loops.
    void UnwrapAndSmoothen(Disclination& d) {
        CheckAndSetIfLoop(d);
        const auto periods = Periods();

        if (d.is_loop) {
            // Unwrap the full closed path (with closing duplicate) so the
            // last unwrapped point sits at c_0 + winding_offset; drop the
            // duplicate for smoothing.
            std::vector<double> unw_all = MinimumImageUnwrap(d.points, periods);
            const int n_total = static_cast<int>(d.points.size() / 3);
            for (int a = 0; a < 3; ++a) {
                d.period_offset[a] = unw_all[3*(n_total-1) + a] - unw_all[a];
            }
            std::vector<double> unique_unw(unw_all.begin(),
                                            unw_all.begin() + 3*(n_total-1));
            std::vector<double> smooth = Smoothen(unique_unw, /*is_loop=*/true, 5);
            d.smooth_points.assign(smooth.begin(), smooth.end());
            d.smooth_points.push_back(smooth[0] + d.period_offset[0]);
            d.smooth_points.push_back(smooth[1] + d.period_offset[1]);
            d.smooth_points.push_back(smooth[2] + d.period_offset[2]);
        } else {
            std::vector<double> unw = MinimumImageUnwrap(d.points, periods);
            d.smooth_points = Smoothen(unw, /*is_loop=*/false, 5);
        }
    }

    void SetTangents(Disclination& d) {
        UnwrapAndSmoothen(d);
        SplineResult result = FitArcLengthSpline(d.smooth_points, d.is_loop);
        d.smooth_tangents = std::move(result.tangents);
    }

    // Populate d.beta with one β (radians) per smooth_points vertex.
    void ComputeBeta(Disclination& d, const QTensorFields& qf) {
        const int n = static_cast<int>(d.smooth_points.size() / 3);
        d.beta.assign(static_cast<size_t>(n), std::nan(""));

        if (!d.TangentsAvailable() || n < 2) return;

        const double R = DefectAnalysis::OmegaRingRadius();
        const auto periods = Periods();

        for (int i = 0; i < n; ++i) {
            DefectAnalysis::Vec3d center{
                d.smooth_points[3*i    ],
                d.smooth_points[3*i + 1],
                d.smooth_points[3*i + 2]
            };
            // Wrap the center back into [0, period) on each periodic axis so
            // trilinear neighbours land at valid vertices. Ring offsets
            // (typically a few lattice units) then only cross the seam near
            // it, which the BC-aware neighbour lookup handles.
            for (int a = 0; a < 3; ++a) {
                if (periods[a] > 0.0) {
                    double& c = (a == 0) ? center.x : (a == 1) ? center.y : center.z;
                    c = std::fmod(c, periods[a]);
                    if (c < 0.0) c += periods[a];
                }
            }
            DefectAnalysis::Vec3d tangent{
                d.smooth_tangents[3*i    ],
                d.smooth_tangents[3*i + 1],
                d.smooth_tangents[3*i + 2]
            };
            auto res = DefectAnalysis::OmegaOnRing<BC>(qf, center, tangent, R);
            d.beta[i] = res.valid
                ? DefectAnalysis::BetaFromOmegaTangent(tangent, res.omega)
                : std::nan("");
        }
    }

public:
    DefectAnalyzer() = default;

    void AnalyzeDefects(DefectFields& df, const QTensorFields& qf) {
        for (Disclination& d : df.disclinations) {
            SetTangents(d);
            ComputeBeta(d, qf);
        }
    }
};

#endif // LBM_AN_ANALYSIS_DEFECT_ANALYZER_H
