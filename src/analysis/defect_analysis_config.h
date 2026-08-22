#ifndef LBM_AN_ANALYSIS_DEFECT_ANALYSIS_CONFIG_H
#define LBM_AN_ANALYSIS_DEFECT_ANALYSIS_CONFIG_H

#include <cmath>
#include <params.h>

// Analysis-time configuration for defect detection and β (tilt angle) computation.
// These constants are deliberately separated from Params (which are simulation-time
// solver constants) to support future shared library usage where post-processing tools
// may use defect analysis without needing the full solver configuration.

namespace DefectAnalysis {
    // ξ in lattice units: the defect core length scale derived from the Landau-de Gennes
    // free energy. The definition accounts for the sign convention in Params.
    inline double DefectCoreXi() {
        if (Params::A != 0.0) {
            // ξ² = L / (2A). With the sign convention in params.h A can be
            // negative in the ordered phase; take the magnitude so ξ is real.
            return std::sqrt(Params::L / (2.0 * (Params::A > 0 ? Params::A : -Params::A)));
        }
        // A = 0: minimum-of-quartic gives ξ = √(L·C/2) / |B|.
        const double absB = (Params::B > 0 ? Params::B : -Params::B);
        return std::sqrt(Params::L * Params::C / 2.0) / absB;
    }

    // Tuning parameters for polar-ring sampling around disclination vertices.
    // The ring radius R is computed as max(kOmegaRingXiFactor·ξ, kOmegaRingMinLU).
    // The factor takes the ring beyond the defect core; the floor keeps R ≥ 2 lattice
    // units (matching Nematics3D's default) so ring interpolation doesn't degenerate
    // for sub-lattice ξ. The arc-spacing drives the number of sample points.
    inline constexpr double kOmegaRingXiFactor = 4.0;
    inline constexpr double kOmegaRingMinLU    = 2.0;
    inline constexpr double kOmegaRingArcDist  = 0.5;

    // Polar-ring radius R in lattice units at which Ω (and hence β) is measured.
    inline double OmegaRingRadius() {
        const double r = kOmegaRingXiFactor * DefectCoreXi();
        return r > kOmegaRingMinLU ? r : kOmegaRingMinLU;
    }
}

#endif // LBM_AN_ANALYSIS_DEFECT_ANALYSIS_CONFIG_H
