#ifndef LBM_AN_ANALYSIS_DEFECT_ANALYZER_H
#define LBM_AN_ANALYSIS_DEFECT_ANALYZER_H

#include "defect_curve_routines.h"
#include "defect_fields.h"

class DefectAnalyzer {
    void CheckAndSetIfLoop(Disclination& d) {
        d.is_loop = IsLoop(d.points);
    }

    void SmoothenDisclination(Disclination& d) {
        CheckAndSetIfLoop(d);
        std::vector<double> smooth_pts = Smoothen(d.points, d.is_loop, 5);
        d.smooth_points = std::move(smooth_pts);
    }

    void SetTangents(Disclination& d) {
        SmoothenDisclination(d);
        SplineResult result = FitArcLengthSpline(d.smooth_points, d.is_loop);
        d.smooth_tangents = std::move(result.tangents);
    }

public:
    DefectAnalyzer() {};
    ~DefectAnalyzer() {};

    void AnalyzeDefects(DefectFields& df) {
        for (Disclination& d : df.disclinations) {
            SetTangents(d);
        }
    }
};

#endif // LBM_AN_ANALYSIS_DEFECT_ANALYZER_H
