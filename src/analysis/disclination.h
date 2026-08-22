#ifndef LBM_AN_ANALYSIS_DISCLINATION_H
#define LBM_AN_ANALYSIS_DISCLINATION_H

#include <array>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

inline constexpr uint8_t kVTK_LINE = 3;

/* !\brief A single disclination line: an ordered sequence of raw defect
 * detections (in wrapped physical coordinates), plus an optional smoothed
 * unwrapped representation with tangents and β at each vertex.
 *
 * Convention for closed loops (including cross-boundary ones): points[0] ==
 * points[last] on entry (the graph traversal ends at the starting face).
 * After smoothing, smooth_points[0] and smooth_points[last] may differ by
 * `period_offset`, an integer-period lattice shift — nonzero exactly when the
 * loop crosses one or more periodic seams. Downstream code (the DisclinationMesh
 * exporter, the β sampler) treats a nonzero period_offset as the signal that
 * the loop is cross-boundary and needs unwrapped-continuous tangents rather
 * than wrapped-image ones.
 */
struct Disclination {
    std::vector<double> points;
    std::vector<double> smooth_points;
    std::vector<double> smooth_tangents; // unit tangent at every smooth point
    std::vector<double> beta;            // one β per smooth point, radians
    std::array<double, 3> period_offset = {0.0, 0.0, 0.0};
    bool is_loop = false;

    bool SmoothingAvailable() const {
        return (smooth_points.size() == points.size());
    }

    bool TangentsAvailable() const {
        return (smooth_tangents.size() == smooth_points.size());
    }

    bool BetaAvailable() const {
        return (3 * beta.size() == smooth_points.size());
    }
};

class DisclinationMesh {
    std::vector<double> points;
    std::vector<int64_t> connectivity;
    int64_t num_points = 0;
    int64_t num_cells = 0;
    std::vector<int64_t> offsets = {0};
    std::vector<uint8_t> cell_types;
    bool tangents_available = false;
    bool beta_available = false;
    std::vector<double> tangents;
    std::vector<double> beta;

public:
    void AddDisclination(const Disclination& d) {

        const std::vector<double>& disc_pts = d.SmoothingAvailable() ? d.smooth_points : d.points;

        const int64_t num_pts = static_cast<int64_t>(disc_pts.size() / 3);
        if (num_pts < 2) return;

        const int64_t num_new_cells = num_pts - 1;
        const int64_t offset_base = offsets.back();
        const int64_t points_base = num_points;

        num_points += num_pts;
        num_cells += num_new_cells;

        points.reserve(3 * static_cast<size_t>(num_points));
        connectivity.reserve(2 * static_cast<size_t>(num_cells));
        offsets.reserve(1 + static_cast<size_t>(num_cells));
        cell_types.reserve(static_cast<size_t>(num_cells));

        points.insert(points.end(), disc_pts.begin(), disc_pts.end());

        tangents.reserve(3 * static_cast<size_t>(num_points));
        beta.reserve(static_cast<size_t>(num_points));

        if (d.TangentsAvailable()) {
            tangents_available = true;
            tangents.insert(tangents.end(), d.smooth_tangents.begin(), d.smooth_tangents.end());
        } else if (tangents_available) {
            throw std::runtime_error("Buggy disclination analysis: some contain tangents, some don't.");
        }

        if (d.BetaAvailable()) {
            beta_available = true;
            beta.insert(beta.end(), d.beta.begin(), d.beta.end());
        } else if (beta_available) {
            throw std::runtime_error("Buggy disclination analysis: some contain beta, some don't.");
        }

        for (int64_t i = 0; i < num_new_cells; ++i) {
            connectivity.push_back(points_base + i);
            connectivity.push_back(points_base + i + 1);
            offsets.push_back(offset_base + 2 * (i + 1));
        }

        cell_types.insert(cell_types.end(), static_cast<size_t>(num_new_cells), kVTK_LINE);
    }

    const std::vector<double>& Points() const { return points; }
    const std::vector<int64_t>& Connectivity() const { return connectivity; }
    const std::vector<int64_t>& Offsets() const { return offsets; }
    const std::vector<uint8_t>& CellTypes() const { return cell_types; }
    int64_t NumPoints() const { return num_points; }
    int64_t NumCells() const { return num_cells; }
    int64_t NumConnectivityIds() const {return connectivity.size();}
    const std::vector<double>& Tangents() const { return tangents; }
    const std::vector<double>& Beta() const { return beta; }

    bool TangentsAvailable() const {return tangents_available;}
    bool BetaAvailable() const {return beta_available;}
};

#endif // LBM_AN_ANALYSIS_DISCLINATION_H
