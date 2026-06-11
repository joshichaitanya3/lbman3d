#ifndef LBM_AN_ANALYSIS_DISCLINATION_H
#define LBM_AN_ANALYSIS_DISCLINATION_H

#include <vector>
#include <cstdint>

inline constexpr uint8_t kVTK_LINE = 3;

/* !\brief A container struct that just holds 3D points
 * [x0,y0,z0, x1, y1, z1,...], but with the added guarantee that
 * consecutive points are connected to each other according to the
 * neighbor rule in PhysRevLett.132.258301
*/
struct Disclination {
    std::vector<double> points;
    std::vector<double> smooth_points;
    std::vector<double> smooth_tangents; // Unit tangent vectors at every point
    bool is_loop = false;

    const bool SmoothingAvailable() const {
        return (smooth_points.size() == points.size());
    }

    const bool TangentsAvailable() const {
        return (smooth_tangents.size() == smooth_points.size());
    }
};

class DisclinationMesh {
    std::vector<double> points;
    std::vector<int64_t> connectivity;
    int64_t num_points = 0;
    int64_t num_cells = 0;
    std::vector<int64_t> offsets = {0};
    std::vector<uint8_t> cell_types;

public:
    void AddDisclination(const Disclination& d) {
        const int64_t num_pts = static_cast<int64_t>(d.points.size() / 3);
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

        points.insert(points.end(), d.points.begin(), d.points.end());

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
};

#endif // LBM_AN_ANALYSIS_DISCLINATION_H
