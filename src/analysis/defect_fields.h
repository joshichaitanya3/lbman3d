#ifndef LBM_AN_ANALYSIS_DEFECT_FIELDS_H
#define LBM_AN_ANALYSIS_DEFECT_FIELDS_H

#include <stdint.h>
#include <array>
#include <vector>
#include <params.h>
#include "defect_connectivity_graph.h"
#include "disclination.h"

using namespace Params;

// Owns face-centered defect fields.
//
// def_x lives on x-normal plaquettes centred at (x, y+0.5, z+0.5); def_y on
// y-normal plaquettes at (x+0.5, y, z+0.5); def_z on z-normal plaquettes at
// (x+0.5, y+0.5, z). Each has its own extent along its two tangent axes: an
// axis is (N-1)-long when the domain is walled along it (the last plaquette
// sits between vertices N-2 and N-1) and N-long when it is periodic (the
// N-th plaquette wraps N-1 → 0 and must be present for seam-crossing
// disclinations to exist).
//
// `periodic_by_axis` is initialised from `periodicity_by_axis<BC>` at
// construction time (via `ActiveNematicSim`), so the per-axis extents are
// determined by the compile-time BC without templating DefectFields itself.
struct DefectFields {
    std::array<int, 3> periodic_by_axis = {0, 0, 0};

    // Per-axis plaquette counts for each face family. The "along-normal" axis
    // is always `n{x,y,z}` (a plaquette sits at every vertex layer); the two
    // tangent axes are `n - !periodic`, which is `n` if periodic and `n-1`
    // otherwise.
    int nfx_x = 0, nfx_y = 0, nfx_z = 0;
    int nfy_x = 0, nfy_y = 0, nfy_z = 0;
    int nfz_x = 0, nfz_y = 0, nfz_z = 0;

    std::vector<uint8_t> def_x;
    std::vector<uint8_t> def_y;
    std::vector<uint8_t> def_z;

    FaceId n_def_x = 0;
    FaceId n_def_y = 0;
    FaceId n_def_z = 0;

    // Flattened per-family indices. Callers must supply x,y,z already
    // reduced into the face family's extent (BuildConnectivityGraph does
    // this via forward/backward wrap helpers on the finder).
    FaceId FlatX(int x, int y, int z) const {
        return static_cast<FaceId>(z * nfx_y + y) * static_cast<FaceId>(nfx_x)
             + static_cast<FaceId>(x);
    }
    FaceId FlatY(int x, int y, int z) const {
        return static_cast<FaceId>(z * nfy_y + y) * static_cast<FaceId>(nfy_x)
             + static_cast<FaceId>(x);
    }
    FaceId FlatZ(int x, int y, int z) const {
        return static_cast<FaceId>(z * nfz_y + y) * static_cast<FaceId>(nfz_x)
             + static_cast<FaceId>(x);
    }

    // Global FaceIds combine the family offset with the flat family index, so
    // one FaceId identifies any face across the union of the three families.
    FaceId FidX(int x, int y, int z) const { return FlatX(x, y, z); }
    FaceId FidY(int x, int y, int z) const { return n_def_x + FlatY(x, y, z); }
    FaceId FidZ(int x, int y, int z) const { return n_def_x + n_def_y + FlatZ(x, y, z); }

    struct Index3D { int x, y, z; };
    struct Location { double x, y, z; };

    Index3D UnflattenX(FaceId id) const {
        const FaceId nx_f = static_cast<FaceId>(nfx_x);
        const FaceId ny_f = static_cast<FaceId>(nfx_y);
        return {
            static_cast<int>(id % nx_f),
            static_cast<int>((id / nx_f) % ny_f),
            static_cast<int>(id / (nx_f * ny_f))
        };
    }
    Index3D UnflattenY(FaceId id) const {
        const FaceId nx_f = static_cast<FaceId>(nfy_x);
        const FaceId ny_f = static_cast<FaceId>(nfy_y);
        return {
            static_cast<int>(id % nx_f),
            static_cast<int>((id / nx_f) % ny_f),
            static_cast<int>(id / (nx_f * ny_f))
        };
    }
    Index3D UnflattenZ(FaceId id) const {
        const FaceId nx_f = static_cast<FaceId>(nfz_x);
        const FaceId ny_f = static_cast<FaceId>(nfz_y);
        return {
            static_cast<int>(id % nx_f),
            static_cast<int>((id / nx_f) % ny_f),
            static_cast<int>(id / (nx_f * ny_f))
        };
    }

    Location FaceIdToLocation(FaceId id) const {
        if (id >= (n_def_x + n_def_y)) { // FidZ
            auto [x, y, z] = UnflattenZ(id - n_def_x - n_def_y);
            return {
                (static_cast<double>(x) + 0.5) * DX,
                (static_cast<double>(y) + 0.5) * DY,
                 static_cast<double>(z)        * DZ
            };
        } else if (id >= n_def_x) { // FidY
            auto [x, y, z] = UnflattenY(id - n_def_x);
            return {
                (static_cast<double>(x) + 0.5) * DX,
                 static_cast<double>(y)        * DY,
                (static_cast<double>(z) + 0.5) * DZ
            };
        } else { // FidX
            auto [x, y, z] = UnflattenX(id);
            return {
                 static_cast<double>(x)        * DX,
                (static_cast<double>(y) + 0.5) * DY,
                (static_cast<double>(z) + 0.5) * DZ
            };
        }
    }

    UndirectedGraph conn;
    std::vector<Disclination> disclinations;

    // Defaults to fully-walled — matches the pre-periodic behaviour on any
    // caller that hasn't opted in yet.
    explicit DefectFields(std::array<int, 3> periodic = {0, 0, 0});
};

bool FlipN(double, double, double, double, double, double);

#endif // LBM_AN_ANALYSIS_DEFECT_FIELDS_H
