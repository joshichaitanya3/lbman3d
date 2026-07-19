#ifndef LBM_AN_ANALYSIS_DEFECT_FIELDS_H
#define LBM_AN_ANALYSIS_DEFECT_FIELDS_H

#include <stdint.h>
#include <vector>
#include <params.h>
#include "defect_connectivity_graph.h"
#include "disclination.h"

using namespace Params;

// Owns defect fields. Face-centered, each with its own (non-uniform) extent,
// so each is indexed via its own FlatX/FlatY/FlatZ (below) rather than the
// shared idx(x,y,z): def_x is (NX, NY-1, NZ-1)-shaped, def_y is
// (NX-1, NY, NZ-1), def_z is (NX-1, NY-1, NZ).
struct DefectFields {
    std::vector<uint8_t> def_x;  // size NX     * (NY-1) * (NZ-1)
    std::vector<uint8_t> def_y;  // size (NX-1) *  NY    * (NZ-1)
    std::vector<uint8_t> def_z;  // size (NX-1) * (NY-1) *  NZ

    FaceId n_def_x = static_cast<FaceId>(nx)   * static_cast<FaceId>(ny-1) * static_cast<FaceId>(nz-1);
    FaceId n_def_y = static_cast<FaceId>(nx-1) * static_cast<FaceId>(ny)   * static_cast<FaceId>(nz-1);
    FaceId n_def_z = static_cast<FaceId>(nx-1) * static_cast<FaceId>(ny-1) * static_cast<FaceId>(nz)  ;
    
    FaceId FlatX(int x, int y, int z) { return static_cast<FaceId>(z * (ny-1) + y) * static_cast<FaceId>(nx  ) + static_cast<FaceId>(x);}
    FaceId FlatY(int x, int y, int z) { return static_cast<FaceId>(z * (ny)   + y) * static_cast<FaceId>(nx-1) + static_cast<FaceId>(x);}
    FaceId FlatZ(int x, int y, int z) { return static_cast<FaceId>(z * (ny-1) + y) * static_cast<FaceId>(nx-1) + static_cast<FaceId>(x);}
    FaceId FidX(int x, int y, int z) {return FlatX(x, y, z);}
    FaceId FidY(int x, int y, int z) {return n_def_x + FlatY(x, y, z);}
    FaceId FidZ(int x, int y, int z) {return n_def_x + n_def_y + FlatZ(x, y, z);}

    struct Index3D {
        int x, y, z;
    };
    
    struct Location {
        double x, y, z;
    };
    
    Index3D UnflattenX(FaceId idx) {
        return {
            static_cast<int>(idx % static_cast<FaceId>(nx)),
            static_cast<int>((idx / static_cast<FaceId>(nx)) % static_cast<FaceId>(ny-1)),
            static_cast<int>(idx / static_cast<FaceId>(nx * (ny-1)))
        };
    }

    Index3D UnflattenY(FaceId idx) {
        return {
            static_cast<int>(idx % static_cast<FaceId>(nx-1)),
            static_cast<int>((idx / static_cast<FaceId>(nx-1)) % static_cast<FaceId>(ny)),
            static_cast<int>(idx / static_cast<FaceId>((nx-1) * ny))
        };
    }

    Index3D UnflattenZ(FaceId idx) {
        return {
            static_cast<int>(idx % static_cast<FaceId>(nx-1)),
            static_cast<int>((idx / static_cast<FaceId>(nx-1)) % static_cast<FaceId>(ny-1)),
            static_cast<int>(idx / static_cast<FaceId>((nx-1) * (ny-1)))
        };
    }

    Location FaceIdToLocation(FaceId idx) {
        if (idx >= (n_def_x + n_def_y)) { // FidZ
            auto [x, y, z] = UnflattenZ(idx - n_def_x - n_def_y);
            return {
                (static_cast<double>(x)+0.5) * DX,
                (static_cast<double>(y)+0.5) * DX,
                static_cast<double>(z)       * DZ
            };
        }
        else if (idx >= n_def_x) { // FidY
            auto [x, y, z] = UnflattenY(idx - n_def_x);
            return {
                (static_cast<double>(x)+0.5) * DX,
                static_cast<double>(y)       * DY,
                (static_cast<double>(z)+0.5) * DZ
            };
        }
        
        else { // FidX
            auto [x, y, z] = UnflattenX(idx);
            return {
                static_cast<double>(x)       * DX,
                (static_cast<double>(y)+0.5) * DX,
                (static_cast<double>(z)+0.5) * DZ
            };
        }
    }

    UndirectedGraph conn;

    std::vector<Disclination> disclinations;

    DefectFields();
};


bool FlipN(
    double,
    double,
    double,
    double,
    double,
    double
);

#endif // LBM_AN_ANALYSIS_DEFECT_FIELDS_H