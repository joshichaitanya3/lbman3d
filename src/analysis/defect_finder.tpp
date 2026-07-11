#include "defect_fields.h"
#include "analysis_fields.h"
#include "qtensor_fields.h"
#include "params.h"
#include <ranges>
#include "vtkhdf_writer.h"
#include "boundary.h"
#include "defect_finder.h"

using namespace Params;

template<typename BC>
void DefectFinder<BC>::ComputeWindingNumbers(const QTensorFields& qf, const AnalysisFields& af, DefectFields& df) {

    // First, we go through all the faces at x = i (faces in the y-z plane)
    for (int z : std::views::iota(0, nz-1)) { // NZ-1 faces for each x, y
        for (int y : std::views::iota(0, ny-1)) {
            for (int x : std::views::iota(0, nx)) {

                /*
                 Our winding loop will be 
                 (y,z) -> (y+1,z) -> (y+1, z+1) -> (y,z+1)
                */
                double n1x = af.director_[dirIdx(x, y, z, 0)];
                double n1y = af.director_[dirIdx(x, y, z, 1)];
                double n1z = af.director_[dirIdx(x, y, z, 2)];

                double n2x = af.director_[dirIdx(x, y+1, z, 0)];
                double n2y = af.director_[dirIdx(x, y+1, z, 1)];
                double n2z = af.director_[dirIdx(x, y+1, z, 2)];

                if (FlipN(n1x, n1y, n1z, n2x, n2y, n2z)) {
                    n2x = -n2x;
                    n2y = -n2y;
                    n2z = -n2z;
                }
                
                double n3x = af.director_[dirIdx(x, y+1, z+1, 0)];
                double n3y = af.director_[dirIdx(x, y+1, z+1, 1)];
                double n3z = af.director_[dirIdx(x, y+1, z+1, 2)];

                if (FlipN(n2x, n2y, n2z, n3x, n3y, n3z)) {
                    n3x = -n3x;
                    n3y = -n3y;
                    n3z = -n3z;
                }

                double n4x = af.director_[dirIdx(x, y, z+1, 0)];
                double n4y = af.director_[dirIdx(x, y, z+1, 1)];
                double n4z = af.director_[dirIdx(x, y, z+1, 2)];

                if (FlipN(n3x, n3y, n3z, n4x, n4y, n4z)) {
                    n4x = -n4x;
                    n4y = -n4y;
                    n4z = -n4z;
                }

                if (FlipN(n4x, n4y, n4z, n1x, n1y, n1z)) {
                    df.def_x[df.FlatX(x, y, z)] = 1;
                }
            }
        }
    }

    // Now, we go through all the faces at y = j (faces in the x-z plane)
    for (int z : std::views::iota(0, nz-1)) { // NZ-1 faces for each x, y
        for (int y : std::views::iota(0, ny)) {
            for (int x : std::views::iota(0, nx-1)) {

                /*
                 Our winding loop will be 
                 (x,z) -> (x+1,z) -> (x+1, z+1) -> (x,z+1)
                */
                double n1x = af.director_[dirIdx(x, y, z, 0)];
                double n1y = af.director_[dirIdx(x, y, z, 1)];
                double n1z = af.director_[dirIdx(x, y, z, 2)];

                double n2x = af.director_[dirIdx(x+1, y, z, 0)];
                double n2y = af.director_[dirIdx(x+1, y, z, 1)];
                double n2z = af.director_[dirIdx(x+1, y, z, 2)];

                if (FlipN(n1x, n1y, n1z, n2x, n2y, n2z)) {
                    n2x = -n2x;
                    n2y = -n2y;
                    n2z = -n2z;
                }
                
                double n3x = af.director_[dirIdx(x+1, y, z+1, 0)];
                double n3y = af.director_[dirIdx(x+1, y, z+1, 1)];
                double n3z = af.director_[dirIdx(x+1, y, z+1, 2)];

                if (FlipN(n2x, n2y, n2z, n3x, n3y, n3z)) {
                    n3x = -n3x;
                    n3y = -n3y;
                    n3z = -n3z;
                }

                double n4x = af.director_[dirIdx(x, y, z+1, 0)];
                double n4y = af.director_[dirIdx(x, y, z+1, 1)];
                double n4z = af.director_[dirIdx(x, y, z+1, 2)];

                if (FlipN(n3x, n3y, n3z, n4x, n4y, n4z)) {
                    n4x = -n4x;
                    n4y = -n4y;
                    n4z = -n4z;
                }

                if (FlipN(n4x, n4y, n4z, n1x, n1y, n1z)) {
                    df.def_y[df.FlatY(x, y, z)] = 1;
                }
            }
        }
    }

    // Lastly, we go through all the faces at z = k (faces in the x-y plane)
    for (int z : std::views::iota(0, nz)) { // NZ-1 faces for each x, y
        for (int y : std::views::iota(0, ny-1)) {
            for (int x : std::views::iota(0, nx-1)) {

                /*
                 Our winding loop will be 
                 (x,y) -> (x+1,y) -> (x+1, y+1) -> (x,y+1)
                */
                double n1x = af.director_[dirIdx(x, y, z, 0)];
                double n1y = af.director_[dirIdx(x, y, z, 1)];
                double n1z = af.director_[dirIdx(x, y, z, 2)];

                double n2x = af.director_[dirIdx(x+1, y, z, 0)];
                double n2y = af.director_[dirIdx(x+1, y, z, 1)];
                double n2z = af.director_[dirIdx(x+1, y, z, 2)];

                if (FlipN(n1x, n1y, n1z, n2x, n2y, n2z)) {
                    n2x = -n2x;
                    n2y = -n2y;
                    n2z = -n2z;
                }
                
                double n3x = af.director_[dirIdx(x+1, y+1, z, 0)];
                double n3y = af.director_[dirIdx(x+1, y+1, z, 1)];
                double n3z = af.director_[dirIdx(x+1, y+1, z, 2)];

                if (FlipN(n2x, n2y, n2z, n3x, n3y, n3z)) {
                    n3x = -n3x;
                    n3y = -n3y;
                    n3z = -n3z;
                }

                double n4x = af.director_[dirIdx(x, y+1, z, 0)];
                double n4y = af.director_[dirIdx(x, y+1, z, 1)];
                double n4z = af.director_[dirIdx(x, y+1, z, 2)];

                if (FlipN(n3x, n3y, n3z, n4x, n4y, n4z)) {
                    n4x = -n4x;
                    n4y = -n4y;
                    n4z = -n4z;
                }

                if (FlipN(n4x, n4y, n4z, n1x, n1y, n1z)) {
                    df.def_z[df.FlatZ(x, y, z)] = 1;
                }
            }
        }
    }

}

template<typename BC>
void DefectFinder<BC>::BuildConnectivityGraph(DefectFields& df) {

    /* Processing def_x (centers of the y-z faces, point `3` in the figure below)
     * After processing all the neighbors of the current voxel (`2`, `4`, `5`, 6`, `1`),
     * we need to look at the neighbors from the voxel on the left.
     * 
     * Consider the point `1` for this analysis. If we are processing from [0,0,0] to [nx-1, ny-1, nz-1],
     * then the edges 6->1, 4->1 and 3->1 will have already been accounted for since their edge vector has
     * all components positive. So, only edges 5->1 and 2->1 need to be accounted for.
     * 
     * When processing def_y (point `4` below), similar logic will apply, however, in the case of 
     * the backward voxel (considering pt `2`), the def_x logic will have already accounted for the 
     * backward 2->1 edge. So We only need to account for the 5->2 edge.
     * 
     * Finally, for def_z (pt `6` and `5`), all the backward neighbors have been 
     * already accounted for, if done in the order def_x, def_y, def_z.
     * 
     * We will use the grid's 
     *   
     *             8_ _ _ _ _ _ _ _ _ _ _ _  7
     *            /|                        /|         z 
     *          /  |                      /  |         ^      y
     *        /    |      5             /    |         |    /^
     *      /      |      .           /      |         |  / 
     *    /        |      .         /        |         |/------> x
     * 9  _ _ _ _ _|_ _ _ . _ _2_10          |
     * |           |      .  .   |           |
     * |           |       .     |           |
     * |     3. . .| . .  0  .  .| . . 1     |
     * |           |    . .      |           |
     * |          12 _._ _._ _ _ | _ _ _ _ _ 11            
     * |          / 4     .      |          /
     * |        /         .      |        /
     * |      /           .      |      /
     * |    /             6      |    /
     * |  /                      |  /
     * 13 _ _ _ _ _ _ _ _ _ _ _ _14
     * 
     */
    auto compute_voxel = [&](int x, int y, int z, int xm, int xp, int ym, int yp, int zm, int zp) {
        if (df.def_x[df.FlatX(x, y, z)] == 1) {
            FaceId first = df.FidX(x, y, z);
            // Check all the forward neighbors. This makes the `first` point pt `3` in the figure above.

            // Checking pt `6`
            if (df.def_z[df.FlatZ(x, y, z)] == 1) {
                FaceId second = df.FidZ(x, y, z);
                df.conn.AddEdge(first, second); // 3 <-> 6
            }
            // Checking pt `5`
            if (df.def_z[df.FlatZ(x, y, zp)] == 1) {
                FaceId second = df.FidZ(x, y, zp);
                df.conn.AddEdge(first, second); // 3 <-> 5
            }

            // Checking pt `4`
            if (df.def_y[df.FlatY(x, y, z)] == 1) {
                FaceId second = df.FidY(x, y, z); 
                df.conn.AddEdge(first, second); // 3 <-> 4
            }
            // Checking pt `2`
            if (df.def_y[df.FlatY(x, yp, z)] == 1) {
                FaceId second = df.FidY(x, yp, z);
                df.conn.AddEdge(first, second); // 2 <-> 3
            }
            // Checking pt `1`
            if (df.def_x[df.FlatX(xp, y, z)] == 1) {
                FaceId second = df.FidX(xp, y, z);
                df.conn.AddEdge(first, second); // 1 <-> 3
            }

            // Now check the *two* remaining backward neighbors.
            // Here, we think of def_x(z, y, x) as pt `1`
            
            // Checking backward pt `5`
            // xm can be nx-1 under periodic BC, which is out of range for def_z (extent nx-1)
            if (xm < nx-1 && df.def_z[df.FlatZ(xm, y, zp)] == 1) {
                FaceId second = df.FidZ(xm, y, zp);
                df.conn.AddEdge(first, second); // 1 <-> 5
            }

            // Checking backward pt `2`
            if (xm < nx-1 && df.def_y[df.FlatY(xm, yp, z)] == 1) {
                FaceId second = df.FidY(xm, yp, z);
                df.conn.AddEdge(first, second); // 1 <-> 2
            }

        }

        // Now, we process the def_y

        if (df.def_y[df.FlatY(x, y, z)] == 1) {
            
            FaceId first = df.FidY(x, y, z);
            // Check all the forward neighbors. 
            // This makes the `first` point pt `4` in the figure above.
            // Note that we need not check the point `3` again

            // Checking pt `6`
            if (df.def_z[df.FlatZ(x, y, z)] == 1) {
                FaceId second = df.FidZ(x, y, z);
                df.conn.AddEdge(first, second); // 4 <-> 6
            }
            
            // Checking pt `5`
            if (df.def_z[df.FlatZ(x, y, zp)] == 1) {
                FaceId second = df.FidZ(x, y, zp);
                df.conn.AddEdge(first, second); // 4 <-> 5
            }

            // Checking pt `1`
            if (df.def_x[df.FlatX(xp, y, z)] == 1) {
                FaceId second = df.FidX(xp, y, z);
                df.conn.AddEdge(first, second); // 1 <-> 4
            }

            // Checking pt `2`
            if (df.def_y[df.FlatY(x, yp, z)] == 1) {
                FaceId second = df.FidY(x, yp, z);
                df.conn.AddEdge(first, second); // 2 <-> 4
            }

            // Now check the *one* remaining backward neighbor.
            // Here, we think of def_y(z, y, x) as pt `2`
            
            // Checking pt `5`
            // ym can be ny-1 under periodic BC, which is out of range for def_z (extent ny-1)
            if (ym < ny-1 && df.def_z[df.FlatZ(x, ym, zp)] == 1) {
                FaceId second = df.FidZ(x, ym, zp);
                df.conn.AddEdge(first, second); // 2 <-> 5
            }
        }

        // Finally, we process the def_z

        if (df.def_z[df.FlatZ(x, y, z)] == 1) {
            
            FaceId first = df.FidZ(x, y, z);
            // Check all the forward neighbors. 
            // This makes the `first` point pt `6` in the figure above.
            // Note that we need not check the points `3` and `4` again
            // Checking pt `1`
            if (df.def_x[df.FlatX(xp, y, z)] == 1) {
                FaceId second = df.FidX(xp, y, z);
                df.conn.AddEdge(first, second); // 1 <-> 6
            }

            // Checking pt `2`
            if (df.def_y[df.FlatY(x, yp, z)] == 1) {
                FaceId second = df.FidY(x, yp, z);
                df.conn.AddEdge(first, second); // 2 <-> 6
            }

            // Checking pt `5`
            if (df.def_z[df.FlatZ(x, y, zp)] == 1) {
                FaceId second = df.FidZ(x, y, zp);
                df.conn.AddEdge(first, second); // 5 <-> 6
            }

            // No backward neighbors are left to check.
        }
        // All done for this voxel.
    };

    // First, process the interior points
    for (int z : std::views::iota(1, nz-2)) {
        for (int y : std::views::iota(1, ny-2)) {
            for (int x : std::views::iota(1, nx-2)) {
                compute_voxel(x, y, z, x-1, x+1, y-1, y+1, z-1, z+1);
            }
        }
    }


    /* Now, the 6 faces.
     * Valid voxel indices are [0, nx-2] x [0, ny-2] x [0, nz-2], so the
     * boundary voxels are at {0, nz-2} in z, etc. The inner loop ranges are
     * also capped at n-2 (not n-1) for the same reason.
     */
     for (int z : {0, nz-2}) {
        for (int y : std::views::iota(0, ny-1)) {
            for (int x : std::views::iota(0, nx-1)) {
                compute_voxel(x, y, z, QXoff<BC>(x,-1), QXoff<BC>(x,1), QYoff<BC>(y,-1), QYoff<BC>(y,1), QZoff<BC>(z,-1), QZoff<BC>(z,1));
            }
        }
    }

    for (int z : std::views::iota(0, nz-1)) {
        for (int y : {0, ny-2}) {
            for (int x : std::views::iota(0, nx-1)) {
                compute_voxel(x, y, z, QXoff<BC>(x,-1), QXoff<BC>(x,1), QYoff<BC>(y,-1), QYoff<BC>(y,1), QZoff<BC>(z,-1), QZoff<BC>(z,1));

            }
        }
    }

    for (int z : std::views::iota(0, nz-1)) {
        for (int y : std::views::iota(0, ny-1)) {
            for (int x : {0, nx-2}) {
                compute_voxel(x, y, z, QXoff<BC>(x,-1), QXoff<BC>(x,1), QYoff<BC>(y,-1), QYoff<BC>(y,1), QZoff<BC>(z,-1), QZoff<BC>(z,1));
            }
        }
    }

}

template<typename BC>
void DefectFinder<BC>::IsolateDisclinationsFromGraph(DefectFields& df) {

    std::vector<std::vector<FaceId>> paths = df.conn.FindPaths();
    for (std::vector<FaceId> path : paths) {
        Disclination d;
        for (FaceId face : path) {
            auto [x, y, z] = df.FaceIdToLocation(face);
            d.points.push_back(x);
            d.points.push_back(y);
            d.points.push_back(z);
        }
        df.disclinations.push_back(d);
    }
}