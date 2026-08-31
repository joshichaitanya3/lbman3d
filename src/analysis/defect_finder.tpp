#include "defect_fields.h"
#include "defect_periodicity.h"
#include "analysis_fields.h"
#include "qtensor_fields.h"
#include <params.h>
#include <ranges>
#include "vtkhdf_writer.h"
#include "boundary.h"
#include "defect_finder.h"

using namespace Params;


namespace {

// Director component at (x, y, z), wrapping per-axis under periodic BC.
// halo_dirIdx has no modulo; the wrap lives here so future MPI code can
// swap this for a halo lookup and leave the finder untouched.
template<typename BC>
inline double Dir(const AnalysisFields& af, int x, int y, int z, int c) {
    const int xw = WrapDirIdx(x, nx, kPX<BC>);
    const int yw = WrapDirIdx(y, ny, kPY<BC>);
    const int zw = WrapDirIdx(z, nz, kPZ<BC>);
    return af.director_[af.grid.halo_dirIdx(xw, yw, zw, c)];
}

// Flag lookup at (x, y, z) in a face family. Returns 0 if the plaquette does
// not exist (past a walled boundary). Under periodic BC the axis wraps.
inline uint8_t LookupDefX(const DefectFields& df, int x, int y, int z,
                          bool px, bool py, bool pz) {
    const int xw = WrapFaceIdx(x, df.nfx_x, px);
    const int yw = WrapFaceIdx(y, df.nfx_y, py);
    const int zw = WrapFaceIdx(z, df.nfx_z, pz);
    if (xw < 0 || yw < 0 || zw < 0) return 0;
    return df.def_x[df.FlatX(xw, yw, zw)];
}
inline uint8_t LookupDefY(const DefectFields& df, int x, int y, int z,
                          bool px, bool py, bool pz) {
    const int xw = WrapFaceIdx(x, df.nfy_x, px);
    const int yw = WrapFaceIdx(y, df.nfy_y, py);
    const int zw = WrapFaceIdx(z, df.nfy_z, pz);
    if (xw < 0 || yw < 0 || zw < 0) return 0;
    return df.def_y[df.FlatY(xw, yw, zw)];
}
inline uint8_t LookupDefZ(const DefectFields& df, int x, int y, int z,
                          bool px, bool py, bool pz) {
    const int xw = WrapFaceIdx(x, df.nfz_x, px);
    const int yw = WrapFaceIdx(y, df.nfz_y, py);
    const int zw = WrapFaceIdx(z, df.nfz_z, pz);
    if (xw < 0 || yw < 0 || zw < 0) return 0;
    return df.def_z[df.FlatZ(xw, yw, zw)];
}

// FaceId at a possibly-wrapped face-family index. Same convention: returns a
// sentinel (kNoFace, ~0ull) if the plaquette does not exist.
inline FaceId FidX(const DefectFields& df, int x, int y, int z,
                   bool px, bool py, bool pz) {
    const int xw = WrapFaceIdx(x, df.nfx_x, px);
    const int yw = WrapFaceIdx(y, df.nfx_y, py);
    const int zw = WrapFaceIdx(z, df.nfx_z, pz);
    if (xw < 0 || yw < 0 || zw < 0) return static_cast<FaceId>(~0ull);
    return df.FidX(xw, yw, zw);
}
inline FaceId FidY(const DefectFields& df, int x, int y, int z,
                   bool px, bool py, bool pz) {
    const int xw = WrapFaceIdx(x, df.nfy_x, px);
    const int yw = WrapFaceIdx(y, df.nfy_y, py);
    const int zw = WrapFaceIdx(z, df.nfy_z, pz);
    if (xw < 0 || yw < 0 || zw < 0) return static_cast<FaceId>(~0ull);
    return df.FidY(xw, yw, zw);
}
inline FaceId FidZ(const DefectFields& df, int x, int y, int z,
                   bool px, bool py, bool pz) {
    const int xw = WrapFaceIdx(x, df.nfz_x, px);
    const int yw = WrapFaceIdx(y, df.nfz_y, py);
    const int zw = WrapFaceIdx(z, df.nfz_z, pz);
    if (xw < 0 || yw < 0 || zw < 0) return static_cast<FaceId>(~0ull);
    return df.FidZ(xw, yw, zw);
}

constexpr FaceId kNoFace = static_cast<FaceId>(~0ull);

} // namespace


template<typename BC>
void DefectFinder<BC>::ComputeWindingNumbers(const AnalysisFields& af, DefectFields& df) {

    constexpr bool px = kPX<BC>;
    constexpr bool py = kPY<BC>;
    constexpr bool pz = kPZ<BC>;

    // Runs the 4-director winding loop and writes the flag. Templated on
    // the plaquette normal so the four director lookups pick the right
    // in-plane neighbours; the "next" indices already carry any BC wrap.
    auto write_winding_flag = [&](double n1x, double n1y, double n1z,
                                  double n2x, double n2y, double n2z,
                                  double n3x, double n3y, double n3z,
                                  double n4x, double n4y, double n4z) -> uint8_t {
        if (FlipN(n1x, n1y, n1z, n2x, n2y, n2z)) { n2x = -n2x; n2y = -n2y; n2z = -n2z; }
        if (FlipN(n2x, n2y, n2z, n3x, n3y, n3z)) { n3x = -n3x; n3y = -n3y; n3z = -n3z; }
        if (FlipN(n3x, n3y, n3z, n4x, n4y, n4z)) { n4x = -n4x; n4y = -n4y; n4z = -n4z; }
        return FlipN(n4x, n4y, n4z, n1x, n1y, n1z) ? 1u : 0u;
    };

    // def_x: plaquettes in the y-z plane, one per x-vertex layer.
    #pragma omp parallel for default(shared) num_threads(kNumOMPThreads)
    for (int z = 0; z < df.nfx_z; ++z) {
        const int zp = z + 1; // "next" z-vertex; may equal nz under periodic-z (Dir wraps)
        for (int y = 0; y < df.nfx_y; ++y) {
            const int yp = y + 1;
            for (int x = 0; x < df.nfx_x; ++x) {
                const double n1x = Dir<BC>(af, x, y,  z,  0);
                const double n1y = Dir<BC>(af, x, y,  z,  1);
                const double n1z = Dir<BC>(af, x, y,  z,  2);
                const double n2x = Dir<BC>(af, x, yp, z,  0);
                const double n2y = Dir<BC>(af, x, yp, z,  1);
                const double n2z = Dir<BC>(af, x, yp, z,  2);
                const double n3x = Dir<BC>(af, x, yp, zp, 0);
                const double n3y = Dir<BC>(af, x, yp, zp, 1);
                const double n3z = Dir<BC>(af, x, yp, zp, 2);
                const double n4x = Dir<BC>(af, x, y,  zp, 0);
                const double n4y = Dir<BC>(af, x, y,  zp, 1);
                const double n4z = Dir<BC>(af, x, y,  zp, 2);
                df.def_x[df.FlatX(x, y, z)] = write_winding_flag(
                    n1x, n1y, n1z, n2x, n2y, n2z, n3x, n3y, n3z, n4x, n4y, n4z);
            }
        }
    }

    // def_y: plaquettes in the x-z plane, one per y-vertex layer.
    #pragma omp parallel for default(shared) num_threads(kNumOMPThreads)
    for (int z = 0; z < df.nfy_z; ++z) {
        const int zp = z + 1;
        for (int y = 0; y < df.nfy_y; ++y) {
            for (int x = 0; x < df.nfy_x; ++x) {
                const int xp = x + 1;
                const double n1x = Dir<BC>(af, x,  y, z,  0);
                const double n1y = Dir<BC>(af, x,  y, z,  1);
                const double n1z = Dir<BC>(af, x,  y, z,  2);
                const double n2x = Dir<BC>(af, xp, y, z,  0);
                const double n2y = Dir<BC>(af, xp, y, z,  1);
                const double n2z = Dir<BC>(af, xp, y, z,  2);
                const double n3x = Dir<BC>(af, xp, y, zp, 0);
                const double n3y = Dir<BC>(af, xp, y, zp, 1);
                const double n3z = Dir<BC>(af, xp, y, zp, 2);
                const double n4x = Dir<BC>(af, x,  y, zp, 0);
                const double n4y = Dir<BC>(af, x,  y, zp, 1);
                const double n4z = Dir<BC>(af, x,  y, zp, 2);
                df.def_y[df.FlatY(x, y, z)] = write_winding_flag(
                    n1x, n1y, n1z, n2x, n2y, n2z, n3x, n3y, n3z, n4x, n4y, n4z);
            }
        }
    }

    // def_z: plaquettes in the x-y plane, one per z-vertex layer.
    #pragma omp parallel for default(shared) num_threads(kNumOMPThreads)
    for (int z = 0; z < df.nfz_z; ++z) {
        for (int y = 0; y < df.nfz_y; ++y) {
            const int yp = y + 1;
            for (int x = 0; x < df.nfz_x; ++x) {
                const int xp = x + 1;
                const double n1x = Dir<BC>(af, x,  y,  z, 0);
                const double n1y = Dir<BC>(af, x,  y,  z, 1);
                const double n1z = Dir<BC>(af, x,  y,  z, 2);
                const double n2x = Dir<BC>(af, xp, y,  z, 0);
                const double n2y = Dir<BC>(af, xp, y,  z, 1);
                const double n2z = Dir<BC>(af, xp, y,  z, 2);
                const double n3x = Dir<BC>(af, xp, yp, z, 0);
                const double n3y = Dir<BC>(af, xp, yp, z, 1);
                const double n3z = Dir<BC>(af, xp, yp, z, 2);
                const double n4x = Dir<BC>(af, x,  yp, z, 0);
                const double n4y = Dir<BC>(af, x,  yp, z, 1);
                const double n4z = Dir<BC>(af, x,  yp, z, 2);
                df.def_z[df.FlatZ(x, y, z)] = write_winding_flag(
                    n1x, n1y, n1z, n2x, n2y, n2z, n3x, n3y, n3z, n4x, n4y, n4z);
            }
        }
    }
    (void)px; (void)py; (void)pz;
}


template<typename BC>
void DefectFinder<BC>::BuildConnectivityGraph(DefectFields& df) {

    /* Voxel-based edge insertion. Every voxel (x,y,z) has six face defects
     * around it (see the numbered figure below); we iterate voxels and add
     * an edge for every pair of face defects that both fire. To avoid
     * counting cross-voxel edges twice, backward-facing candidate edges
     * (into the xm/ym voxels) are only added where they cannot have been
     * inserted by an earlier voxel iteration. The bounds below iterate
     * every voxel that owns at least one plaquette; under periodic BC on
     * an axis the range extends to n (with wrapping done inside the face
     * lookups), otherwise it stops at n-1.
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
     */

    constexpr bool px = kPX<BC>;
    constexpr bool py = kPY<BC>;
    constexpr bool pz = kPZ<BC>;

    const int vx_end = px ? nx : nx - 1;
    const int vy_end = py ? ny : ny - 1;
    const int vz_end = pz ? nz : nz - 1;

    // Local helpers that look up the six face defects around a voxel, with
    // BC-aware wrap. FaceId comes back as kNoFace when the plaquette doesn't
    // exist (past a walled boundary), and lookups return 0 in that case.
    auto compute_voxel = [&](int x, int y, int z, int xm, int xp, int ym, int yp, int zp) {

        // ── pt 3 = def_x at (x, y, z): current voxel's low-x face ─────────
        if (LookupDefX(df, x, y, z, px, py, pz) == 1) {
            const FaceId first = FidX(df, x, y, z, px, py, pz);
            // forward: 3 <-> 6, 5, 4, 2, 1
            if (LookupDefZ(df, x,  y,  z,  px, py, pz) == 1) df.conn.AddEdge(first, FidZ(df, x, y, z,  px, py, pz));
            if (LookupDefZ(df, x,  y,  zp, px, py, pz) == 1) df.conn.AddEdge(first, FidZ(df, x, y, zp, px, py, pz));
            if (LookupDefY(df, x,  y,  z,  px, py, pz) == 1) df.conn.AddEdge(first, FidY(df, x, y, z,  px, py, pz));
            if (LookupDefY(df, x,  yp, z,  px, py, pz) == 1) df.conn.AddEdge(first, FidY(df, x, yp, z, px, py, pz));
            if (LookupDefX(df, xp, y,  z,  px, py, pz) == 1) df.conn.AddEdge(first, FidX(df, xp, y, z, px, py, pz));
            // backward: 1 <-> 5, 2 (through the xm voxel; the low-x face of xm
            // voxel — pt 3 there — was handled in that voxel's own iteration,
            // but 1 <-> 5 and 1 <-> 2 are cross-voxel across x=x and must be
            // added here).
            if (LookupDefZ(df, xm, y,  zp, px, py, pz) == 1) df.conn.AddEdge(first, FidZ(df, xm, y, zp, px, py, pz));
            if (LookupDefY(df, xm, yp, z,  px, py, pz) == 1) df.conn.AddEdge(first, FidY(df, xm, yp, z, px, py, pz));
        }

        // ── pt 4 = def_y at (x, y, z): current voxel's low-y face ─────────
        if (LookupDefY(df, x, y, z, px, py, pz) == 1) {
            const FaceId first = FidY(df, x, y, z, px, py, pz);
            // forward: 4 <-> 6, 5, 1, 2
            if (LookupDefZ(df, x,  y,  z,  px, py, pz) == 1) df.conn.AddEdge(first, FidZ(df, x, y, z,  px, py, pz));
            if (LookupDefZ(df, x,  y,  zp, px, py, pz) == 1) df.conn.AddEdge(first, FidZ(df, x, y, zp, px, py, pz));
            if (LookupDefX(df, xp, y,  z,  px, py, pz) == 1) df.conn.AddEdge(first, FidX(df, xp, y, z, px, py, pz));
            if (LookupDefY(df, x,  yp, z,  px, py, pz) == 1) df.conn.AddEdge(first, FidY(df, x, yp, z, px, py, pz));
            // backward: 2 <-> 5 (through the ym voxel)
            if (LookupDefZ(df, x, ym, zp, px, py, pz) == 1) df.conn.AddEdge(first, FidZ(df, x, ym, zp, px, py, pz));
        }

        // ── pt 6 = def_z at (x, y, z): current voxel's low-z face ─────────
        if (LookupDefZ(df, x, y, z, px, py, pz) == 1) {
            const FaceId first = FidZ(df, x, y, z, px, py, pz);
            // forward: 6 <-> 1, 2, 5
            if (LookupDefX(df, xp, y,  z,  px, py, pz) == 1) df.conn.AddEdge(first, FidX(df, xp, y, z, px, py, pz));
            if (LookupDefY(df, x,  yp, z,  px, py, pz) == 1) df.conn.AddEdge(first, FidY(df, x, yp, z, px, py, pz));
            if (LookupDefZ(df, x,  y,  zp, px, py, pz) == 1) df.conn.AddEdge(first, FidZ(df, x, y, zp, px, py, pz));
            // No backward edges remaining under the x→y→z voxel ordering.
        }
    };

    for (int z = 0; z < vz_end; ++z) {
        const int zp = z + 1; // periodic z-wrap happens inside the face lookups
        for (int y = 0; y < vy_end; ++y) {
            const int ym = y - 1;
            const int yp = y + 1;
            for (int x = 0; x < vx_end; ++x) {
                const int xm = x - 1;
                const int xp = x + 1;
                compute_voxel(x, y, z, xm, xp, ym, yp, zp);
            }
        }
    }
}


template<typename BC>
void DefectFinder<BC>::IsolateDisclinationsFromGraph(DefectFields& df) {

    std::vector<std::vector<FaceId>> paths = df.conn.FindPaths();
    for (const std::vector<FaceId>& path : paths) {
        Disclination d;
        d.points.reserve(3 * path.size());
        for (FaceId face : path) {
            auto [x, y, z] = df.FaceIdToLocation(face);
            d.points.push_back(x);
            d.points.push_back(y);
            d.points.push_back(z);
        }
        df.disclinations.push_back(std::move(d));
    }
}
