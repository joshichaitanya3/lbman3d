#ifndef LBM_AN_MPI_HALO_EXCHANGE_LBM_H_
#define LBM_AN_MPI_HALO_EXCHANGE_LBM_H_

#include "fluid_fields.h"
#include "local_grid.h"
#include "mpi_context.h"

#ifdef LBM_ENABLE_MPI
#include <mpi.h>
#include <vector>
#include <array>
#include "lattice_stencil.h"

// Halo exchange for LBM post-push-stream populations.
//
// Direction: rank r packs its GHOST layer (where its own push streaming
// deposited outgoing crossings) and sends it to the neighbour, which unpacks
// into its OWNED boundary — where next step's collision reads it. This
// orientation (ghost -> owned) is the exact opposite of the Q-tensor /
// passive-stress halo (owned -> ghost), so LBM lives in its own struct.
//
// Full contract (three invariants, all silent-failure modes if broken):
// see src/mpi/CLAUDE.md, section "Post-stream LBM exchange: three
// non-obvious invariants". In brief:
//   1. Pack from GHOST, unpack into OWNED.
//   2. Only the 5-direction crossing subset per face (Lattice::missing*).
//   3. Skip unpack at cells where a physical wall bounce already wrote the
//      same dir slot (needs is_wall_by_face<BC>).
// Plus: skip pack/send/recv/unpack entirely on axes with dims[d]==1
// (Plan A wraps unsplit axes locally at streaming time).
struct HaloExchangeLBM {
    MPI_Comm   cart_comm_;
    int        world_size_;
    int        neighbor_lo_[3], neighbor_hi_[3];
    int        dims_[3];   // ranks per axis; dims_[d]==1 means the axis is unsplit
    LocalGrid  grid_;
    int local_nx_, local_ny_, local_nz_;
    // XLo, XHi, YLo, YHi, ZLo, ZHi: true where the face is a physical wall.
    // Used by SkipUnpack* to preserve locally-bounced values at corners.
    std::array<bool, 6> is_wall_ = {};
    std::vector<double> send_buf_[6];
    std::vector<double> recv_buf_[6];
    size_t max_yz;
    size_t max_xz;
    size_t max_xy;
    // D3Q15: 5 crossing directions per face (Lattice::missing{X,Y,Z}{Lo,Hi}).
    static constexpr int kCrossingDirs = 5;

    explicit HaloExchangeLBM(LocalGrid g = LocalGrid::SingleRank()) :
        grid_(g)
    {}

    explicit HaloExchangeLBM(const LocalGrid& grid, const MPIContext& mpi,
                             std::array<bool, 6> is_wall = {})
        : cart_comm_(mpi.cart_comm),
          world_size_(mpi.world_size),
          grid_(grid),
          local_nx_(grid_.local_nx),
          local_ny_(grid_.local_ny),
          local_nz_(grid_.local_nz),
          is_wall_(is_wall)
    {
        for (int d = 0; d < 3; ++d) {
            MPI_Cart_shift(cart_comm_, d, 1, &neighbor_lo_[d], &neighbor_hi_[d]);
            dims_[d] = mpi.dims[d];
        }

        // Buffer size = largest possible face area across all ranks × kCrossingDirs.
        // Sweep is z -> y -> x, so the face-area widening rule (each ghost cell
        // packed by the first sweep-axis on which it lies outside owned) gives:
        //   z-face (XY plane, first hop):  widened in x AND y  → (max_nx + 2h)(max_ny + 2h)
        //   y-face (XZ plane, second hop): widened in x only   → (max_nx + 2h) max_nz
        //   x-face (YZ plane, third hop):  no widening         → max_ny max_nz
        // Across a seam the transverse dims always match (only the split axis
        // differs between neighbours), so per-hop slot indexing uses local_n
        // safely even under uneven splits.
        const int h = grid_.kHaloMPI;
        auto max_dim = [](int global, int n) { return (global + n - 1) / n; };
        const size_t max_nx_w = max_dim(Params::nx, mpi.dims[0]) + 2*h;
        const size_t max_ny_w = max_dim(Params::ny, mpi.dims[1]) + 2*h;
        max_yz = max_dim(Params::ny, mpi.dims[1]) * max_dim(Params::nz, mpi.dims[2]);
        max_xz = max_nx_w                          * max_dim(Params::nz, mpi.dims[2]);
        max_xy = max_nx_w                          * max_ny_w;
        // faces: 0=lo-x, 1=hi-x, 2=lo-y, 3=hi-y, 4=lo-z, 5=hi-z
        for (int f : {0,1}) { send_buf_[f].resize(kCrossingDirs * max_yz); recv_buf_[f].resize(kCrossingDirs * max_yz); }
        for (int f : {2,3}) { send_buf_[f].resize(kCrossingDirs * max_xz); recv_buf_[f].resize(kCrossingDirs * max_xz); }
        for (int f : {4,5}) { send_buf_[f].resize(kCrossingDirs * max_xy); recv_buf_[f].resize(kCrossingDirs * max_xy); }
    }

    // `dir` must be a crossing direction for the face being touched:
    //   Pack   lo-x GHOST   : ex[dir] < 0   (Lattice::missingXHi)
    //   Pack   hi-x GHOST   : ex[dir] > 0   (Lattice::missingXLo)
    //   Unpack owned lo-x   : ex[dir] > 0   (arrivals from -x neighbour)
    //   Unpack owned hi-x   : ex[dir] < 0   (arrivals from +x neighbour)
    //   (y and z analogous)
    // Non-crossing dirs at the boundary are set by local streaming and must
    // not be exchanged, or they clobber valid owned data. `k` is the 0..4
    // slot index within the 5-element crossing subset for that face.
    void PackLBMLoYZ(double* field, size_t dir, int k) {
        for (int z = 0; z < local_nz_; ++z)
            for (int y = 0; y < local_ny_; ++y)
                send_buf_[0][max_yz * k + z * local_ny_ + y] = field[grid_.halo_idx(-1, y, z, dir)];
    }
    void PackLBMHiYZ(double* field, size_t dir, int k) {
        for (int z = 0; z < local_nz_; ++z)
            for (int y = 0; y < local_ny_; ++y)
                send_buf_[1][max_yz * k + z * local_ny_ + y] = field[grid_.halo_idx(local_nx_, y, z, dir)];
    }
    void PackLBMLoXZ(double* field, size_t dir, int k) {
        for (int z = 0; z < local_nz_; ++z)
            for (int x = 0; x < local_nx_; ++x)
                send_buf_[2][max_xz * k + z * local_nx_ + x] = field[grid_.halo_idx(x, -1, z, dir)];
    }
    void PackLBMHiXZ(double* field, size_t dir, int k) {
        for (int z = 0; z < local_nz_; ++z)
            for (int x = 0; x < local_nx_; ++x)
                send_buf_[3][max_xz * k + z * local_nx_ + x] = field[grid_.halo_idx(x, local_ny_, z, dir)];
    }
    void PackLBMLoXY(double* field, size_t dir, int k) {
        for (int y = 0; y < local_ny_; ++y)
            for (int x = 0; x < local_nx_; ++x)
                send_buf_[4][max_xy * k + y * local_nx_ + x] = field[grid_.halo_idx(x, y, -1, dir)];
    }
    void PackLBMHiXY(double* field, size_t dir, int k) {
        for (int y = 0; y < local_ny_; ++y)
            for (int x = 0; x < local_nx_; ++x)
                send_buf_[5][max_xy * k + y * local_nx_ + x] = field[grid_.halo_idx(x, y, local_nz_, dir)];
    }

    // Widened variants for the multi-axis corner sweep (PR VI step 3b).
    // Sweep order: z -> y -> x. Per the invariant "each ghost cell is packed
    // by the first sweep-axis on which it lies outside owned", the widening
    // shrinks along the sweep:
    //   z-hop (XY face): widened in BOTH x and y transverse   (used first)
    //   y-hop (XZ face): widened in x only, z already swept   (used second)
    //   x-hop (YZ face): owned-only in both                   (uses the plain
    //                                                         PackLBMLoYZ/HiYZ
    //                                                         above, no need
    //                                                         for a widened
    //                                                         variant)
    // Widening in an unsplit transverse axis reads stale-zero ghost cells and
    // is harmless: the receiver writes those zeros into its own (unread)
    // ghost cells on that axis. A tighter version would gate widening on
    // dims_[transverse]>1; leaving that as a later optimization.

    // z-hop: XY face, widened in x AND y.
    void PackLBMLoXY_wideXY(double* field, size_t dir, int k) {
        const int h = grid_.kHaloMPI;
        const int row_stride = local_nx_ + 2*h;
        for (int y = -h; y < local_ny_ + h; ++y)
            for (int x = -h; x < local_nx_ + h; ++x)
                send_buf_[4][max_xy * k + (y + h) * row_stride + (x + h)]
                    = field[grid_.halo_idx(x, y, -1, dir)];
    }
    void PackLBMHiXY_wideXY(double* field, size_t dir, int k) {
        const int h = grid_.kHaloMPI;
        const int row_stride = local_nx_ + 2*h;
        for (int y = -h; y < local_ny_ + h; ++y)
            for (int x = -h; x < local_nx_ + h; ++x)
                send_buf_[5][max_xy * k + (y + h) * row_stride + (x + h)]
                    = field[grid_.halo_idx(x, y, local_nz_, dir)];
    }
    void UnpackLBMLoXY_wideXY(double* field, size_t dir, int k) {
        const int h = grid_.kHaloMPI;
        const int row_stride = local_nx_ + 2*h;
        for (int y = -h; y < local_ny_ + h; ++y)
            for (int x = -h; x < local_nx_ + h; ++x)
                if (!SkipUnpackXY(dir, x, y))
                    field[grid_.halo_idx(x, y, 0, dir)]
                        = recv_buf_[4][max_xy * k + (y + h) * row_stride + (x + h)];
    }
    void UnpackLBMHiXY_wideXY(double* field, size_t dir, int k) {
        const int h = grid_.kHaloMPI;
        const int row_stride = local_nx_ + 2*h;
        for (int y = -h; y < local_ny_ + h; ++y)
            for (int x = -h; x < local_nx_ + h; ++x)
                if (!SkipUnpackXY(dir, x, y))
                    field[grid_.halo_idx(x, y, local_nz_ - 1, dir)]
                        = recv_buf_[5][max_xy * k + (y + h) * row_stride + (x + h)];
    }

    // y-hop: XZ face, widened in x only (z is owned since it was swept first).
    void PackLBMLoXZ_wideX(double* field, size_t dir, int k) {
        const int h = grid_.kHaloMPI;
        const int row_stride = local_nx_ + 2*h;
        for (int z = 0; z < local_nz_; ++z)
            for (int x = -h; x < local_nx_ + h; ++x)
                send_buf_[2][max_xz * k + z * row_stride + (x + h)]
                    = field[grid_.halo_idx(x, -1, z, dir)];
    }
    void PackLBMHiXZ_wideX(double* field, size_t dir, int k) {
        const int h = grid_.kHaloMPI;
        const int row_stride = local_nx_ + 2*h;
        for (int z = 0; z < local_nz_; ++z)
            for (int x = -h; x < local_nx_ + h; ++x)
                send_buf_[3][max_xz * k + z * row_stride + (x + h)]
                    = field[grid_.halo_idx(x, local_ny_, z, dir)];
    }
    void UnpackLBMLoXZ_wideX(double* field, size_t dir, int k) {
        const int h = grid_.kHaloMPI;
        const int row_stride = local_nx_ + 2*h;
        for (int z = 0; z < local_nz_; ++z)
            for (int x = -h; x < local_nx_ + h; ++x)
                if (!SkipUnpackXZ(dir, x, z))
                    field[grid_.halo_idx(x, 0, z, dir)]
                        = recv_buf_[2][max_xz * k + z * row_stride + (x + h)];
    }
    void UnpackLBMHiXZ_wideX(double* field, size_t dir, int k) {
        const int h = grid_.kHaloMPI;
        const int row_stride = local_nx_ + 2*h;
        for (int z = 0; z < local_nz_; ++z)
            for (int x = -h; x < local_nx_ + h; ++x)
                if (!SkipUnpackXZ(dir, x, z))
                    field[grid_.halo_idx(x, local_ny_ - 1, z, dir)]
                        = recv_buf_[3][max_xz * k + z * row_stride + (x + h)];
    }

    // Skip-rules for unpacks at corner cells: a physical wall's local bounce
    // (NoSlip / SpecularReflection / MovingWall) writes into some
    // crossing-dir slots at the boundary cell. Overwriting with the
    // cross-rank exchange value (which is 0 at those slots, since the
    // neighbour's ghost was never written there by streaming) would clobber
    // the correct bounce value. Skip when the cell sits on such a wall AND
    // the dir is a possible "into" target of the wall-bounce reflection.
    //
    // Reflection maps flip the wall-normal velocity component: at a Y-lo
    // wall (y_global==0), outgoing dirs with ey<0 bounce to dirs with ey>0
    // at the same cell (same set for NoSlip's opp and SpecularReflection's
    // specY — the SET of target dirs is identical even when individual
    // mappings differ). So the skip predicate on a walled Y-lo cell is
    // ey[dir]>0. Analogous for Y-hi, Z-lo, Z-hi.
    bool SkipUnpackYZ(size_t dir, int y, int z) const {
        const int y_global = grid_.offset_y + y;
        const int z_global = grid_.offset_z + z;
        if (is_wall_[2] && y_global == 0            && Lattice::ey[dir] > 0) return true;
        if (is_wall_[3] && y_global == Params::ny-1 && Lattice::ey[dir] < 0) return true;
        if (is_wall_[4] && z_global == 0            && Lattice::ez[dir] > 0) return true;
        if (is_wall_[5] && z_global == Params::nz-1 && Lattice::ez[dir] < 0) return true;
        return false;
    }
    bool SkipUnpackXZ(size_t dir, int x, int z) const {
        const int x_global = grid_.offset_x + x;
        const int z_global = grid_.offset_z + z;
        if (is_wall_[0] && x_global == 0            && Lattice::ex[dir] > 0) return true;
        if (is_wall_[1] && x_global == Params::nx-1 && Lattice::ex[dir] < 0) return true;
        if (is_wall_[4] && z_global == 0            && Lattice::ez[dir] > 0) return true;
        if (is_wall_[5] && z_global == Params::nz-1 && Lattice::ez[dir] < 0) return true;
        return false;
    }
    bool SkipUnpackXY(size_t dir, int x, int y) const {
        const int x_global = grid_.offset_x + x;
        const int y_global = grid_.offset_y + y;
        if (is_wall_[0] && x_global == 0            && Lattice::ex[dir] > 0) return true;
        if (is_wall_[1] && x_global == Params::nx-1 && Lattice::ex[dir] < 0) return true;
        if (is_wall_[2] && y_global == 0            && Lattice::ey[dir] > 0) return true;
        if (is_wall_[3] && y_global == Params::ny-1 && Lattice::ey[dir] < 0) return true;
        return false;
    }

    void UnpackLBMLoYZ(double* field, size_t dir, int k) {
        for (int z = 0; z < local_nz_; ++z)
            for (int y = 0; y < local_ny_; ++y)
                if (!SkipUnpackYZ(dir, y, z))
                    field[grid_.halo_idx(0, y, z, dir)] = recv_buf_[0][max_yz * k + z * local_ny_ + y];
    }
    void UnpackLBMHiYZ(double* field, size_t dir, int k) {
        for (int z = 0; z < local_nz_; ++z)
            for (int y = 0; y < local_ny_; ++y)
                if (!SkipUnpackYZ(dir, y, z))
                    field[grid_.halo_idx(local_nx_ - 1, y, z, dir)] = recv_buf_[1][max_yz * k + z * local_ny_ + y];
    }
    void UnpackLBMLoXZ(double* field, size_t dir, int k) {
        for (int z = 0; z < local_nz_; ++z)
            for (int x = 0; x < local_nx_; ++x)
                if (!SkipUnpackXZ(dir, x, z))
                    field[grid_.halo_idx(x, 0, z, dir)] = recv_buf_[2][max_xz * k + z * local_nx_ + x];
    }
    void UnpackLBMHiXZ(double* field, size_t dir, int k) {
        for (int z = 0; z < local_nz_; ++z)
            for (int x = 0; x < local_nx_; ++x)
                if (!SkipUnpackXZ(dir, x, z))
                    field[grid_.halo_idx(x, local_ny_ - 1, z, dir)] = recv_buf_[3][max_xz * k + z * local_nx_ + x];
    }
    void UnpackLBMLoXY(double* field, size_t dir, int k) {
        for (int y = 0; y < local_ny_; ++y)
            for (int x = 0; x < local_nx_; ++x)
                if (!SkipUnpackXY(dir, x, y))
                    field[grid_.halo_idx(x, y, 0, dir)] = recv_buf_[4][max_xy * k + y * local_nx_ + x];
    }
    void UnpackLBMHiXY(double* field, size_t dir, int k) {
        for (int y = 0; y < local_ny_; ++y)
            for (int x = 0; x < local_nx_; ++x)
                if (!SkipUnpackXY(dir, x, y))
                    field[grid_.halo_idx(x, y, local_nz_ - 1, dir)] = recv_buf_[5][max_xy * k + y * local_nx_ + x];
    }

    // Sequential axis sweep z -> y -> x. Each hop must complete before the
    // next, because the y-hop pack reads cells the z-hop unpack just wrote,
    // and likewise the x-hop reads what y-hop wrote. Body-diagonal pops
    // (dirs 7..14) enter the sweep at whichever axis their crossing places
    // them outside owned first, and are consumed at the axis whose owned
    // face they land in. Plan A: axes with dims_[d]==1 wrap locally at
    // streaming — skip pack/exchange/unpack for those.
    void ExchangeLBM(FluidFields& ff) {
        if (world_size_ == 1) return;

        const bool split[3] = { dims_[0] > 1, dims_[1] > 1, dims_[2] > 1 };
        double* f = ff.f.data();

        // A PROC_NULL neighbour means the corresponding face is a physical
        // wall edge (non-periodic axis with this rank at the domain boundary).
        // The local wall bounce has already written the correct value into
        // the same slot at the owned boundary cell, so unpacking would just
        // stomp it with zeros (the recv from PROC_NULL leaves recv_buf_
        // untouched). Guard each side's unpack with this gate. This is the
        // "on-axis" companion to SkipUnpack{YZ,XZ,XY}'s transverse-wall skip.
        const bool recv_lo[3] = {
            neighbor_lo_[0] != MPI_PROC_NULL,
            neighbor_lo_[1] != MPI_PROC_NULL,
            neighbor_lo_[2] != MPI_PROC_NULL,
        };
        const bool recv_hi[3] = {
            neighbor_hi_[0] != MPI_PROC_NULL,
            neighbor_hi_[1] != MPI_PROC_NULL,
            neighbor_hi_[2] != MPI_PROC_NULL,
        };

        // ---- z-hop (first): XY face, widened in both x and y ----
        if (split[2]) {
            for (int k = 0; k < kCrossingDirs; ++k) {
                PackLBMLoXY_wideXY(f, Lattice::missingZHi[k], k);
                PackLBMHiXY_wideXY(f, Lattice::missingZLo[k], k);
            }
            SendrecvAxis(2);
            for (int k = 0; k < kCrossingDirs; ++k) {
                if (recv_lo[2]) UnpackLBMLoXY_wideXY(f, Lattice::missingZLo[k], k);
                if (recv_hi[2]) UnpackLBMHiXY_wideXY(f, Lattice::missingZHi[k], k);
            }
        }

        // ---- y-hop (second): XZ face, widened in x only ----
        if (split[1]) {
            for (int k = 0; k < kCrossingDirs; ++k) {
                PackLBMLoXZ_wideX(f, Lattice::missingYHi[k], k);
                PackLBMHiXZ_wideX(f, Lattice::missingYLo[k], k);
            }
            SendrecvAxis(1);
            for (int k = 0; k < kCrossingDirs; ++k) {
                if (recv_lo[1]) UnpackLBMLoXZ_wideX(f, Lattice::missingYLo[k], k);
                if (recv_hi[1]) UnpackLBMHiXZ_wideX(f, Lattice::missingYHi[k], k);
            }
        }

        // ---- x-hop (third): YZ face, owned only (identical to the plain
        // 3a face exchange — reuses the unwidened primitives) ----
        if (split[0]) {
            for (int k = 0; k < kCrossingDirs; ++k) {
                PackLBMLoYZ(f, Lattice::missingXHi[k], k);
                PackLBMHiYZ(f, Lattice::missingXLo[k], k);
            }
            SendrecvAxis(0);
            for (int k = 0; k < kCrossingDirs; ++k) {
                if (recv_lo[0]) UnpackLBMLoYZ(f, Lattice::missingXLo[k], k);
                if (recv_hi[0]) UnpackLBMHiYZ(f, Lattice::missingXHi[k], k);
            }
        }
    }

    // Post + wait for one axis's face exchange. `d` selects the axis (0=x,
    // 1=y, 2=z). recv_buf_/send_buf_[2*d]   -> lo-face against neighbor_lo_.
    //                 recv_buf_/send_buf_[2*d+1] -> hi-face against neighbor_hi_.
    void SendrecvAxis(int d) {
        const int face_size_area[3] = {
            static_cast<int>(max_yz),
            static_cast<int>(max_xz),
            static_cast<int>(max_xy)
        };
        const int face_size = kCrossingDirs * face_size_area[d];
        MPI_Request reqs[4];
        MPI_Irecv(recv_buf_[2*d].data()  , face_size, MPI_DOUBLE, neighbor_lo_[d], d+3, cart_comm_, &reqs[0]);
        MPI_Irecv(recv_buf_[2*d+1].data(), face_size, MPI_DOUBLE, neighbor_hi_[d], d  , cart_comm_, &reqs[1]);
        MPI_Isend(send_buf_[2*d].data()  , face_size, MPI_DOUBLE, neighbor_lo_[d], d  , cart_comm_, &reqs[2]);
        MPI_Isend(send_buf_[2*d+1].data(), face_size, MPI_DOUBLE, neighbor_hi_[d], d+3, cart_comm_, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
    }
};

#else

struct HaloExchangeLBM {
    explicit HaloExchangeLBM(LocalGrid = LocalGrid::SingleRank()) {}
    explicit HaloExchangeLBM(const LocalGrid&, const MPIContext&, std::array<bool, 6> = {}) {}

    void ExchangeLBM(FluidFields&) {}
};

#endif

#endif // LBM_AN_MPI_HALO_EXCHANGE_LBM_H_
