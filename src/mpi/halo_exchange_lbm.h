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
        // For the multi-axis corner sweep (PR VI step 3b) these will grow along
        // the sweep — widened in transverse axes that haven't been visited yet.
        // Keep the sizing here so 3b only touches this file.
        auto max_dim = [](int global, int n) { return (global + n - 1) / n; };
        max_yz = max_dim(Params::ny, mpi.dims[1]) * max_dim(Params::nz, mpi.dims[2]);
        max_xz = max_dim(Params::nx, mpi.dims[0]) * max_dim(Params::nz, mpi.dims[2]);
        max_xy = max_dim(Params::nx, mpi.dims[0]) * max_dim(Params::ny, mpi.dims[1]);
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

    void ExchangeLBM(FluidFields& ff) {
        if (world_size_ == 1) return;

        // Plan A: unsplit axes (dims_[d]==1) wrap locally during streaming,
        // so their ghosts hold no valid data and their owned boundaries are
        // already correct. Skip pack/send/recv/unpack on those axes entirely
        // — otherwise the unpack (with an empty recv_buf) would overwrite
        // valid locally-wrapped values with zeros.
        const bool split[3] = { dims_[0] > 1, dims_[1] > 1, dims_[2] > 1 };

        MPI_Request reqs[12];
        int n = 0;

        const int face_size[3] = {
            kCrossingDirs * static_cast<int>(max_yz),
            kCrossingDirs * static_cast<int>(max_xz),
            kCrossingDirs * static_cast<int>(max_xy)
        };

        double* f = ff.f.data();
        for (int k = 0; k < kCrossingDirs; ++k) {
            if (split[0]) {
                PackLBMLoYZ(f, Lattice::missingXHi[k], k);
                PackLBMHiYZ(f, Lattice::missingXLo[k], k);
            }
            if (split[1]) {
                PackLBMLoXZ(f, Lattice::missingYHi[k], k);
                PackLBMHiXZ(f, Lattice::missingYLo[k], k);
            }
            if (split[2]) {
                PackLBMLoXY(f, Lattice::missingZHi[k], k);
                PackLBMHiXY(f, Lattice::missingZLo[k], k);
            }
        }

        for (int d = 0; d < 3; ++d) {
            if (!split[d]) continue;
            MPI_Irecv(recv_buf_[2*d].data()  , face_size[d], MPI_DOUBLE, neighbor_lo_[d], d+3, cart_comm_, &reqs[n++]);
            MPI_Irecv(recv_buf_[2*d+1].data(), face_size[d], MPI_DOUBLE, neighbor_hi_[d], d  , cart_comm_, &reqs[n++]);
            MPI_Isend(send_buf_[2*d].data()  , face_size[d], MPI_DOUBLE, neighbor_lo_[d], d  , cart_comm_, &reqs[n++]);
            MPI_Isend(send_buf_[2*d+1].data(), face_size[d], MPI_DOUBLE, neighbor_hi_[d], d+3, cart_comm_, &reqs[n++]);
        }
        MPI_Waitall(n, reqs, MPI_STATUSES_IGNORE);

        // recv_buf_[0]/lo-x-recv holds the -x neighbour's hi-x GHOST — dirs
        // with ex>0 = missingXLo — which land in our owned lo-x boundary.
        // recv_buf_[1]/hi-x-recv symmetric with missingXHi at owned hi-x. y/z
        // follow the same pattern. SkipUnpack* elides overwrites at wall
        // corners where a local bounce has already set the same dir slot.
        for (int k = 0; k < kCrossingDirs; ++k) {
            if (split[0]) {
                UnpackLBMLoYZ(f, Lattice::missingXLo[k], k);
                UnpackLBMHiYZ(f, Lattice::missingXHi[k], k);
            }
            if (split[1]) {
                UnpackLBMLoXZ(f, Lattice::missingYLo[k], k);
                UnpackLBMHiXZ(f, Lattice::missingYHi[k], k);
            }
            if (split[2]) {
                UnpackLBMLoXY(f, Lattice::missingZLo[k], k);
                UnpackLBMHiXY(f, Lattice::missingZHi[k], k);
            }
        }
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
