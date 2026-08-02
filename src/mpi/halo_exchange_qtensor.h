#ifndef LBM_AN_MPI_HALO_EXCHANGE_QTENSOR_H_
#define LBM_AN_MPI_HALO_EXCHANGE_QTENSOR_H_

#include "fluid_fields.h"
#include "qtensor_fields.h"
#include "local_grid.h"
#include "mpi_context.h"

#ifdef LBM_ENABLE_MPI
#include <mpi.h>
#include <vector>

// Halo exchange for the Q-tensor + passive-stress FD stencils.
//
// Direction: rank r packs its OWNED boundary planes, sends them to the
// neighbour, and the neighbour unpacks into its GHOST layer. This is the
// star-stencil convention: each rank fills its own ghosts with the
// neighbour's owned values that its FD stencils need to read. Every field is
// scalar; the LBM per-population exchange runs in the opposite direction
// (ghost -> owned) and lives in halo_exchange_lbm.h.
//
// Buffer sizing: ExchangePassiveStresses packs the most fields — 13, being
// 5 Q +  5 symmetric stress Sigma + 3 antisymmetric stress tau; kMaxFields=13 covers
// it. ExchangeQTensor packs 8 (5 Q + 3 velocity). Bump kMaxFields whenever a
// field is added to either exchange: the buffers are sized kMaxFields × face
// area, so an extra PackField would write past the end.
struct HaloExchangeQTensor {
    MPI_Comm   cart_comm_;
    int        world_size_;
    int        neighbor_lo_[3], neighbor_hi_[3];
    LocalGrid  grid_;
    int local_nx_, local_ny_, local_nz_;
    std::vector<double> send_buf_[6];
    std::vector<double> recv_buf_[6];
    size_t max_yz;
    size_t max_xz;
    size_t max_xy;
    static constexpr size_t kMaxFields = 13;

    explicit HaloExchangeQTensor(LocalGrid g = LocalGrid::SingleRank()) :
        grid_(g)
    {}

    explicit HaloExchangeQTensor(const LocalGrid& grid, const MPIContext& mpi)
        : cart_comm_(mpi.cart_comm),
          world_size_(mpi.world_size),
          grid_(grid),
          local_nx_(grid_.local_nx),
          local_ny_(grid_.local_ny),
          local_nz_(grid_.local_nz)
    {
        for (int d = 0; d < 3; ++d)
            MPI_Cart_shift(cart_comm_, d, 1, &neighbor_lo_[d], &neighbor_hi_[d]);

        // Buffer size = largest possible face area across all ranks × kMaxFields
        auto max_dim = [](int global, int n) { return (global + n - 1) / n; };
        max_yz = max_dim(Params::ny, mpi.dims[1]) * max_dim(Params::nz, mpi.dims[2]);
        max_xz = max_dim(Params::nx, mpi.dims[0]) * max_dim(Params::nz, mpi.dims[2]);
        max_xy = max_dim(Params::nx, mpi.dims[0]) * max_dim(Params::ny, mpi.dims[1]);
        // faces: 0=lo-x, 1=hi-x, 2=lo-y, 3=hi-y, 4=lo-z, 5=hi-z
        for (int f : {0,1}) { send_buf_[f].resize(kMaxFields * max_yz); recv_buf_[f].resize(kMaxFields * max_yz); }
        for (int f : {2,3}) { send_buf_[f].resize(kMaxFields * max_xz); recv_buf_[f].resize(kMaxFields * max_xz); }
        for (int f : {4,5}) { send_buf_[f].resize(kMaxFields * max_xy); recv_buf_[f].resize(kMaxFields * max_xy); }
    }

    // Owned coordinates are 0-based (0 .. local_n-1); halo_idx adds the ghost
    // offset internally. Owned boundary planes are logical 0 / local_n-1.
    void PackFieldXY(double* field, size_t fieldIdx) {
        for (int y = 0; y < local_ny_; ++y) {
            for (int x = 0; x < local_nx_; ++x) {
                send_buf_[4][(max_xy * fieldIdx) + y * local_nx_ + x] = field[grid_.halo_idx(x, y, 0)];
                send_buf_[5][(max_xy * fieldIdx) + y * local_nx_ + x] = field[grid_.halo_idx(x, y, local_nz_ - 1)];
            }
        }
    }
    void PackFieldXZ(double* field, size_t fieldIdx) {
        for (int z = 0; z < local_nz_; ++z) {
            for (int x = 0; x < local_nx_; ++x) {
                send_buf_[2][(max_xz * fieldIdx) + z * local_nx_ + x] = field[grid_.halo_idx(x, 0, z)];
                send_buf_[3][(max_xz * fieldIdx) + z * local_nx_ + x] = field[grid_.halo_idx(x, local_ny_ - 1, z)];
            }
        }
    }
    void PackFieldYZ(double* field, size_t fieldIdx) {
        for (int z = 0; z < local_nz_; ++z) {
            for (int y = 0; y < local_ny_; ++y) {
                send_buf_[0][(max_yz * fieldIdx) + z * local_ny_ + y] = field[grid_.halo_idx(0, y, z)];
                send_buf_[1][(max_yz * fieldIdx) + z * local_ny_ + y] = field[grid_.halo_idx(local_nx_ - 1, y, z)];
            }
        }
    }

    void PackField(double* field, size_t fieldIdx, int d) {
        if      (d == 0) PackFieldYZ(field, fieldIdx);
        else if (d == 1) PackFieldXZ(field, fieldIdx);
        else             PackFieldXY(field, fieldIdx);
    }

    // Ghost planes are the layer just outside owned: logical -1 / local_n.
    void UnpackFieldXY(double* field, size_t fieldIdx) {
        for (int y = 0; y < local_ny_; ++y) {
            for (int x = 0; x < local_nx_; ++x) {
                field[grid_.halo_idx(x, y, -1)]        = recv_buf_[4][(max_xy * fieldIdx) + y * local_nx_ + x];
                field[grid_.halo_idx(x, y, local_nz_)] = recv_buf_[5][(max_xy * fieldIdx) + y * local_nx_ + x];
            }
        }
    }
    void UnpackFieldXZ(double* field, size_t fieldIdx) {
        for (int z = 0; z < local_nz_; ++z) {
            for (int x = 0; x < local_nx_; ++x) {
                field[grid_.halo_idx(x, -1, z)]        = recv_buf_[2][(max_xz * fieldIdx) + z * local_nx_ + x];
                field[grid_.halo_idx(x, local_ny_, z)] = recv_buf_[3][(max_xz * fieldIdx) + z * local_nx_ + x];
            }
        }
    }
    void UnpackFieldYZ(double* field, size_t fieldIdx) {
        for (int z = 0; z < local_nz_; ++z) {
            for (int y = 0; y < local_ny_; ++y) {
                field[grid_.halo_idx(-1, y, z)]        = recv_buf_[0][(max_yz * fieldIdx) + z * local_ny_ + y];
                field[grid_.halo_idx(local_nx_, y, z)] = recv_buf_[1][(max_yz * fieldIdx) + z * local_ny_ + y];
            }
        }
    }

    void UnpackField(double* field, size_t fieldIdx, int d) {
        if      (d == 0) UnpackFieldYZ(field, fieldIdx);
        else if (d == 1) UnpackFieldXZ(field, fieldIdx);
        else             UnpackFieldXY(field, fieldIdx);
    }

    void ExchangeQTensor(QTensorFields& qf, FluidFields& ff) {
        if (world_size_ == 1) return;
        MPI_Request reqs[12];   // 3 axes × (2 sends + 2 recvs)
        int n = 0;

        const int face_size[3] = {
            static_cast<int>(kMaxFields * max_yz),
            static_cast<int>(kMaxFields * max_xz),
            static_cast<int>(kMaxFields * max_xy)
        };

        for (int d = 0; d < 3; ++d) {
            size_t fieldIdx = 0;
            PackField(qf.qxx.data(), fieldIdx++, d);
            PackField(qf.qxy.data(), fieldIdx++, d);
            PackField(qf.qxz.data(), fieldIdx++, d);
            PackField(qf.qyy.data(), fieldIdx++, d);
            PackField(qf.qyz.data(), fieldIdx++, d);
            PackField(ff.ux.data() , fieldIdx++, d);
            PackField(ff.uy.data() , fieldIdx++, d);
            PackField(ff.uz.data() , fieldIdx++, d);

            MPI_Irecv(recv_buf_[2*d].data()  , face_size[d], MPI_DOUBLE, neighbor_lo_[d], d+3, cart_comm_, &reqs[n++]);
            MPI_Irecv(recv_buf_[2*d+1].data(), face_size[d], MPI_DOUBLE, neighbor_hi_[d], d  , cart_comm_, &reqs[n++]);
            MPI_Isend(send_buf_[2*d].data()  , face_size[d], MPI_DOUBLE, neighbor_lo_[d], d  , cart_comm_, &reqs[n++]);
            MPI_Isend(send_buf_[2*d+1].data(), face_size[d], MPI_DOUBLE, neighbor_hi_[d], d+3, cart_comm_, &reqs[n++]);
        }

        MPI_Waitall(12, reqs, MPI_STATUSES_IGNORE);

        for (int d = 0; d < 3; ++d) {
            size_t fieldIdx = 0;
            UnpackField(qf.qxx.data(), fieldIdx++, d);
            UnpackField(qf.qxy.data(), fieldIdx++, d);
            UnpackField(qf.qxz.data(), fieldIdx++, d);
            UnpackField(qf.qyy.data(), fieldIdx++, d);
            UnpackField(qf.qyz.data(), fieldIdx++, d);
            UnpackField(ff.ux.data() , fieldIdx++, d);
            UnpackField(ff.uy.data() , fieldIdx++, d);
            UnpackField(ff.uz.data() , fieldIdx++, d);
        }
    }

    void ExchangePassiveStresses(QTensorFields& qf) {
        if (world_size_ == 1) return;
        MPI_Request reqs[12];
        int n = 0;

        const int face_size[3] = {
            static_cast<int>(kMaxFields * max_yz),
            static_cast<int>(kMaxFields * max_xz),
            static_cast<int>(kMaxFields * max_xy)
        };

        for (int d = 0; d < 3; ++d) {
            size_t fieldIdx = 0;
            PackField(qf.qxx.data(), fieldIdx++, d);
            PackField(qf.qxy.data(), fieldIdx++, d);
            PackField(qf.qxz.data(), fieldIdx++, d);
            PackField(qf.qyy.data(), fieldIdx++, d);
            PackField(qf.qyz.data(), fieldIdx++, d);
            PackField(qf.Sigma_xx.data(), fieldIdx++, d);
            PackField(qf.Sigma_xy.data(), fieldIdx++, d);
            PackField(qf.Sigma_xz.data(), fieldIdx++, d);
            PackField(qf.Sigma_yy.data(), fieldIdx++, d);
            PackField(qf.Sigma_yz.data(), fieldIdx++, d);
            // Antisymmetric part: phase 2's PassiveStressDivergence reads it at
            // neighbours just as it does S, so it needs a valid halo too.
            PackField(qf.Tau_xy.data(), fieldIdx++, d);
            PackField(qf.Tau_xz.data(), fieldIdx++, d);
            PackField(qf.Tau_yz.data(), fieldIdx++, d);

            MPI_Irecv(recv_buf_[2*d].data()  , face_size[d], MPI_DOUBLE, neighbor_lo_[d], d+3, cart_comm_, &reqs[n++]);
            MPI_Irecv(recv_buf_[2*d+1].data(), face_size[d], MPI_DOUBLE, neighbor_hi_[d], d  , cart_comm_, &reqs[n++]);
            MPI_Isend(send_buf_[2*d].data()  , face_size[d], MPI_DOUBLE, neighbor_lo_[d], d  , cart_comm_, &reqs[n++]);
            MPI_Isend(send_buf_[2*d+1].data(), face_size[d], MPI_DOUBLE, neighbor_hi_[d], d+3, cart_comm_, &reqs[n++]);
        }

        MPI_Waitall(12, reqs, MPI_STATUSES_IGNORE);

        for (int d = 0; d < 3; ++d) {
            size_t fieldIdx = 0;
            UnpackField(qf.qxx.data(), fieldIdx++, d);
            UnpackField(qf.qxy.data(), fieldIdx++, d);
            UnpackField(qf.qxz.data(), fieldIdx++, d);
            UnpackField(qf.qyy.data(), fieldIdx++, d);
            UnpackField(qf.qyz.data(), fieldIdx++, d);
            UnpackField(qf.Sigma_xx.data(), fieldIdx++, d);
            UnpackField(qf.Sigma_xy.data(), fieldIdx++, d);
            UnpackField(qf.Sigma_xz.data(), fieldIdx++, d);
            UnpackField(qf.Sigma_yy.data(), fieldIdx++, d);
            UnpackField(qf.Sigma_yz.data(), fieldIdx++, d);
            UnpackField(qf.Tau_xy.data(), fieldIdx++, d);
            UnpackField(qf.Tau_xz.data(), fieldIdx++, d);
            UnpackField(qf.Tau_yz.data(), fieldIdx++, d);
        }
    }
};

#else

struct HaloExchangeQTensor {
    explicit HaloExchangeQTensor(LocalGrid = LocalGrid::SingleRank()) {}
    explicit HaloExchangeQTensor(const LocalGrid&, const MPIContext&) {}

    void ExchangeQTensor(QTensorFields&, FluidFields&) {}
    void ExchangePassiveStresses(QTensorFields&) {}
};

#endif

#endif // LBM_AN_MPI_HALO_EXCHANGE_QTENSOR_H_
