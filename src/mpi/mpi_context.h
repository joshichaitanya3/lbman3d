#ifndef LBM_AN_MPI_MPI_CONTEXT_H_
#define LBM_AN_MPI_MPI_CONTEXT_H_

#include "local_grid.h"
#ifdef LBM_ENABLE_MPI
#include <mpi.h>
#include <array>

struct MPIContext {
    bool owns_mpi_; // Did we start this MPI instance?
    int world_rank, world_size;
    int dims[3];          // {px, py, pz}: ranks per axis, filled by MPI_Dims_create
    int coords[3];        // {ix, iy, iz}: this rank's position in the 3D grid
    MPI_Comm cart_comm;
    MPI_Info info;
    // periods: whether each axis wraps. Pass {1,1,1} for fully periodic,
    // {0,0,0} for walled — MPI_Cart_shift returns MPI_PROC_NULL at physical edges.
    explicit MPIContext(std::array<int, 3> periods = {1, 1, 1}) {
        int initialized;
        MPI_Initialized(&initialized);
        owns_mpi_ = !initialized;
        if (owns_mpi_) MPI_Init(NULL, NULL);

        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        info = MPI_INFO_NULL;

        dims[0] = dims[1] = dims[2] = 0;              // 0 = let MPI decide
        MPI_Dims_create(world_size, 3, dims);

        int reorder = 1;                               // allow topology-aware rank assignment
        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods.data(), reorder, &cart_comm);
        MPI_Comm_rank(cart_comm, &world_rank);         // re-query: reorder may have changed it
        MPI_Cart_coords(cart_comm, world_rank, 3, coords);
    }

    // These default to MPI_COMM_WORLD, which happens to contain the same
    // ranks as cart_comm today (cart_comm is built from the entirety of
    // MPI_COMM_WORLD, with no sub-communicator splitting). If that ever
    // changes (e.g. ensemble runs splitting MPI_COMM_WORLD across
    // independent simulations), pass the relevant MPIContext's cart_comm
    // explicitly instead of relying on the default.
    static void SumDoubles(double* local_sum, double* global_sum, MPI_Comm comm = MPI_COMM_WORLD) {
        MPI_Allreduce(local_sum, global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    }

    static void SumInts(int* local_sum, int* global_sum, MPI_Comm comm = MPI_COMM_WORLD) {
        MPI_Allreduce(local_sum, global_sum, 1, MPI_INT, MPI_SUM, comm);
    }

    // True on exactly one rank in `comm` — the one responsible for file I/O
    // (log file, exports) so every rank doesn't write/truncate the same path.
    static bool IsRoot(MPI_Comm comm = MPI_COMM_WORLD) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        return rank == 0;
    }

    LocalGrid MakeLocalGrid() const {
        auto local_dim = [](int global, int n, int coord) {
            return global / n + (coord < global % n ? 1 : 0);
        };
        auto offset = [](int global, int n, int coord) {
            return coord * (global / n) + std::min(coord, global % n);
        };
        return {
            local_dim(Params::nx, dims[0], coords[0]),
            local_dim(Params::ny, dims[1], coords[1]),
            local_dim(Params::nz, dims[2], coords[2]),
            offset(Params::nx, dims[0], coords[0]),
            offset(Params::ny, dims[1], coords[1]),
            offset(Params::nz, dims[2], coords[2]),
            1
        };
    }

    ~MPIContext() {
        int finalized;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Comm_free(&cart_comm);  // only reached if MPI is still live. Always free our communicator
            if (owns_mpi_) MPI_Finalize(); // only finalize if we were the ones who initialized
        }
    }

};

#else

struct MPIContext {
    int world_rank = 0, world_size = 1;
    int dims[3]   = {1, 1, 1};
    int coords[3] = {0, 0, 0};

    static void SumDoubles(double* local_sum, double* global_sum) {
        *global_sum = *local_sum;
    }

    static void SumInts(int* local_sum, int* global_sum) {
        *global_sum = *local_sum;
    }

    static bool IsRoot() { return true; }

    LocalGrid MakeLocalGrid() const {
        auto local_dim = [](int global, int n, int coord) {
            return global / n + (coord < global % n ? 1 : 0);
        };
        auto offset = [](int global, int n, int coord) {
            return coord * (global / n) + std::min(coord, global % n);
        };
        return {
            local_dim(Params::nx, dims[0], coords[0]),
            local_dim(Params::ny, dims[1], coords[1]),
            local_dim(Params::nz, dims[2], coords[2]),
            offset(Params::nx, dims[0], coords[0]),
            offset(Params::ny, dims[1], coords[1]),
            offset(Params::nz, dims[2], coords[2]),
            0
        };
    }

};

#endif

#endif // LBM_AN_MPI_MPI_CONTEXT_H_
