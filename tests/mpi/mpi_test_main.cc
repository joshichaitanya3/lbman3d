// tests/mpi/mpi_test_main.cc
#include <mpi.h>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int local = RUN_ALL_TESTS();
    int global;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global;   // non-zero if any rank failed any test
}
