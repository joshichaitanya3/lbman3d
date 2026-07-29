if(LBM_ENABLE_MPI)
    add_executable(test_mpi_context
        test_mpi_context.cc
        mpi_test_main.cc
    )
    target_link_libraries(test_mpi_context PRIVATE MPI::MPI_CXX GTest::gtest)
    target_include_directories(test_mpi_context PRIVATE ${CMAKE_SOURCE_DIR}/src/)
    target_compile_definitions(test_mpi_context PRIVATE LBM_ENABLE_MPI)

    add_test(NAME mpi_context_np1
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                $<TARGET_FILE:test_mpi_context>)
    add_test(NAME mpi_context_np4
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                $<TARGET_FILE:test_mpi_context>)

    add_executable(test_halo_exchange_qtensor
        test_halo_exchange_qtensor.cc
        mpi_test_main.cc
        ${CMAKE_SOURCE_DIR}/src/fluid_fields.cc
        ${CMAKE_SOURCE_DIR}/src/qtensor_fields.cc
    )
    target_link_libraries(test_halo_exchange_qtensor PRIVATE MPI::MPI_CXX GTest::gtest)
    target_include_directories(test_halo_exchange_qtensor PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/params/unit   # shadows src/params.h with 8×8×8 grid
        ${CMAKE_SOURCE_DIR}/src/
    )
    target_compile_definitions(test_halo_exchange_qtensor PRIVATE LBM_ENABLE_MPI)

    add_test(NAME mpi_halo_qtensor_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                $<TARGET_FILE:test_halo_exchange_qtensor>)

    add_executable(test_halo_exchange_lbm
        test_halo_exchange_lbm.cc
        mpi_test_main.cc
        ${CMAKE_SOURCE_DIR}/src/fluid_fields.cc
        ${CMAKE_SOURCE_DIR}/src/qtensor_fields.cc
    )
    target_link_libraries(test_halo_exchange_lbm PRIVATE MPI::MPI_CXX GTest::gtest)
    target_include_directories(test_halo_exchange_lbm PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/params/unit   # shadows src/params.h with 8×8×8 grid
        ${CMAKE_SOURCE_DIR}/src/
    )
    target_compile_definitions(test_halo_exchange_lbm PRIVATE LBM_ENABLE_MPI)

    add_test(NAME mpi_halo_lbm_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                $<TARGET_FILE:test_halo_exchange_lbm>)

    # PR VI step 3b — multi-axis corner sweep (z -> y -> x). Body-diagonal
    # pops routed via the sequential axis sweep must arrive at the diagonal
    # rank's owned origin.
    add_executable(test_corner_sweep
        test_corner_sweep.cc
        mpi_test_main.cc
        ${CMAKE_SOURCE_DIR}/src/fluid_fields.cc
        ${CMAKE_SOURCE_DIR}/src/qtensor_fields.cc
    )
    target_link_libraries(test_corner_sweep PRIVATE MPI::MPI_CXX GTest::gtest)
    target_include_directories(test_corner_sweep PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/params/unit   # shadows src/params.h with 8×8×8 grid
        ${CMAKE_SOURCE_DIR}/src/
    )
    target_compile_definitions(test_corner_sweep PRIVATE LBM_ENABLE_MPI)

    add_test(NAME mpi_corner_sweep_np8
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8
                $<TARGET_FILE:test_corner_sweep>)

    add_executable(test_exchange_correctness
        test_exchange_correctness.cc
        mpi_test_main.cc
        ${CMAKE_SOURCE_DIR}/src/fluid_fields.cc
        ${CMAKE_SOURCE_DIR}/src/qtensor_fields.cc
    )
    target_link_libraries(test_exchange_correctness PRIVATE MPI::MPI_CXX GTest::gtest)
    target_include_directories(test_exchange_correctness PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/params/unit   # shadows src/params.h with 8×8×8 grid
        ${CMAKE_SOURCE_DIR}/src/
    )
    target_compile_definitions(test_exchange_correctness PRIVATE LBM_ENABLE_MPI)

    add_test(NAME mpi_halo_exchange_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                $<TARGET_FILE:test_exchange_correctness>)

    add_executable(test_poiseuille_mpi
        test_poiseuille_mpi.cc
        mpi_test_main.cc
        ${CMAKE_SOURCE_DIR}/src/fluid_fields.cc
        ${CMAKE_SOURCE_DIR}/src/qtensor_fields.cc
    )
    target_link_libraries(test_poiseuille_mpi PRIVATE MPI::MPI_CXX GTest::gtest)
    target_include_directories(test_poiseuille_mpi PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/params/poiseuille   # shadows src/params.h
        ${CMAKE_SOURCE_DIR}/src/
    )
    target_compile_definitions(test_poiseuille_mpi PRIVATE LBM_ENABLE_MPI)
    # device_fields.h pulls in format_compat.h; match the format backend the
    # rest of the build uses.
    if(HAVE_STD_FORMAT)
        target_compile_definitions(test_poiseuille_mpi PRIVATE HAVE_STD_FORMAT)
    else()
        target_link_libraries(test_poiseuille_mpi PRIVATE fmt::fmt)
    endif()

    add_test(NAME mpi_poiseuille_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                $<TARGET_FILE:test_poiseuille_mpi>)

    # np=2 exchange round-trip deferred to PR VII when idx() covers halo coordinates

    add_executable(test_vtkhdf_write
        test_vtkhdf_write.cc
        mpi_test_main.cc
    )
    target_link_libraries(test_vtkhdf_write PRIVATE MPI::MPI_CXX GTest::gtest ${HDF5_C_LIBRARIES})
    target_include_directories(test_vtkhdf_write PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/params/unit   # shadows src/params.h with 8x8x8 grid
        ${CMAKE_SOURCE_DIR}/src/
        ${HDF5_INCLUDE_DIRS}
    )
    target_compile_definitions(test_vtkhdf_write PRIVATE LBM_ENABLE_MPI)

    # np=1: single-rank baseline; np=2,3: even and odd decompositions;
    # np=5: uneven split (8 cells / 5 ranks) — exercises shared-chunk writes
    foreach(np 1 2 3 5)
        add_test(NAME vtkhdf_write_np${np}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                    $<TARGET_FILE:test_vtkhdf_write>)
    endforeach()

else()
    add_executable(test_mpi_context_stub test_mpi_context_stub.cc)
    target_link_libraries(test_mpi_context_stub PRIVATE GTest::gtest_main)
    target_include_directories(test_mpi_context_stub PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME mpi_context_stub COMMAND test_mpi_context_stub)
endif()
