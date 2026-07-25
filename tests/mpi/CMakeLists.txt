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

    add_executable(test_halo_exchange
        test_halo_exchange.cc
        mpi_test_main.cc
        ${CMAKE_SOURCE_DIR}/src/fluid_fields.cc
        ${CMAKE_SOURCE_DIR}/src/qtensor_fields.cc
    )
    target_link_libraries(test_halo_exchange PRIVATE MPI::MPI_CXX GTest::gtest)
    target_include_directories(test_halo_exchange PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/params/unit   # shadows src/params.h with 8×8×8 grid
        ${CMAKE_SOURCE_DIR}/src/
    )
    target_compile_definitions(test_halo_exchange PRIVATE LBM_ENABLE_MPI)

    add_test(NAME mpi_halo_np1
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                $<TARGET_FILE:test_halo_exchange>)

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

    add_test(NAME mpi_halo_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                $<TARGET_FILE:test_exchange_correctness>)
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
