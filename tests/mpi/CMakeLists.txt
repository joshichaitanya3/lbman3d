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

else()
    add_executable(test_mpi_context_stub test_mpi_context_stub.cc)
    target_link_libraries(test_mpi_context_stub PRIVATE GTest::gtest_main)
    target_include_directories(test_mpi_context_stub PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME mpi_context_stub COMMAND test_mpi_context_stub)
endif()
