if(LBM_ENABLE_MPI)
    add_executable(test_mpi_context
        test_mpi_context.cc
        mpi_test_main.cc
    )
    target_link_libraries(test_mpi_context PRIVATE MPI::MPI_CXX GTest::gtest)
    target_include_directories(test_mpi_context PRIVATE ${CMAKE_SOURCE_DIR}/src/)
    target_compile_definitions(test_mpi_context PRIVATE LBM_ENABLE_MPI)

    add_test(NAME mpi_context_np1
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 1
                $<TARGET_FILE:test_mpi_context>)
    add_test(NAME mpi_context_np4
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 4
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
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2
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
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2
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
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 8
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
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2
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
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2
                $<TARGET_FILE:test_poiseuille_mpi>)

    # np=2 exchange round-trip deferred to PR VII when idx() covers halo coordinates

    # PR VII-e — NVSHMEM Q-tensor halo tests. Gated on LBM_ENABLE_NVSHMEM
    # because they call NVSHMEM directly (via HaloExchangeQTensorNvshmem) and
    # the class disappears without that flag. Each test binary is its own
    # executable so mpirun -n N picks up NVSHMEM's per-rank GPU affinity;
    # gtest_discover_tests is not used (matches every other MPI target above —
    # see "Do not use gtest_discover_tests for MPI targets" in src/mpi/CLAUDE.md).
    if(LBM_ENABLE_NVSHMEM)
        # On Cray/NERSC, find_package(MPI) leaves MPI_CXX_INCLUDE_DIRS empty
        # because the Cray wrapper compiler (CC/mpicxx) injects the path itself.
        # nvcc does not go through that wrapper, so locate mpi.h explicitly via
        # the environment variables set by the cray-mpich module.
        if(NOT MPI_CXX_INCLUDE_DIRS)
            find_path(_lbm_cray_mpi_inc mpi.h
                HINTS
                    $ENV{CRAY_MPICH_DIR}/include
                    $ENV{MPICH_DIR}/include
                    $ENV{MPI_HOME}/include
                    $ENV{MPI_ROOT}/include
                NO_DEFAULT_PATH
            )
            if(_lbm_cray_mpi_inc)
                message(STATUS
                    "NVSHMEM tests: MPI_CXX_INCLUDE_DIRS empty; using ${_lbm_cray_mpi_inc} (from env)")
                set(MPI_CXX_INCLUDE_DIRS "${_lbm_cray_mpi_inc}")
            endif()
        endif()

        set(_lbm_nvshmem_test_common_sources
            mpi_test_main.cc
            ${CMAKE_SOURCE_DIR}/src/fluid_fields.cc
            ${CMAKE_SOURCE_DIR}/src/qtensor_fields.cc
            ${CMAKE_SOURCE_DIR}/src/cuda/nvshmem_halo.cu
            ${CMAKE_SOURCE_DIR}/src/cuda/device_fields.cu
        )
        set(_lbm_nvshmem_test_include_dirs
            ${CMAKE_SOURCE_DIR}/tests/params/unit
            ${CMAKE_SOURCE_DIR}/src/
            ${CMAKE_BINARY_DIR}/generated
            ${MPI_CXX_INCLUDE_DIRS}
        )
        foreach(_lbm_nvshmem_test IN ITEMS test_pack_qtensor_nvshmem test_exchange_qtensor_nvshmem)
            add_executable(${_lbm_nvshmem_test}
                ${_lbm_nvshmem_test}.cu
                ${_lbm_nvshmem_test_common_sources}
            )
            target_link_libraries(${_lbm_nvshmem_test} PRIVATE
                MPI::MPI_CXX GTest::gtest CUDA::toolkit
                nvshmem::nvshmem_host nvshmem::nvshmem_device
            )
            target_include_directories(${_lbm_nvshmem_test} PRIVATE
                ${_lbm_nvshmem_test_include_dirs}
            )
            target_compile_definitions(${_lbm_nvshmem_test} PRIVATE
                LBM_ENABLE_MPI LBM_ENABLE_NVSHMEM SIM_WITH_CUDA
            )
            # nvshmem_halo.cu needs -rdc=true (device-side nvshmem_my_pe use);
            # scope the flag to that TU inside this target — mirrors the same
            # set_source_files_properties call in src/cuda/CMakeLists.txt.
            set_source_files_properties(
                ${CMAKE_SOURCE_DIR}/src/cuda/nvshmem_halo.cu
                TARGET_DIRECTORY ${_lbm_nvshmem_test}
                PROPERTIES COMPILE_FLAGS "-rdc=true"
            )
            set_target_properties(${_lbm_nvshmem_test} PROPERTIES
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
            )
            # device_fields.h -> format_compat.h; match the format backend used
            # by the rest of the build (test_poiseuille_mpi does the same).
            if(HAVE_STD_FORMAT)
                target_compile_definitions(${_lbm_nvshmem_test} PRIVATE HAVE_STD_FORMAT)
            else()
                target_link_libraries(${_lbm_nvshmem_test} PRIVATE fmt::fmt)
            endif()
        endforeach()

        # Pack test runs with a single PE — the pack kernel is purely local and
        # nranks=1 is where a wrong slot layout is loudest (nranks=2 would hide
        # it if pack and unpack shared the same wrong stride).
        add_test(NAME nvshmem_pack_qtensor_np1
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 1
                    $<TARGET_FILE:test_pack_qtensor_nvshmem>)

        # Halo round-trip needs a real neighbour. np=1 (self-put) is a useful
        # smoke check on the dev box; np=2 is the actual VII-e correctness gate.
        add_test(NAME nvshmem_exchange_qtensor_np1
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 1
                    $<TARGET_FILE:test_exchange_qtensor_nvshmem>)
        add_test(NAME nvshmem_exchange_qtensor_np2
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2
                    $<TARGET_FILE:test_exchange_qtensor_nvshmem>)
    endif()

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
    # vtkhdf_writer.h pulls in format_compat.h; match the format backend the
    # rest of the build uses.
    if(HAVE_STD_FORMAT)
        target_compile_definitions(test_vtkhdf_write PRIVATE HAVE_STD_FORMAT)
    else()
        target_link_libraries(test_vtkhdf_write PRIVATE fmt::fmt)
    endif()

    # np=1: single-rank baseline; np=2,3: even and odd decompositions;
    # np=5: uneven split (8 cells / 5 ranks) — exercises shared-chunk writes
    foreach(np 1 2 3 5)
        add_test(NAME vtkhdf_write_np${np}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} ${np}
                    $<TARGET_FILE:test_vtkhdf_write>)
    endforeach()

else()
    add_executable(test_mpi_context_stub test_mpi_context_stub.cc)
    target_link_libraries(test_mpi_context_stub PRIVATE GTest::gtest_main)
    target_include_directories(test_mpi_context_stub PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME mpi_context_stub COMMAND test_mpi_context_stub)
endif()
