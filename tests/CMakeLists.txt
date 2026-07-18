function(lbm_add_test target params_dir)
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/params/${params_dir}  # shadows src/params.h
        ${CMAKE_SOURCE_DIR}/src
        ${HDF5_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PRIVATE GTest::gtest_main ${HDF5_C_LIBRARIES} lbm_warnings)

    if(HAVE_STD_FORMAT)
        target_compile_definitions(${target} PRIVATE HAVE_STD_FORMAT)
    else()
        target_link_libraries(${target} PRIVATE fmt::fmt)
    endif()
    if(LBM_USE_CUDA)
        target_compile_definitions(${target} PRIVATE SIM_WITH_CUDA)
        target_link_libraries(${target} PRIVATE CUDA::toolkit)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --use_fast_math>
        )
    endif()
    gtest_discover_tests(${target})
endfunction()

add_subdirectory(unit)
add_subdirectory(integration)
