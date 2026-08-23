target_sources(main
    PRIVATE
    main.cc
    fluid_fields.cc
    qtensor_fields.cc
    analysis_fields.cc
    sim_io.cc
)

target_sources(benchmark
    PRIVATE
    benchmark.cc
    fluid_fields.cc
    qtensor_fields.cc
    analysis_fields.cc
    sim_io.cc
)

# find_defects doesn't link SimIO (no lbm.log; writes only via
# disclination_io.h), FluidFields (no LBM state), or the CUDA layer.
target_sources(find_defects
    PRIVATE
    qtensor_fields.cc
    analysis_fields.cc
)

add_subdirectory(analysis)
add_subdirectory(cuda)
