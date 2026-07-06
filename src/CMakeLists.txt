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

add_subdirectory(analysis)
add_subdirectory(cuda)
