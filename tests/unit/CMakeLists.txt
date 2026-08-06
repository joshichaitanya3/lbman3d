# Create test executable
add_executable(unit_tests
    test_lattice_stencil.cc
    test_feq_moments.cc
    test_bkg_collide.cc
    test_boundary_handler.cc
    test_idx.cc
    test_offsets.cc
    test_anchored_q.cc
    test_backflow_contraction.cc
    test_antisym_stress.cc
    test_qtensor_init.cc
    # Add more test files as needed
)

target_sources(unit_tests
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/analysis_fields.cc
    ${CMAKE_SOURCE_DIR}/src/qtensor_fields.cc   # QTensorFields ctor, for Initialize()
)

lbm_add_test(unit_tests unit)
