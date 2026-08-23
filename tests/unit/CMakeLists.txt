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
    test_flow_alignment_trace.cc
    test_qtensor_init.cc
    test_upwind_advection.cc
    test_boundary_names.cc
    test_sim_config_validate.cc
    test_order_director_roundtrip.cc
    # Add more test files as needed
)

target_sources(unit_tests
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/analysis_fields.cc
    ${CMAKE_SOURCE_DIR}/src/qtensor_fields.cc   # QTensorFields ctor, for Initialize()
)

lbm_add_test(unit_tests unit)
