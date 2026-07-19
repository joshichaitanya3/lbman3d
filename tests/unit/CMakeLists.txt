# Create test executable
add_executable(unit_tests
    test_lattice_stencil.cc
    test_feq_moments.cc
    test_bkg_collide.cc
    test_boundary_handler.cc
    test_idx.cc
    # Add more test files as needed
)

lbm_add_test(unit_tests unit)

