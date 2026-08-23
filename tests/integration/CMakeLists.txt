# Create test executable
add_executable(poiseuille_test
    test_poiseuille.cc
)

target_sources(poiseuille_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/fluid_fields.cc
    ${CMAKE_SOURCE_DIR}/src/qtensor_fields.cc
)

lbm_add_test(poiseuille_test poiseuille)

add_executable(qtrelax_test
    test_qtensor_relaxation.cc
)

target_sources(qtrelax_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/fluid_fields.cc
    ${CMAKE_SOURCE_DIR}/src/qtensor_fields.cc
    ${CMAKE_SOURCE_DIR}/src/analysis_fields.cc
)

lbm_add_test(qtrelax_test qtrelax)

add_executable(bc_combinations_test
    test_bc_combinations.cc
)

target_sources(bc_combinations_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/fluid_fields.cc
    ${CMAKE_SOURCE_DIR}/src/qtensor_fields.cc
)

lbm_add_test(bc_combinations_test bc_check)

add_executable(coupled_backflow_test
    test_coupled_backflow.cc
)

target_sources(coupled_backflow_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/fluid_fields.cc
    ${CMAKE_SOURCE_DIR}/src/qtensor_fields.cc
    ${CMAKE_SOURCE_DIR}/src/analysis_fields.cc
)

lbm_add_test(coupled_backflow_test coupled)

# ImageDataWriter → ImageDataReader roundtrip. No physics — reuses the unit
# params (8³ grid, single OMP thread). The test does touch the HDF5 write
# path, so it lives here rather than in unit_tests (per tests/CLAUDE.md
# guidance to keep unit tests off the HDF5 write path).
add_executable(vtkhdf_roundtrip_test
    test_vtkhdf_roundtrip.cc
)

lbm_add_test(vtkhdf_roundtrip_test unit)
