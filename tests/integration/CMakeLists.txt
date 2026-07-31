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
