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

