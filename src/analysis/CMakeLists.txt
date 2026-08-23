foreach(target main benchmark)
    target_sources(${target}
        PRIVATE
        defect_fields.cc
    )
endforeach()

target_sources(find_defects
    PRIVATE
    defect_fields.cc
    find_defects.cc
)
