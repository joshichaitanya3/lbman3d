foreach(target main benchmark)
    target_sources(${target}
        PRIVATE
        defect_fields.cc
    )
endforeach()
