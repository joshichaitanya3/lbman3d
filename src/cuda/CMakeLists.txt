
if(LBM_USE_CUDA)
    foreach(target main benchmark)
        target_sources(${target}
            PRIVATE
            device_fields.cu
        )
    endforeach()
endif()