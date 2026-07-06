
if(CMAKE_CUDA_COMPILER)
    foreach(target main benchmark)
        target_sources(${target}
            PRIVATE
            device_fields.cu
        )
    endforeach()
endif()