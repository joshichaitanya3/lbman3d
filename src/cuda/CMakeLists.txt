
if(LBM_USE_CUDA)
    foreach(target main benchmark)
        target_sources(${target}
            PRIVATE
            device_fields.cu
            device_solver.cu
        )
        if(LBM_ENABLE_NVSHMEM)
            # halo_pack_kernels.cu — transport-agnostic pack/unpack kernels,
            # no NVSHMEM device API, no -rdc=true.
            # nvshmem_halo.cu — NVSHMEM transport only; all NVSHMEM device-API
            # surface is isolated here so the rest of the build stays on
            # whole-program compilation. CUDA_RESOLVE_DEVICE_SYMBOLS on the
            # executable target (top-level CMakeLists.txt) resolves -rdc=true
            # symbols at link time.
            target_sources(${target} PRIVATE halo_pack_kernels.cu nvshmem_halo.cu)
        endif()
    endforeach()

    if(LBM_ENABLE_NVSHMEM)
        # Source-file properties are scoped to the directory that *defines the
        # target*, not the one that sets the property (CMake docs, "source file
        # properties"). `main` and `benchmark` are declared at the top level, so
        # without TARGET_DIRECTORY the property attaches to a source-in-this-dir
        # phantom that no target actually compiles — nvcc runs without -rdc and
        # the build silently "works" only because nvshmem_my_pe is header-
        # inlined. TARGET_DIRECTORY (CMake ≥3.18) routes the property to the
        # directory scope those targets live in.
        set_source_files_properties(nvshmem_halo.cu
            TARGET_DIRECTORY main benchmark
            PROPERTIES COMPILE_FLAGS "-rdc=true"
        )
    endif()
endif()
