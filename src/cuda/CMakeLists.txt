
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
        #
        # -diag-suppress=3012,3013 — root cause: NVSHMEM 3.2.x device headers
        # (barrier.cuh, reduce.cuh, wait_until_apis.cuh) use two patterns
        # that C++20 P1152 "Deprecating volatile" made deprecated: (a)
        # `volatile T*` function parameters in nvshmemi_wait_until_*, and (b)
        # compound assignment `sync_counter[0] += 1` where sync_counter is
        # volatile in the barrier/reduce collective paths. We compile with
        # -std=c++20 (nvcc doesn't support C++23), so cudafe1 emits #3013
        # and #3012 respectively. The templates are instantiated whether or
        # not we call them, because -rdc=true triggers device-symbol
        # resolution across all templates in the included headers.
        #
        # This is not fixable in our code:
        #   - Dropping this TU to -std=c++17 avoids the deprecation, but
        #     cuda_utils.h transitively uses std::format (C++20), so the TU
        #     genuinely needs C++20.
        #   - -isystem on NVSHMEM's include dir would suppress warnings by
        #     origin rather than by ID, but nvcc's cudafe1 does not reliably
        #     honor system-header status for these particular diagnostics
        #     (behaviour varies across CUDA versions on the cluster).
        #   - Upstream fix: NVSHMEM 3.4+ removes the deprecated volatile
        #     usage; NERSC's install is pinned at 3.2.5-1 as of this writing.
        #
        # -diag-suppress by ID is therefore the correct local action. Scope
        # is this TU only (the only one that includes NVSHMEM device
        # headers), so any of our own code that ever uses volatile in a way
        # C++20 flags will still warn from every other TU.
        set_source_files_properties(nvshmem_halo.cu
            TARGET_DIRECTORY main benchmark
            PROPERTIES COMPILE_FLAGS "-rdc=true -diag-suppress=3012 -diag-suppress=3013"
        )
    endif()
endif()
