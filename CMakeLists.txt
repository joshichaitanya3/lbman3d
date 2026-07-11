cmake_minimum_required(VERSION 3.23)

project(LBM)

option(LBM_FORCE_CPU "Build CPU-only, even if a CUDA compiler is available" OFF)

# Probe for CUDA before committing to a C++ standard: nvcc does not support C++23.
include(CheckLanguage)
if(NOT LBM_FORCE_CPU)
    check_language(CUDA)
endif()

# Derived once here so every later check (this file and subdirectories) agrees,
# rather than each re-checking CMAKE_CUDA_COMPILER (which stays cached across
# configures — re-checking it directly would ignore LBM_FORCE_CPU on a build
# directory that was previously configured with CUDA enabled).
if(CMAKE_CUDA_COMPILER AND NOT LBM_FORCE_CPU)
    set(LBM_USE_CUDA ON)
else()
    set(LBM_USE_CUDA OFF)
endif()

if(LBM_USE_CUDA)
    # Must be set before enable_language(CUDA) — that call writes the default (52)
    # into the cache, after which CACHE STRING without FORCE has no effect.
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures to compile for")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA found — using C++20 (nvcc does not support C++23)")
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CUDA_STANDARD 20)
    # Device assert() is gated on NDEBUG only, same as host — it doesn't need
    # -G (full device debug info). -G disables device-side register-allocation
    # optimization almost entirely, which blows the register budget for
    # GpuQTensorStep's 512-thread blocks ("too many resources requested for
    # launch"); keep -O3 in both configs and only toggle NDEBUG.
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_DEBUG "-O3")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    if(LBM_FORCE_CPU)
        message(STATUS "LBM_FORCE_CPU=ON — building CPU-only even though CUDA may be available")
    endif()
    message(STATUS "CUDA not found — using C++23")
    set(CMAKE_CXX_STANDARD 23)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -DCMAKE_BUILD_TYPE=Release for -Ofast (NDEBUG also disables idx()'s
# InDomain() bounds-check asserts in physics_helpers.h). Omit, or pass
# =Debug, to keep those asserts live.
set(CMAKE_CXX_FLAGS "-fopenmp -g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-Og")

# Detect whether this toolchain's standard library supports <format> and <print>.
# GCC 13's libstdc++ ships incomplete C++23 headers; GCC 15 is fine.
# Under C++20, <print> is unavailable and the fmt fallback is used automatically.
include(CheckSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-std=c++${CMAKE_CXX_STANDARD}")
check_source_compiles(CXX
    "#include <format>\n#include <print>\nint main(){std::println(\"{}\",42);}"
    HAVE_STD_FORMAT
)

include(FetchContent)

# Fetch {fmt} only when the standard library lacks <format>/<print>
if(NOT HAVE_STD_FORMAT)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 12.1.0
    )
    FetchContent_MakeAvailable(fmt)
endif()

find_package(HDF5 REQUIRED COMPONENTS C)

foreach(target main benchmark)
    add_executable(${target})
    target_include_directories(${target} PUBLIC
            ${HDF5_INCLUDE_DIRS}
            src/
    )
    target_link_libraries(${target} PRIVATE
        ${HDF5_C_LIBRARIES}
    )
    if(HAVE_STD_FORMAT)
        target_compile_definitions(${target} PRIVATE HAVE_STD_FORMAT)
    else()
        target_link_libraries(${target} PRIVATE fmt::fmt)
    endif()
    if(LBM_USE_CUDA)
        target_compile_definitions(${target} PRIVATE SIM_WITH_CUDA)
        target_link_libraries(${target} PRIVATE CUDA::toolkit)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --use_fast_math>
        )
    endif()
endforeach()

add_subdirectory(src)

