cmake_minimum_required(VERSION 3.23)

project(LBM)

# Probe for CUDA before committing to a C++ standard: nvcc does not support C++23.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA found — using C++20 (nvcc does not support C++23)")
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CUDA_STANDARD 20)
else()
    message(STATUS "CUDA not found — using C++23")
    set(CMAKE_CXX_STANDARD 23)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-fopenmp -g -Ofast")

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

# Fetch Kokkos mdspan
FetchContent_Declare(
        mdspan
        GIT_REPOSITORY https://github.com/kokkos/mdspan.git
        GIT_TAG stable
)
FetchContent_MakeAvailable(mdspan)

# Fetch {fmt} only when the standard library lacks <format>/<print>
if(NOT HAVE_STD_FORMAT)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 12.1.0
    )
    FetchContent_MakeAvailable(fmt)
endif()

# message("mdspan_SOURCE_DIR is set to ${mdspan_SOURCE_DIR}")
# message("mdspan_INCLUDE_DIRS is set to ${mdspan_INCLUDE_DIR}")
find_package(HDF5 REQUIRED COMPONENTS C)

foreach(target main benchmark)
    add_executable(${target})
    target_include_directories(${target} PUBLIC
            ${HDF5_INCLUDE_DIRS}
            ${mdspan_SOURCE_DIR}
            src/
    )
    target_link_libraries(${target} PRIVATE
        ${HDF5_C_LIBRARIES}
        mdspan
    )
    if(HAVE_STD_FORMAT)
        target_compile_definitions(${target} PRIVATE HAVE_STD_FORMAT)
    else()
        target_link_libraries(${target} PRIVATE fmt::fmt)
    endif()
endforeach()

add_subdirectory(src)

