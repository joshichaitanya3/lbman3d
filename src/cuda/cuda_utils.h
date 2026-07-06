#pragma once

#include <stdexcept>
#include <string>
#include <format>

inline void checkCuda(cudaError_t err, const char* func, const char* file, int line) {
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::format("{}({}) \"{}\": [{}] {}", file, line, func, static_cast<int>(err), cudaGetErrorString(err)));
}
#define checkCudaErrors(err) checkCuda(err, #err, __FILE__, __LINE__)
