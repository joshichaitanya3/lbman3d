#ifndef LBM_AN_QTENSOR_FIELDS_H_
#define LBM_AN_QTENSOR_FIELDS_H_

#include <vector>
#include <mdspan/mdspan.hpp>
#include "params.h"

// Owns Q-tensor state: current and scratch (new) components.
struct QTensorFields {
    std::vector<double> qxx_data, qxy_data, qxz_data, qyy_data, qyz_data;
    std::vector<double> qxx_new_data, qxy_new_data, qxz_new_data, qyy_new_data, qyz_new_data;

    using ext3_t  = Kokkos::extents<int, Params::nx, Params::ny, Params::nz>;
    Kokkos::mdspan<double, ext3_t> qxx, qxy, qxz, qyy, qyz;
    Kokkos::mdspan<double, ext3_t> qxx_new, qxy_new, qxz_new, qyy_new, qyz_new;

    QTensorFields();

    // Atomically swap each component's data and view together so the mdspans
    // remain consistent with their backing vectors.
    void SwapWithNew() {
        SwapComponent(qxx_data, qxx_new_data, qxx, qxx_new);
        SwapComponent(qxy_data, qxy_new_data, qxy, qxy_new);
        SwapComponent(qxz_data, qxz_new_data, qxz, qxz_new);
        SwapComponent(qyy_data, qyy_new_data, qyy, qyy_new);
        SwapComponent(qyz_data, qyz_new_data, qyz, qyz_new);
    }

private:
    static void SwapComponent(std::vector<double>& a_data, std::vector<double>& b_data,
                               Kokkos::mdspan<double, ext3_t>& a_view,
                               Kokkos::mdspan<double, ext3_t>& b_view) {
        std::swap(a_data, b_data);
        a_view = Kokkos::mdspan<double, ext3_t>(a_data.data(), Params::nx, Params::ny, Params::nz);
        b_view = Kokkos::mdspan<double, ext3_t>(b_data.data(), Params::nx, Params::ny, Params::nz);
    }
};

#endif // LBM_AN_QTENSOR_FIELDS_H_
