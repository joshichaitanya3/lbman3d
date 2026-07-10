#ifndef LBM_AN_QTENSOR_FIELDS_H_
#define LBM_AN_QTENSOR_FIELDS_H_

#include <vector>
#include "params.h"

// Owns Q-tensor state: current and scratch (new) components.
// Flat, row-major storage indexed via idx(x,y,z) from physics_helpers.h.
struct QTensorFields {
    std::vector<double> qxx, qxy, qxz, qyy, qyz;
    std::vector<double> qxx_new, qxy_new, qxz_new, qyy_new, qyz_new;

    // Nematic stress tensor (active + passive)
    std::vector<double> Pxx, Pxy, Pxz, Pyy, Pyz;

    double nematic_energy = 0.0;

    QTensorFields();

    void SwapWithNew() {
        std::swap(qxx, qxx_new);
        std::swap(qxy, qxy_new);
        std::swap(qxz, qxz_new);
        std::swap(qyy, qyy_new);
        std::swap(qyz, qyz_new);
    }
};

#endif // LBM_AN_QTENSOR_FIELDS_H_
