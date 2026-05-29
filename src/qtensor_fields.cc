#include "qtensor_fields.h"
#include "params.h"

QTensorFields::QTensorFields() :
    qxx_data     (Params::nx * Params::ny * Params::nz, 0.0),
    qxy_data     (Params::nx * Params::ny * Params::nz, 0.0),
    qxz_data     (Params::nx * Params::ny * Params::nz, 0.0),
    qyy_data     (Params::nx * Params::ny * Params::nz, 0.0),
    qyz_data     (Params::nx * Params::ny * Params::nz, 0.0),
    qxx_new_data (Params::nx * Params::ny * Params::nz, 0.0),
    qxy_new_data (Params::nx * Params::ny * Params::nz, 0.0),
    qxz_new_data (Params::nx * Params::ny * Params::nz, 0.0),
    qyy_new_data (Params::nx * Params::ny * Params::nz, 0.0),
    qyz_new_data (Params::nx * Params::ny * Params::nz, 0.0),
    qxx    (qxx_data.data(),     Params::nz, Params::ny, Params::nx),
    qxy    (qxy_data.data(),     Params::nz, Params::ny, Params::nx),
    qxz    (qxz_data.data(),     Params::nz, Params::ny, Params::nx),
    qyy    (qyy_data.data(),     Params::nz, Params::ny, Params::nx),
    qyz    (qyz_data.data(),     Params::nz, Params::ny, Params::nx),
    qxx_new(qxx_new_data.data(), Params::nz, Params::ny, Params::nx),
    qxy_new(qxy_new_data.data(), Params::nz, Params::ny, Params::nx),
    qxz_new(qxz_new_data.data(), Params::nz, Params::ny, Params::nx),
    qyy_new(qyy_new_data.data(), Params::nz, Params::ny, Params::nx),
    qyz_new(qyz_new_data.data(), Params::nz, Params::ny, Params::nx)
{}
