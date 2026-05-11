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
    qxx    (qxx_data.data(),     Params::nx, Params::ny, Params::nz),
    qxy    (qxy_data.data(),     Params::nx, Params::ny, Params::nz),
    qxz    (qxz_data.data(),     Params::nx, Params::ny, Params::nz),
    qyy    (qyy_data.data(),     Params::nx, Params::ny, Params::nz),
    qyz    (qyz_data.data(),     Params::nx, Params::ny, Params::nz),
    qxx_new(qxx_new_data.data(), Params::nx, Params::ny, Params::nz),
    qxy_new(qxy_new_data.data(), Params::nx, Params::ny, Params::nz),
    qxz_new(qxz_new_data.data(), Params::nx, Params::ny, Params::nz),
    qyy_new(qyy_new_data.data(), Params::nx, Params::ny, Params::nz),
    qyz_new(qyz_new_data.data(), Params::nx, Params::ny, Params::nz)
{}
