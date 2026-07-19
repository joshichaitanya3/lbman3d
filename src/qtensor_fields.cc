#include "qtensor_fields.h"
#include <params.h>

QTensorFields::QTensorFields() :
    qxx     (Params::nx * Params::ny * Params::nz, 0.0),
    qxy     (Params::nx * Params::ny * Params::nz, 0.0),
    qxz     (Params::nx * Params::ny * Params::nz, 0.0),
    qyy     (Params::nx * Params::ny * Params::nz, 0.0),
    qyz     (Params::nx * Params::ny * Params::nz, 0.0),
    qxx_new (Params::nx * Params::ny * Params::nz, 0.0),
    qxy_new (Params::nx * Params::ny * Params::nz, 0.0),
    qxz_new (Params::nx * Params::ny * Params::nz, 0.0),
    qyy_new (Params::nx * Params::ny * Params::nz, 0.0),
    qyz_new (Params::nx * Params::ny * Params::nz, 0.0),
    Pxx     (Params::nx * Params::ny * Params::nz, 0.0),
    Pxy     (Params::nx * Params::ny * Params::nz, 0.0),
    Pxz     (Params::nx * Params::ny * Params::nz, 0.0),
    Pyy     (Params::nx * Params::ny * Params::nz, 0.0),
    Pyz     (Params::nx * Params::ny * Params::nz, 0.0)
{}
