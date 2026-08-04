#include "qtensor_fields.h"
#include <params.h>
#include "local_grid.h"

QTensorFields::QTensorFields(LocalGrid g) :
    grid    (g),
    qxx     (g.HaloVolume(), 0.0),
    qxy     (g.HaloVolume(), 0.0),
    qxz     (g.HaloVolume(), 0.0),
    qyy     (g.HaloVolume(), 0.0),
    qyz     (g.HaloVolume(), 0.0),
    qxx_new (g.HaloVolume(), 0.0),
    qxy_new (g.HaloVolume(), 0.0),
    qxz_new (g.HaloVolume(), 0.0),
    qyy_new (g.HaloVolume(), 0.0),
    qyz_new (g.HaloVolume(), 0.0),
    Sigma_xx     (g.HaloVolume(), 0.0),
    Sigma_xy     (g.HaloVolume(), 0.0),
    Sigma_xz     (g.HaloVolume(), 0.0),
    Sigma_yy     (g.HaloVolume(), 0.0),
    Sigma_yz     (g.HaloVolume(), 0.0),
    Tau_xy     (g.HaloVolume(), 0.0),
    Tau_xz     (g.HaloVolume(), 0.0),
    Tau_yz     (g.HaloVolume(), 0.0)
{}
