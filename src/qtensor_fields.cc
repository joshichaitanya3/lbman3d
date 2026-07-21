#include "qtensor_fields.h"
#include <params.h>
#include "local_grid.h"

QTensorFields::QTensorFields(LocalGrid g) :
    qxx     (g.Volume(), 0.0),
    qxy     (g.Volume(), 0.0),
    qxz     (g.Volume(), 0.0),
    qyy     (g.Volume(), 0.0),
    qyz     (g.Volume(), 0.0),
    qxx_new (g.Volume(), 0.0),
    qxy_new (g.Volume(), 0.0),
    qxz_new (g.Volume(), 0.0),
    qyy_new (g.Volume(), 0.0),
    qyz_new (g.Volume(), 0.0),
    Pxx     (g.Volume(), 0.0),
    Pxy     (g.Volume(), 0.0),
    Pxz     (g.Volume(), 0.0),
    Pyy     (g.Volume(), 0.0),
    Pyz     (g.Volume(), 0.0)
{}
