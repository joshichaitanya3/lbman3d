#include "fluid_fields.h"
#include <params.h>
#include "lattice_stencil.h"
#include "local_grid.h"

FluidFields::FluidFields(LocalGrid g) :
    f       (g.Volume() * Lattice::ndir, 0.0),
    f_new   (g.Volume() * Lattice::ndir, 0.0),
    fx      (g.Volume(), 0.0),
    fy      (g.Volume(), 0.0),
    fz      (g.Volume(), 0.0),
    rho     (g.Volume(), Params::kDensity),
    ux      (g.Volume(), 0.0),
    uy      (g.Volume(), 0.0),
    uz      (g.Volume(), 0.0)
{}
