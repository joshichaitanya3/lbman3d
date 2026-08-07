#include "fluid_fields.h"
#include <params.h>
#include "lattice_stencil.h"
#include "local_grid.h"

FluidFields::FluidFields(LocalGrid g) :
    grid    (g),
    f       (g.HaloVolume() * Lattice::ndir, 0.0),
    f_new   (g.HaloVolume() * Lattice::ndir, 0.0),
    fx      (g.HaloVolume(), 0.0),
    fy      (g.HaloVolume(), 0.0),
    fz      (g.HaloVolume(), 0.0),
    rho     (g.HaloVolume(), Params::kDensity),
    ux      (g.HaloVolume(), 0.0),
    uy      (g.HaloVolume(), 0.0),
    uz      (g.HaloVolume(), 0.0)
{}
