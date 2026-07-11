#include "fluid_fields.h"
#include "params.h"
#include "lattice_stencil.h"

FluidFields::FluidFields() :
    f       (Params::nx * Params::ny * Params::nz * Lattice::ndir, 0.0),
    f_new   (Params::nx * Params::ny * Params::nz * Lattice::ndir, 0.0),
    fx      (Params::nx * Params::ny * Params::nz, 0.0),
    fy      (Params::nx * Params::ny * Params::nz, 0.0),
    fz      (Params::nx * Params::ny * Params::nz, 0.0),
    rho     (Params::nx * Params::ny * Params::nz, Params::kDensity),
    ux      (Params::nx * Params::ny * Params::nz, 0.0),
    uy      (Params::nx * Params::ny * Params::nz, 0.0),
    uz      (Params::nx * Params::ny * Params::nz, 0.0)
{}
