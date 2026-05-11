#include "fluid_fields.h"
#include "params.h"

FluidFields::FluidFields() :
    f_data       (Params::nx * Params::ny * Params::nz * Params::ndir, 0.0),
    f_new_data   (Params::nx * Params::ny * Params::nz * Params::ndir, 0.0),
    f_eq_data    (Params::nx * Params::ny * Params::nz * Params::ndir, 0.0),
    forcing_data (Params::nx * Params::ny * Params::nz * Params::ndir, 0.0),
    fx_data      (Params::nx * Params::ny * Params::nz, 0.0),
    fy_data      (Params::nx * Params::ny * Params::nz, 0.0),
    fz_data      (Params::nx * Params::ny * Params::nz, 0.0),
    rho_data     (Params::nx * Params::ny * Params::nz, Params::kDensity),
    ux_data      (Params::nx * Params::ny * Params::nz, 0.0),
    uy_data      (Params::nx * Params::ny * Params::nz, 0.0),
    uz_data      (Params::nx * Params::ny * Params::nz, 0.0),
    rho    (rho_data.data(),     Params::nx, Params::ny, Params::nz),
    ux     (ux_data.data(),      Params::nx, Params::ny, Params::nz),
    uy     (uy_data.data(),      Params::nx, Params::ny, Params::nz),
    uz     (uz_data.data(),      Params::nx, Params::ny, Params::nz),
    fx     (fx_data.data(),      Params::nx, Params::ny, Params::nz),
    fy     (fy_data.data(),      Params::nx, Params::ny, Params::nz),
    fz     (fz_data.data(),      Params::nx, Params::ny, Params::nz),
    f      (f_data.data(),       Params::nx, Params::ny, Params::nz, Params::ndir),
    f_new  (f_new_data.data(),   Params::nx, Params::ny, Params::nz, Params::ndir),
    f_eq   (f_eq_data.data(),    Params::nx, Params::ny, Params::nz, Params::ndir),
    forcing(forcing_data.data(), Params::nx, Params::ny, Params::nz, Params::ndir)
{}
