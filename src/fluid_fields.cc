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
    rho    (rho_data.data(),     Params::nz, Params::ny, Params::nx),
    ux     (ux_data.data(),      Params::nz, Params::ny, Params::nx),
    uy     (uy_data.data(),      Params::nz, Params::ny, Params::nx),
    uz     (uz_data.data(),      Params::nz, Params::ny, Params::nx),
    fx     (fx_data.data(),      Params::nz, Params::ny, Params::nx),
    fy     (fy_data.data(),      Params::nz, Params::ny, Params::nx),
    fz     (fz_data.data(),      Params::nz, Params::ny, Params::nx),
    f      (f_data.data(),       Params::nz, Params::ny, Params::nx, Params::ndir),
    f_new  (f_new_data.data(),   Params::nz, Params::ny, Params::nx, Params::ndir),
    f_eq   (f_eq_data.data(),    Params::nz, Params::ny, Params::nx, Params::ndir),
    forcing(forcing_data.data(), Params::nz, Params::ny, Params::nx, Params::ndir)
{}
