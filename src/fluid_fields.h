#ifndef LBM_AN_FLUID_FIELDS_H_
#define LBM_AN_FLUID_FIELDS_H_

#include <vector>
#include <params.h>

// Owns all LBM state: distribution functions, macroscopic fields, body force.
// fx/fy/fz are the only write surface shared with QTensorSolver.
// Flat, row-major storage indexed via idx(x,y,z[,i]) from physics_helpers.h.
struct FluidFields {
    std::vector<double> f, f_new;
    std::vector<double> fx, fy, fz;
    std::vector<double> rho, ux, uy, uz;

    FluidFields();

    void SwapFandFnew() {
        std::swap(f, f_new);
    }
};

#endif // LBM_AN_FLUID_FIELDS_H_
