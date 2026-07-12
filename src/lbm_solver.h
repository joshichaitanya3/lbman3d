#ifndef LBM_AN_LBM_SOLVER_H_
#define LBM_AN_LBM_SOLVER_H_

#include "offsets.h"
#include "boundary_handler.h"
#include "fluid_fields.h"

// Pure LBM fluid solver — no knowledge of Q-tensor.
// fx/fy/fz in FluidFields are the only coupling point; set them externally
// (e.g. by QTensorSolver) before calling LatticeBoltzmannStep.
template<typename BC>
class LbmSolver {
public:
    LbmSolver();

    // Set f = f_eq at initial rho/u (call once before the time loop).
    void Initialize(FluidFields& ff) const;

    // Single LBM step: compute moments → collide → stream + apply boundary conditions.
    void LatticeBoltzmannStep(FluidFields& ff) const;
};

#include "lbm_solver.tpp"

#endif // LBM_AN_LBM_SOLVER_H_
