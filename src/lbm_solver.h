#ifndef LBM_AN_LBM_SOLVER_H_
#define LBM_AN_LBM_SOLVER_H_

#include "grid.h"
#include "fluid_fields.h"

// Pure LBM fluid solver — no knowledge of Q-tensor.
// fx/fy in FluidFields are the only coupling point; set them externally
// (e.g. by QTensorSolver) before calling LatticeBoltzmannStep.
template<typename BC>
class LbmSolver {
    Grid<BC> grid_;

    int  UXoff(int x, int s) const { return grid_.UXoff(x, s); }
    int  UYoff(int y, int s) const { return grid_.UYoff(y, s); }
    int  UZoff(int z, int s) const { return grid_.UZoff(z, s); }
    bool InDomain(int x, int y, int z) const;

    double Feq(double rho, double ux, double uy, double uz, int i) const;
    void ResetFeq(FluidFields& ff) const;
    void ComputeForcingTerms(FluidFields& ff) const;
    void ComputeMoments(FluidFields& ff) const;
    void Collide(FluidFields& ff) const;
    void Stream(FluidFields& ff) const;

    // Per-wall bounce-back / specular-reflection handlers.
    // Dispatched at compile time via if constexpr on WallSpec::UBC.
    template<typename WallSpec> void HandleWallZHi(FluidFields& ff) const;
    template<typename WallSpec> void HandleWallZLo(FluidFields& ff) const;
    template<typename WallSpec> void HandleWallYHi(FluidFields& ff) const;
    template<typename WallSpec> void HandleWallYLo(FluidFields& ff) const;
    template<typename WallSpec> void HandleWallXLo(FluidFields& ff) const;
    template<typename WallSpec> void HandleWallXHi(FluidFields& ff) const;

    void HandleBoundaries(FluidFields& ff) const;

public:
    explicit LbmSolver(Grid<BC> grid);

    // Set f = f_eq at initial rho/u (call once before the time loop).
    void Initialize(FluidFields& ff) const;

    // Single LBM step: ResetFeq → ComputeForcingTerms → Collide → Stream
    //                  → HandleBoundaries → ComputeMoments.
    void LatticeBoltzmannStep(FluidFields& ff) const;
};

#include "lbm_solver.tpp"

#endif // LBM_AN_LBM_SOLVER_H_
