#ifndef LBM_AN_LBM_SOLVER_H_
#define LBM_AN_LBM_SOLVER_H_

#include "grid.h"
#include "fluid_fields.h"

// Pure LBM fluid solver — no knowledge of Q-tensor.
// fx/fy/fz in FluidFields are the only coupling point; set them externally
// (e.g. by QTensorSolver) before calling LatticeBoltzmannStep.
template<typename BC>
class LbmSolver {
    Grid<BC> grid_;

    int  UXoff(int x, int s) const { return grid_.UXoff(x, s); }
    int  UYoff(int y, int s) const { return grid_.UYoff(y, s); }
    int  UZoff(int z, int s) const { return grid_.UZoff(z, s); }
    bool InDomain(int x, int y, int z) const;

    double Feq(double rho, double ux, double uy, double uz, int i) const;

    // Apply the boundary condition for WallSpec to a single out-of-domain stream.
    // Called when direction i at node (x,y,z) streams outside the domain.
    //
    // Parameters:
    //   x, y, z  — coordinates of the source fluid node
    //   i        — lattice direction index that streamed out of domain
    //   i_refl   — the specular-reflection partner of i for the relevant wall axis
    //              (specX[i], specY[i], or specZ[i]); used only for SpecularReflection
    //   f_star   — post-collision value of f at (x,y,z) for direction i
    //   ff       — fluid fields (reads rho for moving-wall correction; writes f_new)
    //
    // Writes to f_new at the SOURCE node (x,y,z), not the out-of-domain destination:
    //   SpecularReflection : f_new[i_refl] = f_star
    //   NoSlip / MovingWall: f_new[opp[i]] = f_star + 6·ρ·w[opp[i]]·(e[opp[i]]·u_wall)
    template<typename WallSpec> void HandleBoundaryPoint(
        int x,
        int y,
        int z,
        int i,
        int i_refl,
        double f_star,
        FluidFields& ff
    ) const;

public:
    explicit LbmSolver(Grid<BC> grid);

    // Set f = f_eq at initial rho/u (call once before the time loop).
    void Initialize(FluidFields& ff) const;

    // Single LBM step: compute moments → collide → stream + apply boundary conditions.
    void LatticeBoltzmannStep(FluidFields& ff) const;
};

#include "lbm_solver.tpp"

#endif // LBM_AN_LBM_SOLVER_H_
