#ifndef LBM_AN_QTENSOR_SOLVER_H_
#define LBM_AN_QTENSOR_SOLVER_H_

#include "grid.h"
#include "fluid_fields.h"
#include "qtensor_fields.h"

// Q-tensor dynamics solver (Beris-Edwards + active stress).
//
// Subclass and override ComputeActiveBodyForce to change the activity model:
//
//   class VaryingAlpha : public QTensorSolver<PeriodicBC> {
//   public:
//       using QTensorSolver::QTensorSolver;   // inherit ctor
//       void ComputeActiveBodyForce(FluidFields&, const QTensorFields&) const override;
//   };
//
//   ActiveNematicSim<PeriodicBC> sim{grid, std::make_unique<VaryingAlpha>(grid)};
//
// For pure relaxation (zero activity), override ComputeActiveBodyForce with an empty body.
template<typename BC>
class QTensorSolver {
    Grid<BC> grid_;

    void HandleQBoundary(QTensorFields& qf) const;
    void UpdateQnewWithQ(QTensorFields& qf) const;

    // Per-wall Q-tensor boundary handlers.
    // For Neumann/Periodic: compile-time no-op (stencil clamping/wrapping suffices).
    // For Anchoring: overwrites qxx_new/qxy_new at the wall with the prescribed value.
    template<typename WallSpec> void HandleQWallZLo(QTensorFields& qf) const;
    template<typename WallSpec> void HandleQWallZHi(QTensorFields& qf) const;
    template<typename WallSpec> void HandleQWallYLo(QTensorFields& qf) const;
    template<typename WallSpec> void HandleQWallYHi(QTensorFields& qf) const;
    template<typename WallSpec> void HandleQWallXLo(QTensorFields& qf) const;
    template<typename WallSpec> void HandleQWallXHi(QTensorFields& qf) const;

protected:
    // Exposed to subclasses for BC-aware stencil access in overrides.
    int QXoff(int x, int s) const { return grid_.QXoff(x, s); }
    int QYoff(int y, int s) const { return grid_.QYoff(y, s); }
    int QZoff(int z, int s) const { return grid_.QZoff(z, s); }

public:
    explicit QTensorSolver(Grid<BC> grid);
    virtual ~QTensorSolver() = default;

    // Set initial Q field with uniform noise (qxx ≈ 0.5, qxy ≈ 0).
    void Initialize(QTensorFields& qf) const;

    // Beris-Edwards FD step — updates Q using the current velocity in ff.
    // Does NOT touch ff.fx/fy.
    void FiniteDifferenceStep(QTensorFields& qf, const FluidFields& ff) const;

    // Compute active body force from Q gradients → writes ff.fx, ff.fy.
    // Override to implement spatiotemporally varying activity or zero activity.
    virtual void ComputeActiveBodyForce(FluidFields& ff, const QTensorFields& qf) const;

    // Convenience: FiniteDifferenceStep then ComputeActiveBodyForce.
    void Step(QTensorFields& qf, FluidFields& ff) const;
};

#include "qtensor_solver.tpp"

#endif // LBM_AN_QTENSOR_SOLVER_H_
