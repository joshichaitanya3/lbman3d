#ifndef LBM_AN_QTENSOR_SOLVER_H_
#define LBM_AN_QTENSOR_SOLVER_H_

#include "offsets.h"
#include "boundary_handler.h"
#include "fluid_fields.h"
#include "qtensor_fields.h"

// Q-tensor dynamics solver (Beris-Edwards + active stress).
//
// Subclass and override SetActiveStressAndComputeBodyForce to change the activity model:
//
//   class VaryingAlpha : public QTensorSolver<PeriodicBC> {
//   public:
//       using QTensorSolver::QTensorSolver;   // inherit ctor
//       void SetActiveStressAndComputeBodyForce(FluidFields&, const QTensorFields&) const override;
//   };
//
//   ActiveNematicSim<PeriodicBC> sim{std::make_unique<VaryingAlpha>()};
//
// For pure relaxation (zero activity), override SetActiveStressAndComputeBodyForce with an empty body.
template<typename BC>
class QTensorSolver {
    void UpdateQnewWithQ(QTensorFields& qf) const;

    // Apply the Q-tensor anchoring BC for WallSpec at a single node (x,y,z).
    // Called after the FD timestep for every boundary node.
    //
    // Parameters:
    //   qf       — Q-tensor fields; writes qxx_new/qxy_new/qxz_new/qyy_new/qyz_new at (z,y,x)
    //   x, y, z  — coordinates of the boundary node
    //
    // Periodic / Neumann: compile-time no-op (stencil clamping/wrapping already enforces
    //   ∂Q/∂n = 0; no further action needed).
    // Anchoring<S,θ,φ>: overwrites q_new at (z,y,x) with the strong-anchoring target
    //   computed from S and the director angles (θ,φ).
    template<typename WallSpec> void HandleQBoundaryPoint(QTensorFields& qf, int x, int y, int z) const;

public:
    QTensorSolver() = default;
    virtual ~QTensorSolver() = default;

    // Set initial Q field with uniform noise (qxx ≈ 0.5, qxy ≈ 0).
    void Initialize(QTensorFields& qf) const;

    /* !\brief Beris-Edwards FD step + setting up the backflow coupling
     *
     * Beris-Edwards FD step — updates Q using the current velocity in `ff`.
     * In addition, it uses the already-computed gradients of Q to compute the 
     * passive component of the nematic stress tensor and add the
     *  advective backflow (not coming from the stress tensor) to the 
     * body force `ff.fx/fy/fz`. The divergence of the stress will be added 
     * to the body force separately.
     */
    void StepAndSetupBodyForce(QTensorFields& qf, FluidFields& ff) const;

    // Compute active body force from Q gradients → writes ff.fx, ff.fy.
    // Then add passive stresses and friction.
    // Override this to implement spatiotemporally varying activity or zero activity.
    virtual void SetActiveStressAndComputeBodyForce(FluidFields& ff, const QTensorFields& qf) const;

    // Convenience: StepAndSetupBodyForce then SetActiveStressAndComputeBodyForce.
    void Step(QTensorFields& qf, FluidFields& ff) const;
};

#include "qtensor_solver.tpp"

#endif // LBM_AN_QTENSOR_SOLVER_H_
