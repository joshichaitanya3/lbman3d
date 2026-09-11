#ifndef LBM_AN_DEVICE_SOLVER_H_
#define LBM_AN_DEVICE_SOLVER_H_

#include <string>
#include <params.h>
#include "device_fields.h"

#ifdef SIM_WITH_CUDA
template<typename BC>
struct DeviceSolver {
    // int gpu_id;
    // cudaStream_t stream;

    void Initialize(DeviceFields& df);
    // Phase 1 (mirrors QTensorSolver<BC>::StepAndSetupBodyForce on the CPU
    // path): Beris-Edwards Q update + swap + write Σ/τ pointwise + seed the
    // Ericksen distortion force. Reads Q, ux/uy/uz at neighbours;
    // ExchangeQTensor must have run first so ghosts are current.
    void StepAndSetupBodyForce(DeviceFields& df);
    // Phase 2 (mirrors QTensorSolver<BC>::SetActiveStressAndComputeBodyForce):
    // add active-stress divergence + passive-stress divergence + friction to
    // the body force. Reads Q, Σ, τ at neighbours; ExchangePassiveStresses
    // must have run between phase 1 and this call so ghosts of the freshly
    // updated Q + newly written Σ/τ are current.
    void SetActiveStressAndComputeBodyForce(DeviceFields& df);
    // Convenience: phase 1 followed by phase 2 with no halo in between. Used
    // by single-rank builds (LBM_ENABLE_NVSHMEM off) where neighbour reads
    // resolve through boundary_handler at read time and no cross-rank halo is
    // needed.
    void QTensorStep(DeviceFields& df);
    void LBMStep(DeviceFields& df);

};

#else
template<typename BC>
struct DeviceSolver {};   // zero-size, optimized away entirely

#endif
#endif // LBM_AN_DEVICE_SOLVER_H_
