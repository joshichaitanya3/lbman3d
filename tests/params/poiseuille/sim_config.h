#ifndef LBM_AN_TESTS_PARAMS_POISEUILLE_SIM_CONFIG_H_
#define LBM_AN_TESTS_PARAMS_POISEUILLE_SIM_CONFIG_H_

// Shadow of src/sim_config.h for the Poiseuille NVSHMEM integration test.
// Sets SimBC = PoiseuilleConfig so device_solver.cu instantiates
// DeviceSolver<PoiseuilleConfig> for this test target.

#include "boundary.h"
#include "qtensor_types.h"

using SimBC = PoiseuilleConfig;

inline constexpr Advection kQAdvection  = Advection::Centred;
inline constexpr int kNumSteps          = 5000;
inline constexpr int kSaveInterval      = 100;
inline constexpr int kLogInterval       = kSaveInterval;

#endif // LBM_AN_TESTS_PARAMS_POISEUILLE_SIM_CONFIG_H_
