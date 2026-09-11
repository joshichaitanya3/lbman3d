#ifndef LBM_AN_TESTS_PARAMS_QTRELAX_SIM_CONFIG_H_
#define LBM_AN_TESTS_PARAMS_QTRELAX_SIM_CONFIG_H_

// Shadowed sim_config.h for the qtrelax params dir. Same shadowing mechanism
// as params.h: `#include <sim_config.h>` in src/ resolves through -I, so
// listing this directory first pulls this file instead of src/sim_config.h.
//
// Purpose: pin SimBC to FullyPeriodicConfig for tests that link
// device_solver.cu — the CU's explicit `template struct DeviceSolver<SimBC>;`
// determines which template instantiation is available. The MPI/NVSHMEM
// qtensor_relaxation test uses FullyPeriodicConfig (no walls interfering
// with the rank split), so SimBC must match. The single-rank
// tests/integration/qtrelax_test does not link device_solver.cu
// (LBM_FORCE_CPU makes DeviceSolver a zero-size stub), so it is not affected
// by whichever config lives here.

#include "boundary.h"
#include "qtensor_types.h"

using SimBC = FullyPeriodicConfig;

inline constexpr Advection kQAdvection = Advection::Centred;

inline constexpr int kNumSteps     = 10000;
inline constexpr int kSaveInterval = 100;
inline constexpr int kLogInterval  = kSaveInterval;

#endif // LBM_AN_TESTS_PARAMS_QTRELAX_SIM_CONFIG_H_
