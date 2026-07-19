#ifndef LBM_AN_TESTS_PARAMS_BC_CHECK_PARAMS_H_
#define LBM_AN_TESTS_PARAMS_BC_CHECK_PARAMS_H_

#include <cmath> // for sqrt

/*
!\brief Custom parameters for the BC-combination integration test
*/
namespace Params {

    // Speed of sound related constants

    static constexpr double kCs2Inv = 3.0; // 1/c_s^2
    static constexpr double kCs2InvTimes2 = 6.0; // 2/c_s^2
    static constexpr double kCs4Inv = 9.0; // 1/c_s^4
    static constexpr double khalfCs2Inv = 1.5; // 1/2 * 1/c_s^2
    static constexpr double khalfCs4Inv = 4.5; // 1/2 * 1/c_s^4
    // Grid
    inline constexpr int nx = 8;
    inline constexpr int ny = 8;
    inline constexpr int nz = 16;
    inline constexpr int nq = 3;
    inline constexpr int numprocs = 1;

    // Spatial / temporal
    inline constexpr double DX = 1.0, DY = 1.0, DZ = 1.0;
    inline constexpr double DT = 1.0;


    // LBM relaxation
    inline constexpr double RHO = 1.1;        // initial lattice density
    inline constexpr double kDensity = 1.0;   // physical density scale
    inline constexpr double TAUF = 1.5 * DT;
    inline constexpr double omega         = 1.0 - DT / TAUF;
    inline constexpr double omega_prime   = DT / TAUF;
    inline constexpr double omega_forcing = 1.0 - DT / 2.0 / TAUF;

    // Free-energy / elasticity
    // Larger than the qtrelax test's L=0.01: the AnchoringConvergesAtWall
    // scenario needs the elastic (wall-coupling) term to dominate within
    // ~500 steps. The relevant relaxation rate scales like L*GAMMA*(pi/nz)^2,
    // so the 16-cell z-extent (the anchored axis) needs a further ~4x over
    // an 8-cell extent to hit the same convergence in the same step budget.
    inline constexpr double L = 0.3;                          // Frank elasticity
    inline constexpr double A = 0;
    inline constexpr double B = -0.3;
    inline constexpr double C = 0.3;

    // Q-tensor dynamics
    inline constexpr double LAMBDA = 0.3;    // flow-aligning
    inline constexpr double GAMMA  = 0.34;    // inverse rotational viscosity

    // Activity & friction
    inline constexpr double ALPHA = 0.0;
    inline constexpr double MU    = 0.0;

    // Initial conditions
    inline constexpr double NOISE = 0.05;

    // Wall BC (used by HandleBoundaries)
    inline constexpr double kLidVelocity = 0.0;

    inline constexpr double kinematicViscosity = kDensity / kCs2Inv * (TAUF - 0.5 * DT);

    // Logging verbosity
    inline constexpr bool kDebugLogging = false;

    // Tracks TotalNematicFreeEnergy (see analysis_fields.h) in SimIO::Log.
    // Off by default: it's an O(N) domain pass with its own neighbor stencil,
    // only meaningful as a solver sanity check — e.g. with ALPHA=0, starting
    // from random Q and flow, the free energy should monotonically decrease.
    inline constexpr bool kTrackNematicEnergy = false;
}

#endif // LBM_AN_TESTS_PARAMS_BC_CHECK_PARAMS_H_
