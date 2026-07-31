#ifndef LBM_AN_TESTS_PARAMS_COUPLED_PARAMS_H_
#define LBM_AN_TESTS_PARAMS_COUPLED_PARAMS_H_

/*
!\brief Custom parameters for the coupled Q<->flow backflow integration test.

Deliberately uses the production timestep DT = 0.05 rather than the DT = 1.0
that every other integration test runs at, so this scenario exercises the
DT-scaled body force and Q right-hand side that production actually uses.

ALPHA = 0 and MU = 0: with no activity injecting energy and no friction
draining it through an external channel, the passive system's total free
energy (nematic + kinetic) must be non-increasing. The backflow term is the
mechanism that exchanges energy between those two reservoirs, so a sign error
there breaks the exchange and shows up as energy injection.
*/
namespace Params {

    // Speed of sound related constants

    static constexpr double kCs2Inv = 3.0; // 1/c_s^2
    static constexpr double kCs2InvTimes2 = 6.0; // 2/c_s^2
    static constexpr double kCs4Inv = 9.0; // 1/c_s^4
    static constexpr double khalfCs2Inv = 1.5; // 1/2 * 1/c_s^2
    static constexpr double khalfCs4Inv = 4.5; // 1/2 * 1/c_s^4
    // Grid
    inline constexpr int nx = 16;
    inline constexpr int ny = 16;
    inline constexpr int nz = 16;
    inline constexpr int nq = 3;
    inline constexpr int kNumOMPThreads = 1;

    // Spatial / temporal
    inline constexpr double DX = 1.0, DY = 1.0, DZ = 1.0;
    inline constexpr double DT = 0.05;   // production value, unlike other integration tests


    inline constexpr double RHO = 1.1;        // initial lattice density
    inline constexpr double kDensity = 1.0;   // physical density scale
    inline constexpr double TAUF = 1.5 * DT;
    inline constexpr double omega         = 1.0 - DT / TAUF;
    inline constexpr double omega_prime   = DT / TAUF;
    inline constexpr double omega_forcing = 1.0 - DT / 2.0 / TAUF;

    // Free-energy / elasticity
    inline constexpr double L = 0.01;                          // Frank elasticity
    inline constexpr double A = 0;
    inline constexpr double B = -0.3;
    inline constexpr double C = 0.3;

    // Q-tensor dynamics
    inline constexpr double LAMBDA = 0.3;    // flow-aligning
    inline constexpr double GAMMA  = 0.34;   // inverse rotational viscosity

    // Activity & friction: both off, so the passive system is a closed
    // energy budget between nematic free energy and kinetic energy.
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
    inline constexpr bool kTrackNematicEnergy = true;
}

#endif // LBM_AN_TESTS_PARAMS_COUPLED_PARAMS_H_
