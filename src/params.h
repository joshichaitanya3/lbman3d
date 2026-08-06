#ifndef LBM_AN_PARAMS_H_
#define LBM_AN_PARAMS_H_

namespace Params {

    // Speed of sound related constants

    static constexpr double kCs2Inv = 3.0; // 1/c_s^2
    static constexpr double kCs2InvTimes2 = 6.0; // 2/c_s^2
    static constexpr double kCs4Inv = 9.0; // 1/c_s^4
    static constexpr double khalfCs2Inv = 1.5; // 1/2 * 1/c_s^2
    static constexpr double khalfCs4Inv = 4.5; // 1/2 * 1/c_s^4
    // Grid
    inline constexpr int nx = 20;    // = H, the plate separation (paper uses 10,15,20,25)
    inline constexpr int ny = 100;
    inline constexpr int nz = 100;
    inline constexpr int kNumOMPThreads = 10;

    // Spatial / temporal
    inline constexpr double DX = 1.0, DY = 1.0, DZ = 1.0;

    // DT MUST BE 1.0. Streaming moves exactly one cell per LatticeBoltzmannStep,
    // so the LBM timestep is 1 lattice unit by construction. DT only scales the
    // Q-tensor right-hand side and the Guo forcing, so any DT != 1 silently
    // decouples the two: at the previous DT = 0.05, Q was advected at u/20
    // (measured), and kinematicViscosity below came out a factor 1/DT too small.
    // To take smaller *effective* Q steps, reduce GAMMA (and the elastic /
    // activity constants) rather than DT — see the stability budget below.
    inline constexpr double DT = 1.0;
    static_assert(DT == 1.0,
        "DT must be 1.0: streaming advances one cell per step, so any other "
        "value desynchronises the Q update and the flow. Scale GAMMA/L/ALPHA "
        "instead.");

    // LBM relaxation
    inline constexpr double kDensity = 1.0;   // physical density scale

    // TAUF/DT is the relaxation time in streaming steps, and it is the ONLY
    // handle on viscosity: nu = c_s^2 (TAUF/DT - 1/2). It was previously pinned
    // at 1.5*DT, which fixes nu = 1/3 for every DT and leaves no handle at all.
    //
    // 2.5*DT gives nu = 2/3, matching the dynamic viscosity eta = 2/3 at rho = 1
    // used by Shendruk et al., PRE 98, 010601(R) (2018) — see the benchmark block
    // below. Their gamma/eta = 2.94/(2/3) = 4.41 then reproduces the "gamma/eta
    // ~ 9/2" quoted in that paper.
    // Do NOT raise TAUF to chase stability. Measured: raising nu to 1.833
    // (TAUF = 6) did not prevent the free-slip divergence — it moved it earlier
    // (step 11000 vs 16000) — and it costs the paper's gamma/eta ~ 9/2
    // (2.94/1.833 = 1.60). The divergence was a boundary-condition problem, not a
    // viscosity one; see SlitNoSlipConfig in sim_config.h.
    inline constexpr double TAUF = 2.5 * DT;
    inline constexpr double omega         = 1.0 - DT / TAUF;
    inline constexpr double omega_prime   = DT / TAUF;
    inline constexpr double omega_forcing = 1.0 - DT / 2.0 / TAUF;

    // ── Benchmark: Shendruk, Thijssen, Yeomans & Doostmohammadi ──────────────
    //   "Twist-induced crossover from two-dimensional to three-dimensional
    //    turbulence in active nematics", Phys. Rev. E 98, 010601(R) (2018).
    //
    // The values below reproduce that paper's setup. Mapping from their symbols:
    //
    //   A, B, C = 0, -0.3, 0.3   ->  same (their Q = 3S(nn - I/3)/2 with
    //                                 S_eq = 1/3 is the SAME tensor as this
    //                                 code's Q = S(nn - I/3) with S_eq = 0.5)
    //   lambda (alignment) = 0.3 ->  LAMBDA
    //   gamma (rot. visc)  = 2.94->  GAMMA = 1/gamma = 0.340
    //   eta (dyn. visc)    = 2/3 ->  nu = eta/rho = 2/3, i.e. TAUF = 2.5*DT
    //   rho                = 1   ->  kDensity
    //   K (Frank, one-const, 0.01..0.05)
    //                            ->  L = 2K/(9 S_eq^2) = 2K  (their S_eq = 1/3)
    //   zeta (activity, extensile, 0.01..0.05)
    //                            ->  ALPHA
    //   domain H x 100 x 100, H in {10,15,20,25}, plates on x
    //                            ->  nx=H, ny=nz=100, SimBC = SlitConfig
    //                                (the paper puts its plates on z; x here
    //                                 only because z is the slowest-varying
    //                                 index — see sim_config.h)
    //
    // Their control parameter is the dimensionless ACTIVITY NUMBER
    //
    //     A_act = H * sqrt(zeta / K) = nx * sqrt(ALPHA / (L/2))
    //
    // with the quasi-2D -> 3D crossover at A_act,cr ~ 17.5, and the paper
    // exploring up to A_act ~ 50. To set a target A_act at fixed nx and K:
    //
    //     ALPHA = (L/2) * (A_act / nx)^2
    //
    //   nx=20, K=0.03 (L=0.06):  A_act = 11.5 -> ALPHA=0.010   (quasi-2D)
    //                            A_act = 17.5 -> ALPHA=0.023   (crossover)
    //                            A_act = 25.8 -> ALPHA=0.050   (3D)
    //   nx=20, K=0.01 (L=0.02):  A_act = 34.6 -> ALPHA=0.030   (deep 3D)
    //
    // NOTE on the previous values: L=0.01, ALPHA=0.04 at H=64 gives
    // A_act ~ 181, about 10x past the crossover and 3-4x beyond anything in the
    // paper — i.e. far deeper into 3D active turbulence than any published
    // comparison. Sub-lattice xi and l_a are NOT the anomaly they look like:
    // at the paper's own crossover point l_a = sqrt(K/zeta) ~ 1.1 dx and
    // xi = sqrt(L/(C S_eq^2)) ~ 0.9 dx.

    // Free-energy / elasticity
    inline constexpr double L = 0.06;        // = 2K with K = 0.03
    inline constexpr double A = 0;
    inline constexpr double B = -0.3;
    inline constexpr double C = 0.3;         // with A=0, B=-0.3: S_eq = 0.5

    // Q-tensor dynamics
    inline constexpr double LAMBDA = 0.3;    // flow-aligning
    inline constexpr double GAMMA  = 0.34;   // = 1/2.94, their rotational viscosity

    // Activity & friction
    inline constexpr double ALPHA = 0.023;   // zeta: A_act = 17.5 at nx = 20
    inline constexpr double MU    = 0.0;

    // ── Explicit-Euler stability budget for the Q update ─────────────────────
    // The Q step is forward Euler, so three limits apply (dx = 1, DT = 1):
    //
    //   1. elastic diffusion   DT * GAMMA * L <= 1/6      (3D 7-point Laplacian)
    //      here 0.0204 vs 0.167                           -> 8.2x margin
    //   2. cell Peclet         |u| <= 2 * GAMMA * L        (centred advection)
    //      MEASURED u_rms = 0.0155 vs 0.0408              -> 2.6x margin
    //   3. CFL                 |u| * DT <= 1              -> trivially satisfied
    //
    // On limit 2: the a-priori estimate |u| ~ sqrt(zeta*K)/eta gives 0.039, which
    // would be marginal. It over-predicts by ~2.5x. The measured steady-state
    // u_rms with no-slip plates is 0.0155 (Ma 0.027, Re 0.46), leaving a
    // comfortable Peclet margin. Symbolically the ratio is scale-free,
    //
    //     |u| / (2*GAMMA*L) = (gamma/eta) * (A_act/H) / 4 = (gamma/eta)/(4*l_a)
    //
    // so it degrades linearly with A_act at fixed geometry: expect trouble around
    // A_act >~ 45 at nx = 20 even with the measured (lower) velocity. Upwinding
    // the advection removes the constraint; increasing nx at fixed A_act (i.e.
    // resolving l_a, currently 1.14 dx) also helps linearly. Neither is needed at
    // A_act = 17.5.
    //
    // Resulting flow regime (order-of-magnitude, verify against a run):
    // |u| ~ 0.039, Ma = |u|/c_s ~ 0.068, Re = |u|*nx/nu ~ 1.2. Note this is
    // deliberately NOT the Stokes limit — the benchmark solves Navier-Stokes at
    // Re ~ 1, so matching it means accepting inertia at that level.
    //
    // Also note NOISE below acts at grid scale, so the initial elastic force
    // scales with L: at L = 0.108 it kicked the run to KE ~ 8 on step 1 and
    // diverged by step ~120. At L = 0.06 that kick is ~2x smaller than the
    // divergent case but still ~6x the original L = 0.01. Watch step 1.

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

#endif // LBM_AN_PARAMS_H_
