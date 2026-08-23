#include "sim_io.h"
#include <params.h>
#include <sim_config.h>
#include "format_compat.h"

#include <cmath>
#include <ranges>
#include <stdexcept>
#include "analysis/defect_fields.h"
#include "physics_helpers.h"
#include "local_grid.h"
#include "mpi/mpi_context.h"

using namespace Params;

SimIO::SimIO()
{
    // Only the root rank owns the log file — every rank would otherwise open
    // and truncate the same path, corrupting each other's output.
    if (MPIContext::IsRoot())
        log_file_.open("lbm.log", std::ios::out);
}

SimIO::~SimIO() {
    if (log_file_.is_open())
        compat::println(log_file_, "LBM program exited.");
}

void SimIO::LogSetupSummary(std::string_view bc_name, std::string_view backend_info) {
    if (!MPIContext::IsRoot()) return;
    compat::println(log_file_, "Hybrid Lattice Boltzmann simulation for 3D active nematics\n");
    compat::println(log_file_, "##########################################################");
    compat::println(log_file_, "#####################   Parameters   #####################");
    compat::println(log_file_, "##########################################################");
    compat::println(log_file_, "");
    compat::println(log_file_, "--- Compute backend ---");
    compat::println(log_file_, "  {}", backend_info);
    compat::println(log_file_, "");
    compat::println(log_file_, "--- Grid ---");
    compat::println(log_file_, "  nx = {}, ny = {}, nz = {}", nx, ny, nz);
    compat::println(log_file_, "  DX = {}, DY = {}, DZ = {}", DX, DY, DZ);
    compat::println(log_file_, "  BC = {}", bc_name);
    compat::println(log_file_, "");
    compat::println(log_file_, "--- Time ---");
    compat::println(log_file_, "  DT (numerical)      = {}", DT);
    compat::println(log_file_, "");
    compat::println(log_file_, "--- LBM ---");
    compat::println(log_file_, "  TAUF                = {}", TAUF);
    compat::println(log_file_, "  Kinematic Viscosity = {}", kinematicViscosity);
    compat::println(log_file_, "  omega               = {}", omega);
    compat::println(log_file_, "  omega_prime         = {}", omega_prime);
    compat::println(log_file_, "  omega_forcing       = {}", omega_forcing);
    compat::println(log_file_, "");
    compat::println(log_file_, "--- Free energy ---");
    compat::println(log_file_, "  L (Frank elasticity) = {}", L);
    compat::println(log_file_, "  A                    = {}", A);
    compat::println(log_file_, "  B                    = {}", B);
    compat::println(log_file_, "  C                    = {}", C);
    compat::println(log_file_, "");
    compat::println(log_file_, "--- Q-tensor dynamics ---");
    compat::println(log_file_, "  GAMMA (rot. viscosity^-1) = {}", GAMMA);
    compat::println(log_file_, "  LAMBDA (flow-aligning)    = {}", LAMBDA);
    compat::println(log_file_, "  NOISE                     = {}", NOISE);
    compat::println(log_file_, "");
    compat::println(log_file_, "--- Activity & friction ---");
    compat::println(log_file_, "  ALPHA (numerical)  = {}", ALPHA);
    compat::println(log_file_, "  MU (linear friction) = {}", MU);
    compat::println(log_file_, "");
    compat::println(log_file_, "##########################################################");
    #ifdef LBM_ENABLE_MPI
    compat::println(log_file_, "NOTE: Defect detection and export are disabled in MPI builds (not yet implemented).");
    #endif
}

bool SimIO::Log(const FluidFields& ff, AnalysisFields& af, const DefectFields& df, int time_step, double nematic_energy) {
    double mass = 0.0, px = 0.0, py = 0.0, pz=0, ke=0.0, e1 = 0.0, e2 = 0.0;
    double umax2 = 0.0;
    const LocalGrid& g = ff.grid;

    #pragma omp parallel for schedule(static) default(shared) \
        reduction(+:mass,px,py,pz,ke,e1,e2) reduction(max:umax2) num_threads(kNumOMPThreads)
    for (int z = 0; z < g.local_nz; ++z) {
        for (int y = 0; y < g.local_ny; ++y) {
            for (int x = 0; x < g.local_nx; ++x) {
                const int idxp = g.halo_idx(x, y, z);
                mass += ff.rho[idxp];
                px   += ff.rho[idxp] * ff.ux[idxp];
                py   += ff.rho[idxp] * ff.uy[idxp];
                pz   += ff.rho[idxp] * ff.uz[idxp];
                const double u2 = ff.ux[idxp]*ff.ux[idxp]
                                  + ff.uy[idxp]*ff.uy[idxp]
                                  + ff.uz[idxp]*ff.uz[idxp];
                ke   += 0.5 * ff.rho[idxp] * u2;
                if (u2 > umax2) umax2 = u2;

                e1   += (ff.ux[idxp]-af.ux_past_[idxp])*(ff.ux[idxp]-af.ux_past_[idxp])
                        + (ff.uy[idxp]-af.uy_past_[idxp])*(ff.uy[idxp]-af.uy_past_[idxp])
                        + (ff.uz[idxp]-af.uz_past_[idxp])*(ff.uz[idxp]-af.uz_past_[idxp]);

                e2   += u2;
                af.ux_past_[idxp] = ff.ux[idxp];
                af.uy_past_[idxp] = ff.uy[idxp];
                af.uz_past_[idxp] = ff.uz[idxp];
            }
        }
    }
    int num_disclinations = df.disclinations.size();
    double te = ke + nematic_energy;

    double global_mass, global_px, global_py, global_pz, global_ke, global_te, global_e1, global_e2;
    double global_umax2;
    int global_num_disclinations;

    // Collective — every rank must call these, even though only the root
    // rank below actually writes the result to the log file.
    MPIContext::SumDoubles(&mass, &global_mass);
    MPIContext::SumDoubles(&px, &global_px);
    MPIContext::SumDoubles(&py, &global_py);
    MPIContext::SumDoubles(&pz, &global_pz);
    MPIContext::SumDoubles(&ke, &global_ke);
    MPIContext::SumDoubles(&te, &global_te);
    MPIContext::SumDoubles(&e1, &global_e1);
    MPIContext::SumDoubles(&e2, &global_e2);
    MPIContext::MaxDoubles(&umax2, &global_umax2);
    MPIContext::SumInts(&num_disclinations, &global_num_disclinations); // This is currently incorrect, since a single disclination could span multiple ranks, but we will keep it for now.

    // Divergence is derived from the globally-reduced quantities, so every
    // rank reaches the same verdict and stays in lockstep — only the actual
    // log writing below is root-only.
    bool diverged = std::isnan(global_mass) || std::isnan(global_px) ||
                     std::isnan(global_py) || std::isnan(global_pz);

    if (MPIContext::IsRoot()) {
        compat::println(
            log_file_,
            // Nematic Energy is logged separately as well as inside Total
            // Energy: with ALPHA=0 it should decrease monotonically on its own,
            // which is unreadable from the ke+nematic sum. Appended last so
            // monitor.py's (unanchored) parser keeps working unchanged.
            "Time {}: Mass: {}, Px: {}, Py: {}, Pz: {}, "
            "Kinetic Energy: {}, Total Energy: {}, Relative Error: {}, "
            "NumDisclinations: {}, Nematic Energy: {}, Max |u|: {}",
            time_step,
            global_mass,
            global_px,
            global_py,
            global_pz,
            global_ke,
            global_te,
            global_e1/global_e2,
            global_num_disclinations,
            global_te - global_ke,
            std::sqrt(global_umax2)
        );
        std::flush(log_file_);
    }
    if (diverged) {
        if (MPIContext::IsRoot())
            compat::println(log_file_, "DIVERGED at time step {} — aborting.", time_step);
        return false;
    }
    return true;
}

// ExportVTKHDF and ExportDisclinations are templated on BC (they stamp the
// compile-time SimBC onto the VTKHDF /VTKHDF group as attributes so
// find_defects can validate on read). Definitions live in sim_io.tpp.
