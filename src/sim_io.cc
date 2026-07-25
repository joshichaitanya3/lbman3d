#include "sim_io.h"
#include <params.h>
#include <sim_config.h>
#include "format_compat.h"

#include <cmath>
#include <ranges>
#include <stdexcept>
#include "vtkhdf_writer.h"
#include "analysis/defect_fields.h"
#include "physics_helpers.h"
#include "lattice_stencil.h"

#include "analysis/disclination.h"
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
    compat::println(log_file_, "  rho0 (numerical)    = {}", RHO);
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
}

bool SimIO::Log(const FluidFields& ff, AnalysisFields& af, const DefectFields& df, int time_step, double nematic_energy) {
    double mass = 0.0, px = 0.0, py = 0.0, pz=0, ke=0.0, e1 = 0.0, e2 = 0.0;

    #pragma omp parallel for schedule(static) default(shared) \
        reduction(+:mass,px,py,pz,ke,e1,e2) num_threads(kNumOMPThreads)
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                mass += ff.rho[idx(x, y, z)];
                px   += ff.rho[idx(x, y, z)] * ff.ux[idx(x, y, z)];
                py   += ff.rho[idx(x, y, z)] * ff.uy[idx(x, y, z)];
                pz   += ff.rho[idx(x, y, z)] * ff.uz[idx(x, y, z)];
                ke   += 0.5 * ff.rho[idx(x, y, z)] * (ff.ux[idx(x, y, z)]*ff.ux[idx(x, y, z)] + 
                                                 ff.uy[idx(x, y, z)]*ff.uy[idx(x, y, z)] + 
                                                 ff.uz[idx(x, y, z)]*ff.uz[idx(x, y, z)]);

                e1   += (ff.ux[idx(x, y, z)]-af.ux_past_[idx(x, y, z)])*(ff.ux[idx(x, y, z)]-af.ux_past_[idx(x, y, z)])
                        + (ff.uy[idx(x, y, z)]-af.uy_past_[idx(x, y, z)])*(ff.uy[idx(x, y, z)]-af.uy_past_[idx(x, y, z)])
                        + (ff.uz[idx(x, y, z)]-af.uz_past_[idx(x, y, z)])*(ff.uz[idx(x, y, z)]-af.uz_past_[idx(x, y, z)]);

                e2   += ff.ux[idx(x, y, z)]*ff.ux[idx(x, y, z)] + ff.uy[idx(x, y, z)]*ff.uy[idx(x, y, z)]  + ff.uz[idx(x, y, z)]*ff.uz[idx(x, y, z)];
                af.ux_past_[idx(x, y, z)] = ff.ux[idx(x, y, z)];
                af.uy_past_[idx(x, y, z)] = ff.uy[idx(x, y, z)];
                af.uz_past_[idx(x, y, z)] = ff.uz[idx(x, y, z)];
            }
        }
    }
    int num_disclinations = df.disclinations.size();
    double te = ke + nematic_energy;

    double global_mass, global_px, global_py, global_pz, global_ke, global_te, global_e1, global_e2;
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
    MPIContext::SumInts(&num_disclinations, &global_num_disclinations); // This is currently incorrect, since a single disclination could span multiple ranks, but we will keep it for now.

    // Divergence is derived from the globally-reduced quantities, so every
    // rank reaches the same verdict and stays in lockstep — only the actual
    // log writing below is root-only.
    bool diverged = std::isnan(global_mass) || std::isnan(global_px) ||
                     std::isnan(global_py) || std::isnan(global_pz);

    if (MPIContext::IsRoot()) {
        compat::println(
            log_file_,
            "Time {}: Mass: {}, Px: {}, Py: {}, Pz: {}, "
            "Kinetic Energy: {}, Total Energy: {}, Relative Error: {}, "
            "NumDisclinations: {}",
            time_step,
            global_mass,
            global_px,
            global_py,
            global_pz,
            global_ke,
            global_te,
            global_e1/global_e2,
            global_num_disclinations
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

void SimIO::ExportCSV(const FluidFields& ff, const QTensorFields& qf,
                   const std::string& path, int step) {
    std::ofstream rho_file, ux_file, uy_file, uz_file;
    std::ofstream qxx_file, qxy_file, qxz_file, qyy_file, qyz_file;

    rho_file.open(compat::format("{}/rho_{}.csv",    path, step), std::ios::out);
    ux_file .open(compat::format("{}/ux_{}.csv",     path, step), std::ios::out);
    uy_file .open(compat::format("{}/uy_{}.csv",     path, step), std::ios::out);
    uz_file .open(compat::format("{}/uz_{}.csv",     path, step), std::ios::out);
    
    qxx_file.open(compat::format("{}/qxx_{}.csv",    path, step), std::ios::out);
    qxy_file.open(compat::format("{}/qxy_{}.csv",    path, step), std::ios::out);
    qxz_file.open(compat::format("{}/qxz_{}.csv",    path, step), std::ios::out);
    qyy_file.open(compat::format("{}/qyy_{}.csv",    path, step), std::ios::out);
    qyz_file.open(compat::format("{}/qyz_{}.csv",    path, step), std::ios::out);

    if (!rho_file.is_open())
        throw std::runtime_error("Failed to open data file");

    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx-1; ++x) {
                compat::print(rho_file,  "{},", ff.rho[idx(x, y, z)]);
                compat::print(ux_file,   "{},", ff.ux[idx(x, y, z)]);
                compat::print(uy_file,   "{},", ff.uy[idx(x, y, z)]);
                compat::print(uz_file,   "{},", ff.uz[idx(x, y, z)]);
                compat::print(qxx_file,  "{},", qf.qxx[idx(x, y, z)]);
                compat::print(qxy_file,  "{},", qf.qxy[idx(x, y, z)]);
                compat::print(qxz_file,  "{},", qf.qxz[idx(x, y, z)]);
                compat::print(qyy_file,  "{},", qf.qyy[idx(x, y, z)]);
                compat::print(qyz_file,  "{},", qf.qyz[idx(x, y, z)]);                
            }
            compat::print(rho_file,  "{}\n", ff.rho[idx(nx-1, y, z)]);
            compat::print(ux_file,   "{}\n", ff.ux[idx(nx-1, y, z)]);
            compat::print(uy_file,   "{}\n", ff.uy[idx(nx-1, y, z)]);
            compat::print(uz_file,   "{}\n", ff.uz[idx(nx-1, y, z)]);
            compat::print(qxx_file,  "{}\n", qf.qxx[idx(nx-1, y, z)]);
            compat::print(qxy_file,  "{}\n", qf.qxy[idx(nx-1, y, z)]);
            compat::print(qxz_file,  "{}\n", qf.qxz[idx(nx-1, y, z)]);
            compat::print(qyy_file,  "{}\n", qf.qyy[idx(nx-1, y, z)]);
            compat::print(qyz_file,  "{}\n", qf.qyz[idx(nx-1, y, z)]);
        }
    }

    if constexpr (Params::kDebugLogging) {
        ExportDistributionCSV(ff, path, step);
    }
}

void SimIO::ExportDistributionCSV(const FluidFields& ff,
                                const std::string& path, int step) {
    std::ofstream f_file;
    for (int i : std::views::iota(0, Lattice::ndir)) {
        f_file.open(compat::format("{}/f_{}_{}.csv", path, i, step), std::ios::out);
        if (!f_file.is_open())
            throw std::runtime_error("Failed to open data file");

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx-1; ++x) {
                    compat::print(f_file, "{},", ff.f[idx(x, y, z, i)]);
                }
                compat::print(f_file, "{}\n", ff.f[idx(nx-1, y, z, i)]);
            }
        }
        f_file.close();
    }
}


void SimIO::ExportVTKHDF(const FluidFields& ff, AnalysisFields& af,
                         const std::string& path, int step) {
    constexpr int kStepWidth = [] {
        int w = 1, n = kNumSteps - 1;
        while (n >= 10) { n /= 10; ++w; }
        return w;
    }();
    const std::string file_path = std::format("{}/lbm_{:0{}}.vtkhdf", path, step, kStepWidth);

    ImageDataWriter writer(file_path);

    // --- PointData datasets, shape [nz, ny, nx] (z slowest, x fastest) ---

    writer.WriteScalarField("rho", ff.rho.data());

    if constexpr (Params::kDebugLogging) {
        // ff.f is laid out with i fastest-varying (host idx() layout), so we still
        // need a scratch buffer per direction with the export's [z,y,x] layout.
        std::vector<double> buf(nx * ny * nz);
        for (int i = 0; i < Lattice::ndir; i++) {
            for (int z = 0; z < nz; ++z)
                for (int y = 0; y < ny; ++y)
                    for (int x = 0; x < nx; ++x)
                        buf[z * ny * nx + y * nx + x] = ff.f[idx(x, y, z, i)];
            writer.WriteScalarField(std::format("f{}", i).c_str(), buf.data());
        }
    }

    writer.WriteScalarField("order", af.order_.data());

    // Velocity: three independent scalar fields, written directly from backing stores.
    writer.WriteScalarField("ux", ff.ux.data());
    writer.WriteScalarField("uy", ff.uy.data());
    writer.WriteScalarField("uz", ff.uz.data());

    // Director: AoS layout [nz, ny, nx, 3] (see dirIdx in analysis_fields.h) passes
    // straight through to WriteVectorField without repacking.
    writer.WriteVectorField("director", af.director_.data());

}

void SimIO::ExportDisclinations(
    const DefectFields& df,
    const std::string& path,
    int step) 
{
    
    DisclinationMesh mesh;

    for (Disclination d : df.disclinations) {
        mesh.AddDisclination(d);
    }

    constexpr int kStepWidth = [] {
        int w = 1, n = kNumSteps - 1;
        while (n >= 10) { n /= 10; ++w; }
        return w;
    }();
    const std::string file_path = std::format("{}/disclinations_{:0{}}.vtkhdf", path, step, kStepWidth);

    UnstructuredGridWriter writer(file_path);

    writer.WriteTopology(mesh.Points(), mesh.Connectivity(),
                         mesh.Offsets(), mesh.CellTypes());

}
