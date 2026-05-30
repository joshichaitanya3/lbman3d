#include "sim_io.h"
#include "params.h"
#include "sim_config.h"
#include "format_compat.h"

#include <cmath>
#include <ranges>
#include <stdexcept>
#include "vtkhdf_writer.h"

using namespace Params;

SimIO::SimIO()
{
    log_file_.open("lbm.log", std::ios::out);
}

SimIO::~SimIO() {
    if (log_file_.is_open())
        compat::println(log_file_, "LBM program exited.");
}

void SimIO::LogSetupSummary(std::string_view bc_name) {
    compat::println(log_file_, "Hybrid Lattice Boltzmann simulation for 3D active nematics\n");
    compat::println(log_file_, "##########################################################");
    compat::println(log_file_, "#####################   Parameters   #####################");
    compat::println(log_file_, "##########################################################");
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

bool SimIO::Log(const FluidFields& ff, AnalysisFields& af, int time_step) {
    double mass = 0.0, px = 0.0, py = 0.0, pz=0, ke=0.0, e1 = 0.0, e2 = 0.0;

    #pragma omp parallel for schedule(static) default(shared) \
        reduction(+:mass,px,py,pz,e1,e2) num_threads(numprocs)
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                mass += ff.rho[z, y, x];
                px   += ff.rho[z, y, x] * ff.ux[z, y, x];
                py   += ff.rho[z, y, x] * ff.uy[z, y, x];
                pz   += ff.rho[z, y, x] * ff.uz[z, y, x];
                ke   += 0.5 * ff.rho[z, y, x] * (ff.ux[z, y, x]*ff.ux[z, y, x] + 
                                                 ff.uy[z, y, x]*ff.uy[z, y, x] + 
                                                 ff.uz[z, y, x]*ff.uz[z, y, x]);

                e1   += (ff.ux[z,y,x]-af.ux_past_[z,y,x])*(ff.ux[z,y,x]-af.ux_past_[z,y,x])
                        + (ff.uy[z,y,x]-af.uy_past_[z,y,x])*(ff.uy[z,y,x]-af.uy_past_[z,y,x])
                        + (ff.uz[z,y,x]-af.uz_past_[z,y,x])*(ff.uz[z,y,x]-af.uz_past_[z,y,x]);

                e2   += ff.ux[z,y,x]*ff.ux[z,y,x] + ff.uy[z,y,x]*ff.uy[z,y,x]  + ff.uz[z,y,x]*ff.uz[z,y,x];
                af.ux_past_[z, y, x] = ff.ux[z, y, x];
                af.uy_past_[z, y, x] = ff.uy[z, y, x];
                af.uz_past_[z, y, x] = ff.uz[z, y, x];
            }
        }
    }

    compat::println(log_file_, "Time {}: Mass: {}, Px: {}, Py: {}, Pz: {}, Kinetic Energy: {} Relative Error: {}",
                    time_step, mass, px, py, pz, ke, e1/e2);
    std::flush(log_file_);
    if (std::isnan(mass) || std::isnan(px) || std::isnan(py) || std::isnan(pz)) {
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
                compat::print(rho_file,  "{},", ff.rho[z, y, x]);
                compat::print(ux_file,   "{},", ff.ux[z, y, x]);
                compat::print(uy_file,   "{},", ff.uy[z, y, x]);
                compat::print(uz_file,   "{},", ff.uz[z, y, x]);
                compat::print(qxx_file,  "{},", qf.qxx[z, y, x]);
                compat::print(qxy_file,  "{},", qf.qxy[z, y, x]);
                compat::print(qxz_file,  "{},", qf.qxz[z, y, x]);
                compat::print(qyy_file,  "{},", qf.qyy[z, y, x]);
                compat::print(qyz_file,  "{},", qf.qyz[z, y, x]);                
            }
            compat::print(rho_file,  "{}\n", ff.rho[z, y, nx-1]);
            compat::print(ux_file,   "{}\n", ff.ux[z, y, nx-1]);
            compat::print(uy_file,   "{}\n", ff.uy[z, y, nx-1]);
            compat::print(uz_file,   "{}\n", ff.uz[z, y, nx-1]);
            compat::print(qxx_file,  "{}\n", qf.qxx[z, y, nx-1]);
            compat::print(qxy_file,  "{}\n", qf.qxy[z, y, nx-1]);
            compat::print(qxz_file,  "{}\n", qf.qxz[z, y, nx-1]);
            compat::print(qyy_file,  "{}\n", qf.qyy[z, y, nx-1]);
            compat::print(qyz_file,  "{}\n", qf.qyz[z, y, nx-1]);
        }
    }

    if constexpr (Params::kDebugLogging) {
        ExportDistributionCSV(ff, path, step);
    }
}

void SimIO::ExportDistributionCSV(const FluidFields& ff,
                                const std::string& path, int step) {
    std::ofstream f_file;
    for (int i : std::views::iota(0, ndir)) {
        f_file.open(compat::format("{}/f_{}_{}.csv", path, i, step), std::ios::out);
        if (!f_file.is_open())
            throw std::runtime_error("Failed to open data file");

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx-1; ++x) {
                    compat::print(f_file, "{},", ff.f[z, y, x, i]);
                }
                compat::print(f_file, "{}\n", ff.f[z, y, nx-1, i]);
            }
        }
        f_file.close();
    }
}


void SimIO::ExportVTKHDF(const FluidFields& ff, const QTensorFields& qf, AnalysisFields& af,
                         const std::string& path, int step, double time) {
    constexpr int kStepWidth = [] {
        int w = 1, n = kNumSteps - 1;
        while (n >= 10) { n /= 10; ++w; }
        return w;
    }();
    const std::string file_path = std::format("{}/lbm_{:0{}}.vtkhdf", path, step, kStepWidth);

    ImageDataWriter writer(file_path);

    // --- PointData datasets, shape [nz, ny, nx] (z slowest, x fastest) ---

    writer.WriteScalarField("rho", ff.rho_data.data());

    if constexpr (Params::kDebugLogging) {
        // f[z,y,x,i] is interleaved, so we still need a scratch buffer per direction.
        std::vector<double> buf(nx * ny * nz);
        for (int i = 0; i < ndir; i++) {
            for (int z = 0; z < nz; ++z)
                for (int y = 0; y < ny; ++y)
                    for (int x = 0; x < nx; ++x)
                        buf[z * ny * nx + y * nx + x] = ff.f[z, y, x, i];
            writer.WriteScalarField(std::format("f{}", i).c_str(), buf.data());
        }
    }

    writer.WriteScalarField("order", af.order_data_.data());

    // Velocity: three independent scalar fields, written directly from backing stores.
    writer.WriteScalarField("ux", ff.ux_data.data());
    writer.WriteScalarField("uy", ff.uy_data.data());
    writer.WriteScalarField("uz", ff.uz_data.data());

    // Director: AoS layout [nz, ny, nx, 3] matches the mdspan backing store directly.
    writer.WriteVectorField("director", af.director_data_.data());

    // --- FieldData: simulation time ---
    // {
    //     hid_t fd = H5Gcreate2(vtkhdf, "FieldData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    //     hsize_t dim = 1;
    //     double time_val = time;  // your simulation time parameter
    //     hid_t sp = H5Screate_simple(1, &dim, nullptr);
    //     hid_t ds = H5Dcreate2(fd, "TimeValue", H5T_NATIVE_DOUBLE, sp,
    //                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    //     H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &time_val);
    //     H5Dclose(ds);
    // }
}

