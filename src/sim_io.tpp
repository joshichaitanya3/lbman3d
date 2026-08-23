#ifndef LBM_AN_SIM_IO_TPP_
#define LBM_AN_SIM_IO_TPP_

#include <sim_config.h>
#include "vtkhdf_writer.h"
#include "sim_config_attrs.h"
#include "disclination_io.h"
#include "analysis/disclination.h"
#include "lattice_stencil.h"
#include "format_compat.h"

template<typename BC>
void SimIO::ExportVTKHDF(const FluidFields& ff, AnalysisFields& af,
                         const std::string& path, int step,
                         const MPIContext& ctx, const LocalGrid& grid) {
    constexpr int kStepWidth = [] {
        int w = 1, n = kNumSteps - 1;
        while (n >= 10) { n /= 10; ++w; }
        return w;
    }();
    const LocalGrid& g = ff.grid;

    const std::string file_path = compat::format("{}/lbm_{:0{}}.vtkhdf", path, step, kStepWidth);

    ImageDataWriter writer(file_path, ctx);
    SimConfigAttr::StampSimConfigAttributes<BC>(writer.root());

    // --- PointData datasets, shape [nz, ny, nx] (z slowest, x fastest) ---

    writer.WriteScalarField("rho", ff.rho.data(), grid);

    if constexpr (Params::kDebugLogging) {
        // ff.f is laid out with i fastest-varying (host idx() layout), so we still
        // need a scratch buffer per direction with the export's [z,y,x] layout.
        std::vector<double> buf(g.HaloVolume());
        for (int i = 0; i < Lattice::ndir; i++) {
            for (int z = 0; z < g.local_nz; ++z)
                for (int y = 0; y < g.local_ny; ++y)
                    for (int x = 0; x < g.local_nx; ++x)
                        buf[g.halo_idx(x, y, z)] = ff.f[g.halo_idx(x, y, z, i)];
            writer.WriteScalarField(compat::format("f{}", i).c_str(), buf.data(), grid);
        }
    }

    writer.WriteScalarField("order", af.order_.data(), grid);

    // Velocity: three independent scalar fields, written directly from backing stores.
    writer.WriteScalarField("ux", ff.ux.data(), grid);
    writer.WriteScalarField("uy", ff.uy.data(), grid);
    writer.WriteScalarField("uz", ff.uz.data(), grid);

    // Director: AoS layout [nz, ny, nx, 3] (see dirIdx in analysis_fields.h) passes
    // straight through to WriteVectorField without repacking.
    writer.WriteVectorField("director", af.director_.data(), grid);
}

template<typename BC>
void SimIO::ExportDisclinations(
    const DefectFields& df,
    const std::string& path,
    int step,
    const MPIContext& ctx,
    const LocalGrid&
) {
    constexpr int kStepWidth = [] {
        int w = 1, n = kNumSteps - 1;
        while (n >= 10) { n /= 10; ++w; }
        return w;
    }();
    const std::string file_path = compat::format("{}/disclinations_{:0{}}.vtkhdf", path, step, kStepWidth);

    // Delegates the mesh build and dataset writes; stamps sim-config attributes
    // so find_defects can trace disclinations_*.vtkhdf back to the run that
    // produced them (symmetric with the lbm_*.vtkhdf stamping above).
    WriteDisclinationsVTKHDF<BC>(df, file_path, ctx);
}

#endif // LBM_AN_SIM_IO_TPP_
