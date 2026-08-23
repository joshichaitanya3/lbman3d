#ifndef LBM_AN_DISCLINATION_IO_H_
#define LBM_AN_DISCLINATION_IO_H_

#include <string>
#include "vtkhdf_writer.h"
#include "sim_config_attrs.h"
#include "analysis/defect_fields.h"
#include "analysis/disclination.h"
#include "mpi/mpi_context.h"

// Builds a DisclinationMesh from the disclinations in `df` and writes it to
// `file_path` as a VTKHDF UnstructuredGrid, stamped with the compile-time
// sim config so downstream tools can trace the file to its run.
//
// Shared between SimIO::ExportDisclinations (the in-simulation path, which
// formats the file path against Params::kNumSteps) and the offline
// find_defects binary (which formats against the max step number it
// enumerated). This helper cares about neither — just the file path.
//
// On empty frames the point-data arrays (Tangents, Beta) are still emitted
// as empty datasets so the time-series schema stays uniform — ParaView
// Calculators in the visualise script trip when an array appears midway
// through a sequence.
template<typename BC>
inline void WriteDisclinationsVTKHDF(const DefectFields& df,
                                     const std::string& file_path,
                                     const MPIContext& ctx) {
    DisclinationMesh mesh;
    for (const Disclination& d : df.disclinations) {
        mesh.AddDisclination(d);
    }

    UnstructuredGridWriter writer(file_path, ctx);
    SimConfigAttr::StampSimConfigAttributes<BC>(writer.root());

    writer.WriteTopology(mesh.Points(), mesh.Connectivity(),
                         mesh.Offsets(), mesh.CellTypes());

    if (mesh.TangentsAvailable() || mesh.NumPoints() == 0) {
        writer.WriteVectorPointField("Tangents", mesh.Tangents());
    }
    if (mesh.BetaAvailable() || mesh.NumPoints() == 0) {
        writer.WriteScalarPointField("Beta", mesh.Beta());
    }
}

#endif // LBM_AN_DISCLINATION_IO_H_
