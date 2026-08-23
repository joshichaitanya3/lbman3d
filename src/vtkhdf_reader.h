#ifndef LBM_AN_VTKHDFREADER_H_
#define LBM_AN_VTKHDFREADER_H_

#include "hdf5_internals.h"
#include "sim_config_attrs.h"
#include "local_grid.h"
#include "mpi/mpi_context.h"
#include "format_compat.h"
#include <cstdint>
#include <params.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Params;

// Mirror of vtkhdf_writer.h for reading VTKHDF ImageData files
// produced by ImageDataWriter. Same fapl setup under LBM_ENABLE_MPI, same
// halo-interior memspace convention, same H5FD_MPIO_INDEPENDENT workaround
// for the HDF5 1.10.7 collective-chunk corruption bug documented in the
// writer — this is the symmetric read path.
//
// UnstructuredGridReader is intentionally omitted; nothing in the current
// find_defects workflow needs it. Add when needed.

inline H5File open_file(const std::string& path, hid_t fapl) {
    return H5File{ H5Fopen(path.c_str(), H5F_ACC_RDONLY, fapl), path.c_str() };
}
inline H5Group open_group(hid_t parent, const char* name) {
    return H5Group{ H5Gopen2(parent, name, H5P_DEFAULT), name };
}

class VTKHDFReader {
protected:
    H5File file_;
    H5Group root_;

    static void check(hid_t id, const char* what) {
        if (id < 0) throw std::runtime_error(std::string("HDF5 failed: ") + what);
    }

    VTKHDFReader(
        const std::string& filepath,
        const std::string& expected_type,
        [[maybe_unused]] const MPIContext& ctx
    ) {
        #ifdef LBM_ENABLE_MPI
        H5Plist fapl{ H5Pcreate(H5P_FILE_ACCESS), "Setup file access with parallel I/O" };
        H5Pset_fapl_mpio(fapl, ctx.cart_comm, ctx.info);
        H5Pset_all_coll_metadata_ops(fapl, true);
        H5Pset_coll_metadata_write(fapl, true);
        file_ = open_file(filepath, fapl);
        #else
        file_ = open_file(filepath, H5P_DEFAULT);
        #endif
        root_ = open_group(file_, "VTKHDF");

        const std::string got_type = SimConfigAttr::read_string_attr(root_, "Type");
        if (got_type != expected_type) {
            throw std::runtime_error(compat::format(
                "VTKHDFReader: {} is a VTKHDF/{}; expected VTKHDF/{}.",
                filepath, got_type, expected_type));
        }
    }

public:
    virtual ~VTKHDFReader() = default;
    VTKHDFReader(const VTKHDFReader&) = delete;
    VTKHDFReader& operator=(const VTKHDFReader&) = delete;

    // Read the sim-config snapshot stamped by the writer. Every field is
    // populated; missing attributes throw (mismatched file format).
    SimConfigAttr::SimConfigSnapshot ReadSimConfig() const {
        return SimConfigAttr::ReadSimConfigAttributes(root_.get());
    }
};


class ImageDataReader : public VTKHDFReader {
    H5Group pdata_;

    static std::vector<int64_t> read_int64_array_attr(hid_t obj, const char* name) {
        H5Attribute attr{ H5Aopen(obj, name, H5P_DEFAULT), name };
        H5Dataspace sp{ H5Aget_space(attr), "H5Aget_space" };
        const int ndims = H5Sget_simple_extent_ndims(sp);
        std::vector<hsize_t> dims(ndims);
        H5Sget_simple_extent_dims(sp, dims.data(), nullptr);
        hsize_t total = 1;
        for (int d : {0}) { (void)d; }
        for (auto v : dims) total *= v;
        std::vector<int64_t> out(total);
        if (H5Aread(attr, H5T_NATIVE_INT64, out.data()) < 0)
            throw std::runtime_error(std::string("H5Aread failed: ") + name);
        return out;
    }

public:
    ImageDataReader(const std::string& filepath, const MPIContext& ctx)
        : VTKHDFReader(filepath, "ImageData", ctx)
    {
        // Validate WholeExtent matches the compile-time Params grid. If a
        // future refactor moves grid dims to runtime (see the "Runtime grid
        // dims" note in CLAUDE.md), this becomes: read WholeExtent → size
        // LocalGrid from it. For now we still size off Params::n{x,y,z}, so
        // a mismatch is a fatal build/data pairing bug.
        const auto extent = read_int64_array_attr(root_, "WholeExtent");
        if (extent.size() != 6)
            throw std::runtime_error("ImageDataReader: WholeExtent must have 6 entries");

        const int64_t got_nx = extent[1] - extent[0] + 1;
        const int64_t got_ny = extent[3] - extent[2] + 1;
        const int64_t got_nz = extent[5] - extent[4] + 1;
        if (got_nx != static_cast<int64_t>(nx) ||
            got_ny != static_cast<int64_t>(ny) ||
            got_nz != static_cast<int64_t>(nz))
        {
            throw std::runtime_error(compat::format(
                "ImageDataReader: {} has WholeExtent {}x{}x{}, "
                "but this build was compiled for {}x{}x{}. "
                "Rebuild with a matching params.h.",
                filepath, got_nx, got_ny, got_nz, nx, ny, nz));
        }

        pdata_ = open_group(root_, "PointData");
    }

    // Read a [nz, ny, nx] scalar field into `data`, filling the interior of
    // a halo-padded buffer sized to grid.HaloVolume() (identical layout to
    // ImageDataWriter::WriteScalarField, so reader and writer round-trip
    // through the same buffer without repacking).
    void ReadScalarField(const char* name, double* data,
                         [[maybe_unused]] const LocalGrid& grid) {
        #ifndef LBM_ENABLE_MPI
        H5Dataset ds{ H5Dopen2(pdata_, name, H5P_DEFAULT), name };
        herr_t st = H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        check(st, "ReadScalarField");
        #else
        // Mirror WriteScalarField's MPI hyperslab logic exactly: this rank
        // reads its owned block into the interior (halo-offset) of the
        // supplied buffer.
        const hsize_t dims_global[3] = {
            (hsize_t)nz, (hsize_t)ny, (hsize_t)nx
        };
        hsize_t dims_mem[3] = {
            (hsize_t)grid.local_nz + 2*grid.kHaloMPI,
            (hsize_t)grid.local_ny + 2*grid.kHaloMPI,
            (hsize_t)grid.local_nx + 2*grid.kHaloMPI
        };

        const int ndims = 3;
        H5Dataset dset{ H5Dopen2(pdata_, name, H5P_DEFAULT), name };
        H5Dataspace filespace{ H5Dget_space(dset), "H5Dget_space (scalar field)" };
        H5Dataspace memspace{
            H5Screate_simple(ndims, dims_mem, nullptr),
            "H5Screate_simple (scalar field memspace)" };

        hsize_t count[3]  = {1, 1, 1};
        hsize_t stride[3] = {1, 1, 1};
        hsize_t block[3]  = {
            (hsize_t)grid.local_nz,
            (hsize_t)grid.local_ny,
            (hsize_t)grid.local_nx
        };
        hsize_t offset[3] = {
            (hsize_t)grid.offset_z,
            (hsize_t)grid.offset_y,
            (hsize_t)grid.offset_x
        };
        herr_t st = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, stride, count, block);
        check(st, "Select hyperslab in the file.");

        hsize_t mem_offset[3] = {
            (hsize_t)grid.kHaloMPI,
            (hsize_t)grid.kHaloMPI,
            (hsize_t)grid.kHaloMPI
        };
        st = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offset, stride, count, block);
        check(st, "Select the interior (non-halo) hyperslab in memory.");

        H5Plist dxpl{ H5Pcreate(H5P_DATASET_XFER), "HDF5 property list for data transfer" };
        // Same reason as WriteScalarField's independent-IO choice — the
        // 1.10.7 collective-chunk corruption applies symmetrically to reads
        // once revisited on newer HDF5.
        H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_INDEPENDENT);

        (void)dims_global;
        st = H5Dread(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl, data);
        check(st, "ReadScalarField");
        #endif
    }

    // Read a [nz, ny, nx, components] vector field into a contiguous AoS
    // buffer; components is 3 for the director field.
    void ReadVectorField(const char* name, double* data,
                         [[maybe_unused]] const LocalGrid& grid,
                         [[maybe_unused]] int components = 3) {
        #ifndef LBM_ENABLE_MPI
        H5Dataset ds{ H5Dopen2(pdata_, name, H5P_DEFAULT), name };
        herr_t st = H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        check(st, "ReadVectorField");
        #else
        const hsize_t dims_global[4] = {
            (hsize_t)nz, (hsize_t)ny, (hsize_t)nx, (hsize_t)components
        };
        hsize_t dims_mem[4] = {
            (hsize_t)grid.local_nz + 2*grid.kHaloMPI,
            (hsize_t)grid.local_ny + 2*grid.kHaloMPI,
            (hsize_t)grid.local_nx + 2*grid.kHaloMPI,
            (hsize_t)components
        };

        const int ndims = 4;
        H5Dataset dset{ H5Dopen2(pdata_, name, H5P_DEFAULT), name };
        H5Dataspace filespace{ H5Dget_space(dset), "H5Dget_space (vector field)" };
        H5Dataspace memspace{
            H5Screate_simple(ndims, dims_mem, nullptr),
            "H5Screate_simple (vector field memspace)" };

        hsize_t count[4]  = {1, 1, 1, 1};
        hsize_t stride[4] = {1, 1, 1, 1};
        hsize_t block[4]  = {
            (hsize_t)grid.local_nz,
            (hsize_t)grid.local_ny,
            (hsize_t)grid.local_nx,
            (hsize_t)components
        };
        hsize_t offset[4] = {
            (hsize_t)grid.offset_z,
            (hsize_t)grid.offset_y,
            (hsize_t)grid.offset_x,
            (hsize_t)0
        };
        herr_t st = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, stride, count, block);
        check(st, "Select hyperslab in the file.");

        hsize_t mem_offset[4] = {
            (hsize_t)grid.kHaloMPI,
            (hsize_t)grid.kHaloMPI,
            (hsize_t)grid.kHaloMPI,
            (hsize_t)0
        };
        st = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offset, stride, count, block);
        check(st, "Select the interior (non-halo) hyperslab in memory.");

        H5Plist dxpl{ H5Pcreate(H5P_DATASET_XFER), "HDF5 property list for data transfer" };
        H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_INDEPENDENT);

        (void)dims_global;
        st = H5Dread(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl, data);
        check(st, "ReadVectorField");
        #endif
    }
};

#endif // LBM_AN_VTKHDFREADER_H_
