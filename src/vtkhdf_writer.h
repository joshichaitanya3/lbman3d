#ifndef LBM_AN_VTKHDFWRITER_H_
#define LBM_AN_VTKHDFWRITER_H_

#include "hdf5_internals.h"
#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <params.h>
#include "mpi/mpi_context.h"
#include "format_compat.h"

using namespace Params;

// Convenience factories that return owning handles.
inline H5File create_file(const std::string& path, hid_t fapl) {
    return H5File{ H5Fcreate(path.c_str(), H5F_ACC_TRUNC,
                             H5P_DEFAULT, fapl), path.c_str() };
}
inline H5Group create_group(hid_t parent, const char* name) {
    return H5Group{ H5Gcreate2(parent, name, H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT), name };
}

// Base class: owns the HDF5 file/group RAII and the low-level write primitives.
// Not intended to be used directly — construct ImageDataWriter or UnstructuredGridWriter.
class VTKHDFWriter {
protected:
    H5File file_;
    H5Group root_;

    static void check(hid_t id, const char* what) {
        if (id < 0) throw std::runtime_error(std::string("HDF5 failed: ") + what);
    }
    // Attach a fixed-length ASCII string attribute to an object.
    static void write_string_attribute(hid_t obj, const char* name,
                                       const std::string& value)
    {
        H5Datatype atype{ H5Tcopy(H5T_C_S1), "H5Tcopy" };
        H5Tset_size(atype, value.size());           // exact length, no NUL
        H5Tset_strpad(atype, H5T_STR_NULLPAD);
        H5Tset_cset(atype, H5T_CSET_ASCII);
        H5Dataspace aspace{ H5Screate(H5S_SCALAR), "H5SCreate" };
        H5Attribute attr  { H5Acreate2(obj, name, atype, aspace, H5P_DEFAULT, H5P_DEFAULT), name };
        if (H5Awrite(attr, atype, value.data()) < 0)
            throw std::runtime_error(std::string("H5Awrite failed: ") + name);
    }

    // Attach a 1-D integer attribute (used for Version = [major, minor]).
    template <typename T>
    static void write_array_attribute(hid_t obj, const char* name,
                                      const std::vector<T>& values)
    {
        hsize_t dims[1] = { values.size() };
        H5Dataspace aspace{ H5Screate_simple(1, dims, nullptr), "H5Screate_simple" };
        H5Attribute attr  { H5Acreate2(obj, name, H5Native<T>::id(), aspace,
                                   H5P_DEFAULT, H5P_DEFAULT), name };
        herr_t st = H5Awrite(attr, H5Native<T>::id(), values.data());
        check(st, "H5Awrite");
    }

    // Write a 1-D dataset of the given native type into the parent group.
    template <typename T>
    static void write_1d(hid_t parent, const char* name,
                         const std::vector<T>& data)
    {
        hsize_t dims[1] = { data.size() };
        H5Dataspace space{ H5Screate_simple(1, dims, nullptr), "H5Screate_simple (1D)" };
        H5Dataset dset   { H5Dcreate2(parent, name, H5Native<T>::id(), space,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), name};
        herr_t st = H5Dwrite(dset, H5Native<T>::id(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());    
        check(st, "H5Dwrite (1D)");
    }

    // Write a 2-D dataset of shape [rows, cols] from a row-major buffer.
    template <typename T>
    static void write_2d(hid_t parent, const char* name,
                         const std::vector<T>& data,
                         hsize_t rows, hsize_t cols)
    {
        if (data.size() != rows * cols)
            throw std::runtime_error("write_2d: size mismatch");
        hsize_t dims[2] = { rows, cols };
        H5Dataspace space{ H5Screate_simple(2, dims, nullptr), "H5Screate_simple (2D)" };
        H5Dataset dset   { H5Dcreate2(parent, name, H5Native<T>::id(), space,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), name };
        herr_t st = H5Dwrite(dset, H5Native<T>::id(), H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, data.data());
        check(st, "H5Dwrite (2D)");
    }

    VTKHDFWriter(
        const std::string& filepath,
        const std::string& type,
        [[maybe_unused]] const MPIContext& ctx
    ) {
        #ifdef LBM_ENABLE_MPI
        /*
        * Set up file access property list with parallel I/O access
        */
        H5Plist fapl {H5Pcreate(H5P_FILE_ACCESS), "Setup file access with parallel I/O"};
        H5Pset_fapl_mpio(fapl, ctx.cart_comm, ctx.info);

        /*
        * OPTIONAL: It is generally recommended to set collective
        *           metadata reads on FAPL to perform metadata reads
        *           collectively, which usually allows datasets
        *           to perform better at scale, although it is not
        *           strictly necessary.
        */
        H5Pset_all_coll_metadata_ops(fapl, true);

        /*
        * OPTIONAL: It is generally recommended to set collective
        *           metadata writes on FAPL to perform metadata writes
        *           collectively, which usually allows datasets
        *           to perform better at scale, although it is not
        *           strictly necessary.
        */
        H5Pset_coll_metadata_write(fapl, true);
        file_ = create_file(filepath, fapl);
        #else
        file_ = create_file(filepath, H5P_DEFAULT);
        #endif
        root_ = create_group(file_, "VTKHDF");
        write_string_attribute(root_, "Type", type);
        write_array_attribute<int64_t>(root_, "Version", {2, 0});
    }

public:
    virtual ~VTKHDFWriter() = default;
    VTKHDFWriter(const VTKHDFWriter&) = delete;
    VTKHDFWriter& operator=(const VTKHDFWriter&) = delete;
};


// Writes a VTKHDF ImageData file (regular grid fields).
// PointData group is created on construction and closed on destruction.
class ImageDataWriter : public VTKHDFWriter {
    H5Group pdata_;

public:
    ImageDataWriter(const std::string& filepath, const MPIContext& ctx)
        : VTKHDFWriter(filepath, "ImageData", ctx)
    {
        write_array_attribute<int64_t>(root_, "WholeExtent",
            {0, nx-1, 0, ny-1, 0, nz-1});
        write_array_attribute<double>(root_, "Origin",  {0.0, 0.0, 0.0});
        write_array_attribute<double>(root_, "Spacing", {1.0, 1.0, 1.0});
        // Direction = identity matrix (3x3, row-major, flattened to 9 doubles)
        write_array_attribute<double>(root_, "Direction",
            {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});

        pdata_ = create_group(root_, "PointData");
    }

    // Write a [nz, ny, nx] scalar field from a contiguous row-major buffer.
    void WriteScalarField(const char* name, const double* data, [[maybe_unused]] const LocalGrid& grid) {

        #ifndef LBM_ENABLE_MPI
        const hsize_t dims[3] = {(hsize_t)nz, (hsize_t)ny, (hsize_t)nx};
        H5Dataspace sp { H5Screate_simple(3, dims, nullptr), "H5Screate_simple (scalar field)" };
        H5Dataset ds   { H5Dcreate2(pdata_, name, H5T_NATIVE_DOUBLE, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), name};
        herr_t st = H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        check(st, "WriteScalarField");
        #else
        const hsize_t dims_global[3] = {
            (hsize_t)nz,
            (hsize_t)ny,
            (hsize_t)nx
        };

        hsize_t dims_mem[3] = {
            (hsize_t)grid.local_nz + 2*grid.kHaloMPI,
            (hsize_t)grid.local_ny + 2*grid.kHaloMPI,
            (hsize_t)grid.local_nx + 2*grid.kHaloMPI
        };

        hsize_t dims_chunk[3] = {
            std::min((hsize_t)CHUNK_LINEAR_SIZE, dims_global[0]),
            std::min((hsize_t)CHUNK_LINEAR_SIZE, dims_global[1]),
            std::min((hsize_t)CHUNK_LINEAR_SIZE, dims_global[2])
        };

        int dataspace_num_dims = 3;

        H5Dataspace filespace { H5Screate_simple(dataspace_num_dims, dims_global, nullptr), "H5Screate_simple (scalar field) (filespace)"};
        H5Dataspace memspace  { H5Screate_simple(dataspace_num_dims, dims_mem, nullptr), "H5Screate_simple (scalar field) (memspace)" };

        /*
        * Create chunked dataset.
        */
        H5Plist dcpl {H5Pcreate(H5P_DATASET_CREATE), "H5Pcreate (chunked dataset)"};
        H5Pset_chunk(dcpl, dataspace_num_dims, dims_chunk);
        
        H5Dataset dset_id { H5Dcreate(pdata_, name, H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT), name};

        hsize_t count[3] = {
            (hsize_t)1,
            (hsize_t)1,
            (hsize_t)1
        };
        hsize_t stride[3] = {
            (hsize_t)1,
            (hsize_t)1,
            (hsize_t)1
        };
        hsize_t block[3] = {
            (hsize_t)grid.local_nz,
            (hsize_t)grid.local_ny,
            (hsize_t)grid.local_nx
        };
        hsize_t offset[3] = {
            (hsize_t)grid.offset_z,
            (hsize_t)grid.offset_y,
            (hsize_t)grid.offset_x
        };
        
        H5Dataspace fs {H5Dget_space(dset_id), "get filespace"};
        herr_t status = H5Sselect_hyperslab(fs, H5S_SELECT_SET, offset, stride, count, block);
        check(status, "Select hyperslab in the file.");

        /*
        * Select the interior (non-halo) hyperslab in memory: same block shape
        * as the file selection, but offset by grid.kHaloMPI in every axis to skip the
        * ghost-cell layer that pads `field`.
        */
        hsize_t mem_offset[3] = {
            (hsize_t)grid.kHaloMPI,
            (hsize_t)grid.kHaloMPI,
            (hsize_t)grid.kHaloMPI
        };

        status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offset, stride, count, block);
        check(status, "Select the interior (non-halo) hyperslab in memory.");

        /*
        * Create property list for collective dataset write.
        */
        H5Plist dxpl {H5Pcreate(H5P_DATASET_XFER), "HDF5 property list for data transfer"};

        /*
            Known HDF5 library bug: collective I/O to shared chunks
            Reproduced on this system's HDF5 1.10.7 (Ubuntu package):
            when a chunk is jointly written by more than one rank
            (chunk shape doesn't align with the decomposition — either because the domain splits unevenly,
            or because the chunk shape is intentionally decoupled from decomposition),
            collective I/O (H5FD_MPIO_COLLECTIVE) silently corrupts data (large zeroed regions),
            reproduced with both the default and forced-linked-chunk (H5FD_MPIO_CHUNK_ONE_IO) collective strategies.
            Independent I/O (H5FD_MPIO_INDEPENDENT) with the identical selections is correct.
            Root cause suspected to be a version-specific bug in 1.10.x's multi-chunk/linked-chunk collective I/O path,
            not confirmed against a newer HDF5. Until re-tested on a newer HDF5 release, use H5FD_MPIO_INDEPENDENT
            for dataset writes whenever chunk sharing is possible.
         */
        H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_INDEPENDENT);

        status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, fs, dxpl, data);
        check(status, "WriteScalarField");
        #endif
    }

    // Write a [nz, ny, nx, components] vector field from a contiguous AoS buffer.
    void WriteVectorField(const char* name, const double* data, [[maybe_unused]] const LocalGrid& grid, int components = 3) {

        #ifndef LBM_ENABLE_MPI
        const hsize_t dims[4] = {
            (hsize_t)nz, (hsize_t)ny, (hsize_t)nx, (hsize_t)components
        };
        H5Dataspace sp{ H5Screate_simple(4, dims, nullptr), "H5Screate_simple (vector field)" };
        H5Dataset ds  { H5Dcreate2(pdata_, name, H5T_NATIVE_DOUBLE, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), name };
        herr_t st = H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        check(st, "WriteVectorField");
        #else
        const hsize_t dims_global[4] = {
            (hsize_t)nz,
            (hsize_t)ny,
            (hsize_t)nx,
            (hsize_t)components
        };

        hsize_t dims_mem[4] = {
            (hsize_t)grid.local_nz + 2*grid.kHaloMPI,
            (hsize_t)grid.local_ny + 2*grid.kHaloMPI,
            (hsize_t)grid.local_nx + 2*grid.kHaloMPI,
            (hsize_t)components
        };

        hsize_t dims_chunk[4] = {
            std::min((hsize_t)CHUNK_LINEAR_SIZE, dims_global[0]),
            std::min((hsize_t)CHUNK_LINEAR_SIZE, dims_global[1]),
            std::min((hsize_t)CHUNK_LINEAR_SIZE, dims_global[2]),
            (hsize_t)components
        };

        int dataspace_num_dims = 4;

        H5Dataspace filespace { H5Screate_simple(dataspace_num_dims, dims_global, nullptr), "H5Screate_simple (vector field) (filespace)"};
        H5Dataspace memspace  { H5Screate_simple(dataspace_num_dims, dims_mem, nullptr), "H5Screate_simple (vector field) (memspace)" };

        /*
        * Create chunked dataset.
        */
        H5Plist dcpl {H5Pcreate(H5P_DATASET_CREATE), "H5Pcreate (chunked dataset)"};
        H5Pset_chunk(dcpl, dataspace_num_dims, dims_chunk);
        
        H5Dataset dset_id { H5Dcreate(pdata_, name, H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT), name};

        hsize_t count[4] = {
            (hsize_t)1,
            (hsize_t)1,
            (hsize_t)1,
            (hsize_t)1
        };
        hsize_t stride[4] = {
            (hsize_t)1,
            (hsize_t)1,
            (hsize_t)1,
            (hsize_t)1
        };
        hsize_t block[4] = {
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
        
        H5Dataspace fs {H5Dget_space(dset_id), "get filespace"};
        herr_t status = H5Sselect_hyperslab(fs, H5S_SELECT_SET, offset, stride, count, block);
        check(status, "Select hyperslab in the file.");

        /*
        * Select the interior (non-halo) hyperslab in memory: same block shape
        * as the file selection, but offset by grid.kHaloMPI in every axis to skip the
        * ghost-cell layer that pads `field`.
        */
        hsize_t mem_offset[4] = {
            (hsize_t)grid.kHaloMPI,
            (hsize_t)grid.kHaloMPI,
            (hsize_t)grid.kHaloMPI,
            (hsize_t)0
        };

        status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offset, stride, count, block);
        check(status, "Select the interior (non-halo) hyperslab in memory.");

        /*
        * Create property list for collective dataset write.
        */
        H5Plist dxpl {H5Pcreate(H5P_DATASET_XFER), "HDF5 property list for data transfer"};

        /*
            Known HDF5 library bug: collective I/O to shared chunks
            Reproduced on this system's HDF5 1.10.7 (Ubuntu package):
            when a chunk is jointly written by more than one rank
            (chunk shape doesn't align with the decomposition — either because the domain splits unevenly,
            or because the chunk shape is intentionally decoupled from decomposition),
            collective I/O (H5FD_MPIO_COLLECTIVE) silently corrupts data (large zeroed regions),
            reproduced with both the default and forced-linked-chunk (H5FD_MPIO_CHUNK_ONE_IO) collective strategies.
            Independent I/O (H5FD_MPIO_INDEPENDENT) with the identical selections is correct.
            Root cause suspected to be a version-specific bug in 1.10.x's multi-chunk/linked-chunk collective I/O path,
            not confirmed against a newer HDF5. Until re-tested on a newer HDF5 release, use H5FD_MPIO_INDEPENDENT
            for dataset writes whenever chunk sharing is possible.
         */
        H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_INDEPENDENT);

        status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, fs, dxpl, data);
        check(status, "WriteVectorField");
        #endif
    }
};


// Writes a VTKHDF UnstructuredGrid file (arbitrary topology).
class UnstructuredGridWriter : public VTKHDFWriter {
    H5Group pdata_;
    int64_t n_points_ = 0;

public:
    UnstructuredGridWriter(const std::string& filepath, const MPIContext& ctx)
        : VTKHDFWriter(filepath, "UnstructuredGrid", ctx)
    {}

    // Write all topology datasets required by the VTKHDF UnstructuredGrid spec.
    // Must be called before WriteScalarPointField.
    void WriteTopology(const std::vector<double>&   points,
                       const std::vector<int64_t>&  connectivity,
                       const std::vector<int64_t>&  offsets,
                       const std::vector<uint8_t>&  cell_types)
    {
        n_points_ = static_cast<int64_t>(points.size() / 3);
        const int64_t n_cells = static_cast<int64_t>(cell_types.size());
        write_1d<int64_t>(root_, "NumberOfPoints",          {n_points_});
        write_1d<int64_t>(root_, "NumberOfCells",           {n_cells});
        write_1d<int64_t>(root_, "NumberOfConnectivityIds", {static_cast<int64_t>(connectivity.size())});
        write_2d<double>(root_, "Points",       points, n_points_, 3);
        write_1d<int64_t>(root_, "Connectivity", connectivity);
        write_1d<int64_t>(root_, "Offsets",      offsets);
        write_1d<uint8_t>(root_, "Types",        cell_types);
    }

    // Write a scalar field over the mesh points into the PointData group.
    // data.size() must equal the number of points written by WriteTopology.
    void WriteScalarPointField(const char* name, const std::vector<double>& data) {
        if (static_cast<int64_t>(data.size()) != n_points_)
            throw std::runtime_error(
                std::string("WriteScalarPointField: size mismatch for ") + name);
        if (pdata_.get() < 0)
            pdata_ = create_group(root_, "PointData");
        write_1d(pdata_, name, data);
    }

    // Write a vector field over the mesh points into the PointData group.
    // data.size() must equal n_points * components.
    void WriteVectorPointField(const char* name, const std::vector<double>& data,
                               int components = 3) {
        if (static_cast<int64_t>(data.size()) != n_points_ * components)
            throw std::runtime_error(
                compat::format("WriteVectorPointField: size mismatch for {} [data.size(): {}, n_points_: {}]", name, data.size(), n_points_)
            );
        if (pdata_.get() < 0)
            pdata_ = create_group(root_, "PointData");
        write_2d(pdata_, name, data, static_cast<hsize_t>(n_points_),
                 static_cast<hsize_t>(components));
    }
};

#endif // LBM_AN_VTKHDFWRITER_H_
