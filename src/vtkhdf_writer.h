#ifndef LBM_AN_VTKHDFWRITER_H_
#define LBM_AN_VTKHDFWRITER_H_

#include "hdf5_internals.h"
#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <params.h>

using namespace Params;

// Convenience factories that return owning handles.
inline H5File create_file(const std::string& path) {
    return H5File{ H5Fcreate(path.c_str(), H5F_ACC_TRUNC,
                             H5P_DEFAULT, H5P_DEFAULT), path.c_str() };
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

    VTKHDFWriter(const std::string& filepath, const std::string& type) {
        file_ = create_file(filepath);
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
    ImageDataWriter(const std::string& filepath)
        : VTKHDFWriter(filepath, "ImageData")
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
    void WriteScalarField(const char* name, const double* data) {
        const hsize_t dims[3] = {(hsize_t)nz, (hsize_t)ny, (hsize_t)nx};
        H5Dataspace sp { H5Screate_simple(3, dims, nullptr), "H5Screate_simple (scalar field)" };
        H5Dataset ds   { H5Dcreate2(pdata_, name, H5T_NATIVE_DOUBLE, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), name};
        herr_t st = H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        check(st, "WriteScalarField");
    }

    // Write a [nz, ny, nx, components] vector field from a contiguous AoS buffer.
    void WriteVectorField(const char* name, const double* data, int components = 3) {
        const hsize_t dims[4] = {
            (hsize_t)nz, (hsize_t)ny, (hsize_t)nx, (hsize_t)components
        };
        H5Dataspace sp{ H5Screate_simple(4, dims, nullptr), "H5Screate_simple (vector field)" };
        H5Dataset ds  { H5Dcreate2(pdata_, name, H5T_NATIVE_DOUBLE, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), name };
        herr_t st = H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        check(st, "WriteVectorField");

    }
};


// Writes a VTKHDF UnstructuredGrid file (arbitrary topology).
class UnstructuredGridWriter : public VTKHDFWriter {
    H5Group pdata_;
    int64_t n_points_ = 0;

public:
    UnstructuredGridWriter(const std::string& filepath)
        : VTKHDFWriter(filepath, "UnstructuredGrid")
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
};

#endif // LBM_AN_VTKHDFWRITER_H_
