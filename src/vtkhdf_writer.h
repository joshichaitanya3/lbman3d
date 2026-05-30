#ifndef LBM_AN_VTKHDFWRITER_H_
#define LBM_AN_VTKHDFWRITER_H_

#include <hdf5.h>
#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>
#include "params.h"

using namespace Params;

/*
 * Here, we create a map between the C++ types and H5T_NATIVE types
 * The primary template H5Native can be used as follows to get the
 * right Native type for a given C++ type:
 * H5Native<T>::id()
 *
 * Instantiating this template is a hard error, catching unsupported
 * types at compile time.
 */
template <class T>
struct H5Native {
    static_assert(sizeof(T) == 0,
        "No HDF5 native type mapping for T. "
        "Add an explicit specialization of H5Native<T>.");
};

// Specializations for the types you actually use.
template <> struct H5Native<double>   { static hid_t id() { return H5T_NATIVE_DOUBLE; } };
template <> struct H5Native<float>    { static hid_t id() { return H5T_NATIVE_FLOAT;  } };
template <> struct H5Native<int32_t>  { static hid_t id() { return H5T_NATIVE_INT32;  } };
template <> struct H5Native<uint32_t> { static hid_t id() { return H5T_NATIVE_UINT32; } };
template <> struct H5Native<int64_t>  { static hid_t id() { return H5T_NATIVE_INT64;  } };
template <> struct H5Native<uint64_t> { static hid_t id() { return H5T_NATIVE_UINT64; } };
template <> struct H5Native<uint8_t>  { static hid_t id() { return H5T_NATIVE_UINT8;  } };


// Base class: owns the HDF5 file/group RAII and the low-level write primitives.
// Not intended to be used directly — construct ImageDataWriter or UnstructuredGridWriter.
class VTKHDFWriter {
protected:
    hid_t file_;
    hid_t root_;

    static void check(hid_t id, const char* what) {
        if (id < 0) throw std::runtime_error(std::string("HDF5 failed: ") + what);
    }

    // Attach a fixed-length ASCII string attribute to an object.
    static void write_string_attribute(hid_t obj, const char* name,
                                       const std::string& value)
    {
        hid_t atype = H5Tcopy(H5T_C_S1);
        check(atype, "H5Tcopy");
        H5Tset_size(atype, value.size());           // exact length, no NUL
        H5Tset_strpad(atype, H5T_STR_NULLPAD);
        H5Tset_cset(atype, H5T_CSET_ASCII);
        hid_t aspace = H5Screate(H5S_SCALAR);
        hid_t attr   = H5Acreate2(obj, name, atype, aspace, H5P_DEFAULT, H5P_DEFAULT);
        check(attr, name);
        H5Awrite(attr, atype, value.data());
        H5Aclose(attr);
        H5Sclose(aspace);
        H5Tclose(atype);
    }

    // Attach a 1-D integer attribute (used for Version = [major, minor]).
    template <typename T>
    static void write_array_attribute(hid_t obj, const char* name,
                                      const std::vector<T>& values)
    {
        hsize_t dims[1] = { values.size() };
        hid_t aspace = H5Screate_simple(1, dims, nullptr);
        hid_t attr   = H5Acreate2(obj, name, H5Native<T>::id(), aspace,
                                   H5P_DEFAULT, H5P_DEFAULT);
        check(attr, name);
        H5Awrite(attr, H5Native<T>::id(), values.data());
        H5Aclose(attr);
        H5Sclose(aspace);
    }

    // Write a 1-D dataset of the given native type into the parent group.
    template <typename T>
    static void write_1d(hid_t parent, const char* name,
                         const std::vector<T>& data)
    {
        hsize_t dims[1] = { data.size() };
        hid_t space = H5Screate_simple(1, dims, nullptr);
        check(space, "H5Screate_simple (1D)");
        hid_t dset = H5Dcreate2(parent, name, H5Native<T>::id(), space,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        check(dset, name);
        herr_t st = H5Dwrite(dset, H5Native<T>::id(), H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, data.data());
        check(st, "H5Dwrite (1D)");
        H5Dclose(dset);
        H5Sclose(space);
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
        hid_t space = H5Screate_simple(2, dims, nullptr);
        check(space, "H5Screate_simple (2D)");
        hid_t dset = H5Dcreate2(parent, name, H5Native<T>::id(), space,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        check(dset, name);
        herr_t st = H5Dwrite(dset, H5Native<T>::id(), H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, data.data());
        check(st, "H5Dwrite (2D)");
        H5Dclose(dset);
        H5Sclose(space);
    }

    VTKHDFWriter(const std::string& filepath, const std::string& type) {
        file_ = H5Fcreate(filepath.c_str(), H5F_ACC_TRUNC,
                          H5P_DEFAULT, H5P_DEFAULT);
        check(file_, "H5Fcreate");
        root_ = H5Gcreate2(file_, "VTKHDF", H5P_DEFAULT,
                           H5P_DEFAULT, H5P_DEFAULT);
        check(root_, "create /VTKHDF");
        write_string_attribute(root_, "Type", type);
        write_array_attribute<int64_t>(root_, "Version", {2, 0});
    }

public:
    virtual ~VTKHDFWriter() {
        H5Gclose(root_);
        H5Fclose(file_);
    }

    VTKHDFWriter(const VTKHDFWriter&) = delete;
    VTKHDFWriter& operator=(const VTKHDFWriter&) = delete;
};


// Writes a VTKHDF ImageData file (regular grid fields).
// PointData group is created on construction and closed on destruction.
class ImageDataWriter : public VTKHDFWriter {
    hid_t pdata_;

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

        pdata_ = H5Gcreate2(root_, "PointData", H5P_DEFAULT,
                            H5P_DEFAULT, H5P_DEFAULT);
        check(pdata_, "create PointData");
    }

    ~ImageDataWriter() {
        H5Gclose(pdata_);
    }

    // Write a [nz, ny, nx] scalar field from a contiguous row-major buffer.
    void WriteScalarField(const char* name, const double* data) {
        const hsize_t dims[3] = {(hsize_t)nz, (hsize_t)ny, (hsize_t)nx};
        hid_t sp = H5Screate_simple(3, dims, nullptr);
        check(sp, "H5Screate_simple (scalar field)");
        hid_t ds = H5Dcreate2(pdata_, name, H5T_NATIVE_DOUBLE, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        check(ds, name);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds);
        H5Sclose(sp);
    }

    // Write a [nz, ny, nx, components] vector field from a contiguous AoS buffer.
    void WriteVectorField(const char* name, const double* data, int components = 3) {
        const hsize_t dims[4] = {
            (hsize_t)nz, (hsize_t)ny, (hsize_t)nx, (hsize_t)components
        };
        hid_t sp = H5Screate_simple(4, dims, nullptr);
        check(sp, "H5Screate_simple (vector field)");
        hid_t ds = H5Dcreate2(pdata_, name, H5T_NATIVE_DOUBLE, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        check(ds, name);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds);
        H5Sclose(sp);
    }
};


// Writes a VTKHDF UnstructuredGrid file (arbitrary topology).
class UnstructuredGridWriter : public VTKHDFWriter {
public:
    UnstructuredGridWriter(const std::string& filepath)
        : VTKHDFWriter(filepath, "UnstructuredGrid")
    {}

    template <typename T>
    void Write1DToRoot(const char* name, const std::vector<T>& data) {
        write_1d(root_, name, data);
    }

    template <typename T>
    void Write2DToRoot(const char* name, const std::vector<T>& data,
                       int rows, int cols)
    {
        write_2d(root_, name, data,
                 static_cast<hsize_t>(rows), static_cast<hsize_t>(cols));
    }
};

#endif // LBM_AN_VTKHDFWRITER_H_
