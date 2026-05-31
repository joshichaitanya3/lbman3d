// h5_handle.hpp
//
// RAII wrappers around HDF5's C handles, plus a small type-trait that maps
// C++ scalar types to their HDF5 native equivalents.
//
// ----------------------------------------------------------------------------
// WHO SHOULD READ THIS FILE
// ----------------------------------------------------------------------------
// You probably don't need to. The user-facing I/O functions (write_1d,
// write_2d, write_string_attr, write_unstructured_grid, etc.) in vtkhdf_io.*
// hide everything here. This header exists to make the implementation of
// those functions safe — specifically, to guarantee that no HDF5 resource
// is ever leaked, even if an exception is thrown mid-write.
//
// You only need to touch this file if you want to:
//   - support a new C++ scalar type (e.g., int16_t) for read/write,
//   - add a new HDF5 resource kind (e.g., property lists with H5Pclose).
// Both cases are short and explained at the relevant spots below.
//
// ----------------------------------------------------------------------------
// THE PROBLEM THIS SOLVES
// ----------------------------------------------------------------------------
// HDF5's C API gives you opaque integer handles (hid_t) for files, groups,
// datasets, dataspaces, datatypes, and attributes. Each kind has its own
// matching close function: H5Fclose, H5Gclose, H5Dclose, H5Sclose, H5Tclose,
// H5Aclose. If you forget to call the close, or if an exception propagates
// past the close call, the resource leaks. After enough leaks HDF5 will start
// returning errors that look unrelated to the actual cause.
//
// The C++ idiom to prevent this is RAII (Resource Acquisition Is
// Initialization): wrap the handle in a small object whose destructor calls
// the appropriate close. The destructor runs automatically when the object
// goes out of scope — including during exception unwinding — so leaks become
// impossible by construction.
// ----------------------------------------------------------------------------

#pragma once
#include <hdf5.h>
#include <stdexcept>
#include <string>

// ----------------------------------------------------------------------------
// H5Handle: a generic RAII wrapper, one closer function per template
// instantiation.
//
// The template parameter `Closer` is a function pointer to one of HDF5's
// close functions (H5Fclose, H5Gclose, etc.). Because it's a template
// parameter (not a member variable), the compiler knows it at compile time
// and the destructor call has zero overhead compared to writing the close
// by hand.
//
// You don't normally use H5Handle directly. Use the aliases below
// (H5File, H5Group, ...).
// ----------------------------------------------------------------------------
template <herr_t (*Closer)(hid_t)>
class H5Handle {
public:
    // Default constructor: produces an empty (non-owning) handle.
    // Used when you want to declare a variable now and assign to it later.
    H5Handle() noexcept = default;

    // Take ownership of a freshly created HDF5 id.
    //
    // Throws std::runtime_error if `id` is negative, which is HDF5's way of
    // saying "the create/open call failed." This means downstream code can
    // assume any H5Handle that exists holds a valid id — no need to check
    // after every construction.
    //
    // `what` is just a label included in the error message (e.g. the name of
    // the dataset being created), so failures point at the right place.
    //
    // `explicit` prevents accidental implicit conversion from hid_t.
    explicit H5Handle(hid_t id, const char* what = "HDF5") : id_(id) {
        if (id_ < 0) throw std::runtime_error(std::string("HDF5 failed: ") + what);
    }

    // Destructor: closes the handle if we still own one.
    // Runs automatically at end of scope, on early return, AND during
    // exception unwinding. This is what makes leaks impossible.
    ~H5Handle() noexcept { if (id_ >= 0) Closer(id_); }

    // Copying is forbidden. If we allowed it, two handles would hold the
    // same id, and both destructors would call close on it -- a use-after-
    // close bug. `= delete` makes the compiler reject any attempt to copy.
    H5Handle(const H5Handle&)            = delete;
    H5Handle& operator=(const H5Handle&) = delete;

    // Moving IS allowed: ownership transfers from `o` to the new object,
    // and `o` is left empty (id_ = -1) so its eventual destructor does
    // nothing. This is how you return a handle from a factory function.
    H5Handle(H5Handle&& o) noexcept : id_(o.id_) { o.id_ = -1; }

    // Move assignment: same idea, but the left-hand side may already own a
    // handle, which must be closed before we overwrite it.
    H5Handle& operator=(H5Handle&& o) noexcept {
        if (this != &o) {                    // guard against `x = std::move(x);`
            if (id_ >= 0) Closer(id_);       // release what we currently hold
            id_ = o.id_;                     // take over o's id
            o.id_ = -1;                      // leave o empty
        }
        return *this;
    }

    // Read-only access to the underlying id.
    hid_t get() const noexcept { return id_; }

    // Implicit conversion to hid_t, so an H5Handle can be passed to any
    // HDF5 C function directly:
    //
    //     H5File f{ H5Fcreate(...) };
    //     H5Dwrite(dset, ..., f, ...);    // f converts to hid_t here
    //
    // Without this we'd write f.get() everywhere, which adds noise.
    operator hid_t() const noexcept { return id_; }

private:
    hid_t id_ = -1;     // -1 means "empty handle, owns nothing"
};

// ----------------------------------------------------------------------------
// Aliases for each HDF5 resource kind. Use these in I/O code instead of
// the raw H5Handle<...> template.
//
// To add a new kind (e.g., property lists), add a new line here:
//     using H5Plist = H5Handle<H5Pclose>;
// ----------------------------------------------------------------------------
using H5File      = H5Handle<H5Fclose>;
using H5Group     = H5Handle<H5Gclose>;
using H5Dataset   = H5Handle<H5Dclose>;
using H5Dataspace = H5Handle<H5Sclose>;
using H5Datatype  = H5Handle<H5Tclose>;
using H5Attribute = H5Handle<H5Aclose>;

// ----------------------------------------------------------------------------
// H5Native<T>: maps a C++ scalar type to its HDF5 native type id.
//
// HDF5's native type ids (H5T_NATIVE_DOUBLE, etc.) are not C++ types -- they
// are integer values defined by the HDF5 library. This trait lets templated
// read/write helpers ask "given that T = double, what's the right HDF5 id?"
// without each caller having to spell it out.
//
// The primary template has a static_assert that fires at compile time if
// someone tries to use a type we haven't mapped yet. The error message
// tells them exactly what to do.
//
// To add support for a new scalar type, add one line below:
//     template <> struct H5Native<int16_t> { static hid_t id() { return H5T_NATIVE_INT16; } };
// ----------------------------------------------------------------------------
template <class T>
struct H5Native {
    static_assert(sizeof(T) == 0,
        "No HDF5 native type mapping for T. "
        "Add a specialization of H5Native<T> in h5_handle.hpp.");
};
template <> struct H5Native<double>   { static hid_t id() { return H5T_NATIVE_DOUBLE; } };
template <> struct H5Native<float>    { static hid_t id() { return H5T_NATIVE_FLOAT;  } };
template <> struct H5Native<int32_t>  { static hid_t id() { return H5T_NATIVE_INT32;  } };
template <> struct H5Native<int64_t>  { static hid_t id() { return H5T_NATIVE_INT64;  } };
template <> struct H5Native<uint8_t>  { static hid_t id() { return H5T_NATIVE_UINT8;  } };
template <> struct H5Native<uint32_t> { static hid_t id() { return H5T_NATIVE_UINT32; } };
template <> struct H5Native<uint64_t> { static hid_t id() { return H5T_NATIVE_UINT64; } };
