#ifndef LBM_AN_FLUID_FIELDS_H_
#define LBM_AN_FLUID_FIELDS_H_

#include <vector>
#include <mdspan/mdspan.hpp>
#include "params.h"

// Owns all LBM state: distribution functions, macroscopic fields, body force.
// fx/fy/fz are the only write surface shared with QTensorSolver.
struct FluidFields {
     // Owned storage — all mdspan views below point into these
    std::vector<double> f_data, f_new_data;
    std::vector<double> fx_data, fy_data, fz_data;
    std::vector<double> rho_data, ux_data, uy_data, uz_data;

    // Non-owning views. MUST stay declared after every std::vector member
    // above: DeviceFields::Initialize/CopyToHost (device_fields.cu, compiled
    // by nvcc) take a FluidFields& and read only these vectors, never the
    // mdspan views. nvcc does not apply the empty-base/no-unique-address
    // compression g++ applies to mdspan's (stateless) mapping/accessor
    // members, so nvcc and g++ compute different sizeof(FluidFields) — but a
    // member's offset only depends on what's declared BEFORE it, so the
    // vectors' offsets still agree across both compilers as long as nothing
    // that could disagree in size is declared earlier. Moving a vector below
    // an mdspan member, or having nvcc-compiled code touch an mdspan member
    // directly, would silently reintroduce the cross-compiler memory
    // corruption this ordering exists to avoid (see device_fields.h).
    using ext3_t  = Kokkos::extents<int, Params::nz, Params::ny, Params::nx>;
    using ext4_t = Kokkos::extents<int, Params::nz, Params::ny, Params::nx, Params::ndir>;
    Kokkos::mdspan<double, ext3_t>  rho, ux, uy, uz;
    Kokkos::mdspan<double, ext3_t>  fx, fy, fz;
    Kokkos::mdspan<double, ext4_t> f, f_new;

    FluidFields();

    void SwapFandFnew() {
        SwapF(f_data, f_new_data, f, f_new);
    }

private:
    static void SwapF(std::vector<double>& a_data, std::vector<double>& b_data,
                      Kokkos::mdspan<double, ext4_t>& a_view,
                      Kokkos::mdspan<double, ext4_t>& b_view) {
        std::swap(a_data, b_data);
        a_view = Kokkos::mdspan<double, ext4_t>(a_data.data(), Params::nz, Params::ny, Params::nx, Params::ndir);
        b_view = Kokkos::mdspan<double, ext4_t>(b_data.data(), Params::nz, Params::ny, Params::nx, Params::ndir);
    }
};

#endif // LBM_AN_FLUID_FIELDS_H_
