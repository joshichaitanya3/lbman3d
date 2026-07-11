#ifndef LBM_AN_PHYSICS_HELPERS_H_
#define LBM_AN_PHYSICS_HELPERS_H_

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

#include <cassert>
#include "params.h"
#include "lattice_stencil.h"

using namespace Params;

struct Idx3 {
    int x, y, z;
};

inline CUDA_HOST_DEVICE bool InDomain(int x, int y, int z) {
    return (x >= 0) && (x < nx) && (y >= 0) && (y < ny) && (z >= 0) && (z < nz);
}

// Flat, periodic (x,y,z) -> offset for grid-sized fields; same layout on
// host and device since there's no direction index to complicate things.
// Every current caller already guarantees (x,y,z) is in-domain (via
// Grid::QXoff/QYoff/QZoff, which always clamp/wrap, or an InDomain guard
// before streaming destinations reach here) — the modulo below is at this
// point just insurance. The assert catches a caller that stops holding that
// invariant; it's compiled out under -DNDEBUG (see CMakeLists.txt's Release
// flags), so it costs nothing in a normal run.
inline CUDA_HOST_DEVICE int idx(int x, int y, int z) {
    assert(InDomain(x, y, z) && "idx(x,y,z): coordinates out of domain");
    return ((z + nz) % nz) * ny * nx + ((y + ny) % ny) * nx + ((x + nx) % nx);
}

// idx(x,y,z,i) has a DIFFERENT layout on host vs device:
//   - host:   i fastest-varying — all directions for one grid point are
//     contiguous, matching the CPU's per-point loop over ndir.
//   - device: i slowest-varying — all grid points for one direction are
//     contiguous, since a kernel step processes one direction across many
//     threads/blocks at once.
// __CUDA_ARCH__ is only defined during nvcc's device-code compilation pass
// of a __host__ __device__ function (undefined during its host pass), so a
// single function body branches on it rather than needing two overloads —
// nvcc treats __host__-tagged and __device__-tagged free functions with the
// same name/signature as a redefinition, not as distinct overloads.
inline CUDA_HOST_DEVICE int idx(int x, int y, int z, int i) {
    assert(InDomain(x, y, z) && "idx(x,y,z,i): coordinates out of domain");
    assert(i >= 0 && i < Lattice::ndir && "idx(x,y,z,i): direction index out of range");
#ifdef __CUDA_ARCH__
    return i * nz * ny * nx + ((z + nz) % nz) * ny * nx + ((y + ny) % ny) * nx + ((x + nx) % nx);
#else
    return (((z + nz) % nz) * ny * nx + ((y + ny) % ny) * nx + ((x + nx) % nx)) * Lattice::ndir + i;
#endif
}

struct Vec3 {
    double x, y, z;

    CUDA_HOST_DEVICE Vec3& operator+=(const Vec3& rhs) // compound assignment (does not need to be a member,
    {                           // but often is, to modify the private members)
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this; // return the result by reference
    }
    
    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_HOST_DEVICE Vec3 operator+(Vec3 lhs,        // passing lhs by value helps optimize chained a+b+c
                       const Vec3& rhs) // otherwise, both parameters may be const references
    {
        lhs += rhs; // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_HOST_DEVICE double Dot(const Vec3& rhs) const {
        return x*rhs.x + y*rhs.y + z*rhs.z;
    }
};


struct FeqForcing {
    double feq, forcing;
};

struct Moments {
    double rho;
    Vec3 u;
};

inline CUDA_HOST_DEVICE double Feq(Moments m, Vec3 e, double u2, double w_i) {
    
    double u_dot_e = m.u.Dot(e);
    return (w_i * m.rho * (1.0 + kCs2Inv * u_dot_e + khalfCs4Inv * u_dot_e * u_dot_e - khalfCs2Inv * u2));
}

inline CUDA_HOST_DEVICE FeqForcing ComputeFeqAndForcing(
    Moments m,
    double u2, // u-squared
    double uF, // Product of force and velocity
    Vec3 force,
    Vec3 e,
    double w_i
) {

    double ue = m.u.Dot(e);
    double eF  = e.Dot(force);

    double feq = (w_i * m.rho * (1.0 + 3.0 * ue + 4.5 * ue * ue - 1.5 * u2));
    double forcing_term = omega_forcing * w_i
        * (3.0 * eF - 3.0 * uF + 9.0 * ue * eF);
    
    return {feq, forcing_term};
}

inline CUDA_HOST_DEVICE Moments ComputeMoments(
    double* f,
    Idx3 point,
    Vec3 force,
    const int* ex,
    const int* ey,
    const int* ez
) {
    double rhop = 0.0;
    double uxp = 0.0;
    double uyp = 0.0;
    double uzp = 0.0;
    for (int i = 0; i < Lattice::ndir; ++i) {

        double fi = f[idx(point.x, point.y, point.z, i)];
        rhop += fi;
        uxp += ex[i] * fi;
        uyp += ey[i] * fi;
        uzp += ez[i] * fi;
    }
    uxp += 0.5 * force.x * DT;
    uyp += 0.5 * force.y * DT;
    uzp += 0.5 * force.z * DT;
    uxp /= rhop;
    uyp /= rhop;
    uzp /= rhop;
    Vec3 up{uxp, uyp, uzp};
    return {rhop, up};
}

#endif // LBM_AN_PHYSICS_HELPERS_H_
