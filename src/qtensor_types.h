#ifndef LBM_AN_QTENSOR_TYPES_H_
#define LBM_AN_QTENSOR_TYPES_H_

#ifndef CUDA_HOST_DEVICE
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif
#endif

enum class QComp { XX, XY, XZ, YY, YZ };

struct SymTrLessTensor5 { double xx, xy, xz, yy, yz; };

// Antisymmetric 3x3 tensor, stored as its three independent components.
// The diagonal vanishes and the lower triangle is minus the upper:
//   τ_yx = -τ_xy,  τ_zx = -τ_xz,  τ_zy = -τ_yz.
//
// Carries the torque-carrying part of the nematic stress, tau = QH - HQ.
// Distinct from SymTrLessTensor5 because that container has no representation
// for the lower triangle, which the stress divergence reads.
struct AntiSymTensor3 { double xy, xz, yz; };

// Central-difference first derivatives, the 7-point Laplacian, and the
// per-axis second differences d2a = Q(a+1) - 2Q(a) + Q(a-1).
//
// Mathematically lap == d2x + d2y + d2z; it is kept as its own member because
// QGradientAndLaplacian sums the six neighbours in one expression, and
// re-associating that sum would perturb existing results at round-off.
//
// The per-axis pieces are carried because they turn a centred first derivative
// into a one-sided one for free:
//
//     Q(a) - Q(a-1) = da - d2a/2      (backward)
//     Q(a+1) - Q(a) = da + d2a/2      (forward)
//
// so a one-sided derivative needs no extra neighbour fetches and no second pass
// through the boundary handler.
struct QDerivs { double dx, dy, dz, lap, d2x, d2y, d2z; };

template<QComp C>
inline CUDA_HOST_DEVICE constexpr double QComponent(const SymTrLessTensor5& t) {
    if constexpr (C == QComp::XX) return t.xx;
    else if constexpr (C == QComp::XY) return t.xy;
    else if constexpr (C == QComp::XZ) return t.xz;
    else if constexpr (C == QComp::YY) return t.yy;
    else return t.yz;
}

#endif // LBM_AN_QTENSOR_TYPES_H_
