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

struct QDerivs { double dx, dy, dz, lap; };

template<QComp C>
inline CUDA_HOST_DEVICE constexpr double QComponent(const SymTrLessTensor5& t) {
    if constexpr (C == QComp::XX) return t.xx;
    else if constexpr (C == QComp::XY) return t.xy;
    else if constexpr (C == QComp::XZ) return t.xz;
    else if constexpr (C == QComp::YY) return t.yy;
    else return t.yz;
}

#endif // LBM_AN_QTENSOR_TYPES_H_
