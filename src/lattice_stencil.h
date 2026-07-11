#ifndef LBM_AN_LATTICE_STENCIL_H_
#define LBM_AN_LATTICE_STENCIL_H_

#include "params.h"

// D3Q15 lattice stencil: the discrete velocity set (e_i, w_i) and its
// symmetry maps (opposite/specular-reflection partners), shared by both the
// CPU (FluidFields / LbmSolver) and GPU (kernels.cu / DeviceFields) LBM
// paths. Plain constexpr C arrays, not std::array, so a single definition
// here can be used directly on the host. CUDA C++ does not allow a raw
// array of scalar type to be used inside a kernel unless it's used inside a
// constexpr __device__ or __host__ __device__ function — so the GPU path
// still needs its own __constant__ copy, populated once from this array via
// cudaMemcpyToSymbol in DeviceFields::Initialize().
namespace Lattice {

inline constexpr int ndir = 15;

/*!\brief  D3Q15 Structure
 *
 *             8_ _ _ _ _ _ _ _ _ _ _ _  7
 *            /|                        /|         z
 *          /  |                      /  |         ^      y
 *        /    |      5             /    |         |    /^
 *      /      |      .           /      |         |  /
 *    /        |      .         /        |         |/------> x
 * 9  _ _ _ _ _|_ _ _ . _ _2_10          |
 * |           |      .  .   |           |
 * |           |       .     |           |
 * |     3. . .| . .  0  .  .| . . 1     |
 * |           |    . .      |           |
 * |          12 _._ _._ _ _ | _ _ _ _ _ 11
 * |          / 4     .      |          /
 * |        /         .      |        /
 * |      /           .      |      /
 * |    /             6      |    /
 * |  /                      |  /
 * 13 _ _ _ _ _ _ _ _ _ _ _ _14
 *
 */
//                                       0  1  2   3   4   5   6  7   8   9  10, 11, 12, 13, 14
inline constexpr int ex[ndir] = {0, 1, 0, -1,  0,  0,  0, 1, -1, -1,  1,  1, -1, -1,  1};
inline constexpr int ey[ndir] = {0, 0, 1,  0, -1,  0,  0, 1,  1, -1, -1,  1,  1, -1, -1};
inline constexpr int ez[ndir] = {0, 0, 0,  0,  0,  1, -1, 1,  1,  1,  1, -1, -1, -1, -1};
inline constexpr double w[ndir] = {
    2.0/9, // 0
    1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, // 1-6
    1.0/72, 1.0/72, 1.0/72, 1.0/72, 1.0/72, 1.0/72, 1.0/72, 1.0/72  // 7-14
};

// Full reversal: opp[i] is the direction opposite to i (used in bounce-back).
//   0↔0  1↔3  2↔4  5↔6  7↔13  8↔14  9↔11  10↔12

//                                        0  1  2  3  4  5  6   7   8   9  10 11  12 13 14
inline constexpr int opp[ndir] = {0, 3, 4, 1, 2, 6, 5, 13, 14, 11, 12, 9, 10, 7, 8};

// Specular reflection partner for Z-walls (reflect ez, keep ey, ex):
//   specZ[i] = direction with (ex[i], ey[i], -ez[i])
//   0↔0  1↔1  2↔2  3↔3  4↔4  5↔6  7↔11  8↔12  9↔13  10↔14

//                                          0  1  2  3  4  5  6   7   8   9  10 11 12 13  14
inline constexpr int specZ[ndir] = {0, 1, 2, 3, 4, 6, 5, 11, 12, 13, 14, 7, 8, 9, 10};

// Specular reflection partner for Y-walls (reflect ey, keep ez, ex):
//   specY[i] = direction with (ex[i], -ey[i], ez[i])
//   0↔0  1↔1  2↔4  3↔3  5↔5  6↔6  7↔10  8↔9  11↔14  12↔13

//                                          0  1  2  3  4  5  6   7  8  9 10  11  12  13  14
inline constexpr int specY[ndir] = {0, 1, 4, 3, 2, 5, 6, 10, 9, 8, 7, 14, 13, 12, 11};

// Specular reflection partner for X-walls (reflect ey, keep ez, ex):
//   specX[i] = direction with (-ex[i], ey[i], ez[i])
//   0↔0  1↔3  2↔2  4↔4  5↔5  6↔6  7↔8  9↔10  11↔12  13↔14

//                                          0  1  2  3  4  5  6  7  8   9 10  11  12  13  14
inline constexpr int specX[ndir] = {0, 3, 2, 1, 4, 5, 6, 8, 7, 10, 9, 12, 11, 14, 13};

/*
 * Missing (incoming) directions per wall face:
 *
 *   ZHi (z = nz-1): {6, 11, 12, 13, 14}    ez < 0
 *   ZLo (z = 0)   : {5,  7,  8,  9, 10}    ex > 0
 *   YHi (y = ny-1): {4,  9, 10, 13, 14}    ey < 0
 *   YLo (y = 0)   : {2,  7,  8, 11, 12}    ey > 0
 *   XHi (x = nx-1): {3,  8,  9, 12, 13}    ex < 0
 *   XLo (x = 0)   : {1,  7, 10, 11, 14}    ex > 0
 */
inline constexpr int missingZHi[5] = {6, 11, 12, 13, 14};
inline constexpr int missingZLo[5] = {5,  7,  8,  9, 10};
inline constexpr int missingYHi[5] = {4,  9, 10, 13, 14};
inline constexpr int missingYLo[5] = {2,  7,  8, 11, 12};
inline constexpr int missingXHi[5] = {3,  8,  9, 12, 13};
inline constexpr int missingXLo[5] = {1,  7, 10, 11, 14};

} // namespace Lattice

#endif // LBM_AN_LATTICE_STENCIL_H_
