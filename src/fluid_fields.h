#ifndef LBM_AN_FLUID_FIELDS_H_
#define LBM_AN_FLUID_FIELDS_H_

#include <array>
#include <vector>
#include <mdspan/mdspan.hpp>
#include "params.h"

// Owns all LBM state: distribution functions, macroscopic fields, body force.
// fx/fy/fz are the only write surface shared with QTensorSolver.
struct FluidFields {
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
    //                                           0  1  2   3   4   5   6  7   8   9  10, 11, 12, 13, 14
    static constexpr std::array<int, Params::ndir> ex = {0, 1, 0, -1,  0,  0,  0, 1, -1, -1,  1,  1, -1, -1,  1};
    static constexpr std::array<int, Params::ndir> ey = {0, 0, 1,  0, -1,  0,  0, 1,  1, -1, -1,  1,  1, -1, -1};
    static constexpr std::array<int, Params::ndir> ez = {0, 0, 0,  0,  0,  1, -1, 1,  1,  1,  1, -1, -1, -1, -1};
    static constexpr std::array<double, Params::ndir> w = {
        2.0/9, // 0
        1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, // 1-6
        1.0/72, 1.0/72, 1.0/72, 1.0/72, 1.0/72, 1.0/72, 1.0/72, 1.0/72  // 7-14
    };

    // Full reversal: opp[i] is the direction opposite to i (used in bounce-back).
    //   0↔0  1↔3  2↔4  5↔6  7↔13  8↔14  9↔11  10↔12
    //                                                     0  1  2  3  4  5  6   7   8   9  10 11  12 13 14
    
    static constexpr std::array<int, Params::ndir> opp  = {0, 3, 4, 1, 2, 6, 5, 13, 14, 11, 12, 9, 10, 7, 8};

    // Specular reflection partner for Z-walls (reflect ez, keep ey, ex):
    //   specZ[i] = direction with (ex[i], ey[i], -ez[i])
    //   0↔0  1↔1  2↔2  3↔3  4↔4  5↔6  7↔11  8↔12  9↔13  10↔14
    //                                                      0  1  2  3  4  5  6   7   8   9  10 11 12 13  14
    
    static constexpr std::array<int, Params::ndir> specZ = {0, 1, 2, 3, 4, 6, 5, 11, 12, 13, 14, 7, 8, 9, 10};

    // Specular reflection partner for Y-walls (reflect ey, keep ez, ex):
    //   specY[i] = direction with (ex[i], -ey[i], ez[i])
    //   0↔0  1↔1  2↔4  3↔3  5↔5  6↔6  7↔10  8↔9  11↔14  12↔13
    //                                              0  1  2  3  4  5  6   7  8  9 10  11  12  13  14
    
    static constexpr std::array<int, Params::ndir> specY = {0, 1, 4, 3, 2, 5, 6, 10, 9, 8, 7, 14, 13, 12, 11};

    // Specular reflection partner for X-walls (reflect ey, keep ez, ex):
    //   specX[i] = direction with (-ex[i], ey[i], ez[i])
    //   0↔0  1↔3  2↔2  4↔4  5↔5  6↔6  7↔8  9↔10  11↔12  13↔14
    //                                              0  1  2  3  4  5  6  7  8   9 10  11  12  13  14
    
    static constexpr std::array<int, Params::ndir> specX = {0, 3, 2, 1, 4, 5, 6, 8, 7, 10, 9, 12, 11, 14, 13};

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
    static constexpr std::array<int, 5> missingZHi = {6, 11, 12, 13, 14};
    static constexpr std::array<int, 5> missingZLo = {5,  7,  8,  9, 10};
    static constexpr std::array<int, 5> missingYHi = {4,  9, 10, 13, 14};
    static constexpr std::array<int, 5> missingYLo = {2,  7,  8, 11, 12};
    static constexpr std::array<int, 5> missingXHi = {3,  8,  9, 12, 13};
    static constexpr std::array<int, 5> missingXLo = {1,  7, 10, 11, 14};

     // Owned storage — all mdspan views below point into these
    std::vector<double> f_data, f_new_data, f_eq_data, forcing_data;
    std::vector<double> fx_data, fy_data, fz_data;
    std::vector<double> rho_data, ux_data, uy_data, uz_data;

    // Non-owning views (declared after _data so initializer-list order is safe)
    using ext3_t  = Kokkos::extents<int, Params::nx, Params::ny, Params::nz>;
    using ext4_t = Kokkos::extents<int, Params::nx, Params::ny, Params::nz, Params::ndir>;
    Kokkos::mdspan<double, ext3_t>  rho, ux, uy, uz;
    Kokkos::mdspan<double, ext3_t>  fx, fy, fz;
    Kokkos::mdspan<double, ext4_t> f, f_new, f_eq, forcing;

    FluidFields();
};

#endif // LBM_AN_FLUID_FIELDS_H_
