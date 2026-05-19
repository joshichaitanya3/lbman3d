#include "params.h"
#include "boundary.h"

#include <cmath>
#include <random>
#include <ranges>

using namespace Params;

template<typename BC>
QTensorSolver<BC>::QTensorSolver(Grid<BC> grid) : grid_(std::move(grid)) {}

template<typename BC>
void QTensorSolver<BC>::Initialize(QTensorFields& qf) const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> noise_dist(-NOISE, NOISE);
    for (int x : std::views::iota(0, nx)) {
        for (int y : std::views::iota(0, ny)) {
            for (int z : std::views::iota(0, nz)) {
                qf.qxx[x, y, z] = 0.66 + noise_dist(gen);
                qf.qxy[x, y, z] = noise_dist(gen);
                qf.qxz[x, y, z] = noise_dist(gen);
                qf.qyy[x, y, z] = -0.33 + noise_dist(gen);
                qf.qyz[x, y, z] = noise_dist(gen);
            }
        }
    }
}

template<typename BC>
void QTensorSolver<BC>::FiniteDifferenceStep(QTensorFields& qf, const FluidFields& ff) const {
    auto compute_cell = [&](int x, int y, int z, int xm, int xp, int ym, int yp, int zm, int zp) {

        // Fields

        const double Qxx = qf.qxx[x, y, z];
        const double Qxy = qf.qxy[x, y, z];
        const double Qxz = qf.qxz[x, y, z];
        const double Qyy = qf.qyy[x, y, z];
        const double Qyz = qf.qyz[x, y, z];

        const double ux = ff.ux[x, y, z];
        const double uy = ff.uy[x, y, z];
        const double uz = ff.uz[x, y, z];

        // Polynomials
        const double TrQ2 = 2.0*(Qxx*Qxx + Qyy*Qyy+ Qxx*Qyy + Qxy*Qxy +Qxz*Qxz +Qyz*Qyz);
        
        // Q2 = Q_ik * Q_kj - 1/3 TrQ2 \delta_ij:
        const double kone_thirds = 1.0/3.0;
        const double Q2_xx = Qxx*Qxx + Qxy*Qxy + Qxz*Qxz - kone_thirds * TrQ2;
        const double Q2_xy = Qxx*Qxy + Qxy*Qyy + Qxz*Qyz;
        const double Q2_xz = Qxy*Qyz - Qxz*Qyy;
        const double Q2_yy = Qxy*Qxy + Qyy*Qyy + Qyz*Qyz - kone_thirds * TrQ2;
        const double Q2_yz = Qxy*Qxz - Qyz*Qxx;
        
        const double TrQ3 = -3.0*(Qxx*Qxx*Qyy - Qxy*Qxy*Qyy + Qxz*Qxz*Qyy - 2*Qxy*Qxz*Qyz+Qxx * (-Qxy*Qxy + Qyy*Qyy + Qyz*Qyz)); 
        
        // First-order derivatives 

        // Velocity gradient tensor (central differences; uses the same Q stencil offsets
        // since HandleBoundaries has already set correct wall velocities from the
        // previous LBM step before FiniteDifferenceStep is called)
        const double uxx = (ff.ux[xp,y,z] - ff.ux[xm,y,z]) / 2.0;
        const double uxy = (ff.ux[x,yp,z] - ff.ux[x,ym,z]) / 2.0;
        const double uxz = (ff.ux[x,y,zp] - ff.ux[x,y,zm]) / 2.0;

        const double uyx = (ff.uy[xp,y,z] - ff.uy[xm,y,z]) / 2.0;
        const double uyy = (ff.uy[x,yp,z] - ff.uy[x,ym,z]) / 2.0;
        const double uyz = (ff.uy[x,y,zp] - ff.uy[x,y,zm]) / 2.0;

        const double uzx = (ff.uz[xp,y,z] - ff.uz[xm,y,z]) / 2.0;
        const double uzy = (ff.uz[x,yp,z] - ff.uz[x,ym,z]) / 2.0;
        const double uzz = - (uxx + uyy); // from incompressibility
        
        /*
            * Exx:
            * Derivative(ux(x, y, z), x)
            * Exy:
            * Derivative(ux(x, y, z), y)/2 + Derivative(uy(x, y, z), x)/2
            * Exz:
            * Derivative(ux(x, y, z), z)/2 + Derivative(uz(x, y, z), x)/2
            * Eyy:
            * Derivative(uy(x, y, z), y)
            * Eyz:
            * Derivative(uy(x, y, z), z)/2 + Derivative(uz(x, y, z), y)/2
            */

        const double Exx = uxx;
        const double Exy = 0.5 * (uxy + uyx);
        const double Exz = 0.5 * (uxz + uzx);
        const double Eyy = uyy;
        const double Eyz = 0.5 * (uyz + uzy);

        /*
        Omegaxy = 1/2(Uxy - Uyx)
        */
        const double Wxy = 0.5 * (uxy - uyx);
        const double Wxz = 0.5 * (uxz - uzx);
        const double Wyx = -Wxy;
        const double Wyz = 0.5 * (uyz - uzy);
        const double Wzx = -Wxz;
        const double Wzy = -Wyz;

        // Q-tensor
        const double Qxxx = (qf.qxx[xp, y, z] - qf.qxx[xm, y, z]) / 2.0;
        const double Qxyx = (qf.qxy[xp, y, z] - qf.qxy[xm, y, z]) / 2.0;
        const double Qxzx = (qf.qxz[xp, y, z] - qf.qxz[xm, y, z]) / 2.0;
        const double Qyyx = (qf.qyy[xp, y, z] - qf.qyy[xm, y, z]) / 2.0;
        const double Qyzx = (qf.qyz[xp, y, z] - qf.qyz[xm, y, z]) / 2.0;

        const double Qxxy = (qf.qxx[x, yp, z] - qf.qxx[x, ym, z]) / 2.0;
        const double Qxyy = (qf.qxy[x, yp, z] - qf.qxy[x, ym, z]) / 2.0;
        const double Qxzy = (qf.qxz[x, yp, z] - qf.qxz[x, ym, z]) / 2.0;
        const double Qyyy = (qf.qyy[x, yp, z] - qf.qyy[x, ym, z]) / 2.0;
        const double Qyzy = (qf.qyz[x, yp, z] - qf.qyz[x, ym, z]) / 2.0;

        const double Qxxz = (qf.qxx[x, y, zp] - qf.qxx[x, y, zm]) / 2.0;
        const double Qxyz = (qf.qxy[x, y, zp] - qf.qxy[x, y, zm]) / 2.0;
        const double Qxzz = (qf.qxz[x, y, zp] - qf.qxz[x, y, zm]) / 2.0;
        const double Qyyz = (qf.qyy[x, y, zp] - qf.qyy[x, y, zm]) / 2.0;
        const double Qyzz = (qf.qyz[x, y, zp] - qf.qyz[x, y, zm]) / 2.0;

        // Laplacian (seven-point stencil)
        const double lap_Qxx = qf.qxx[xp,y,z] + qf.qxx[xm,y,z] + qf.qxx[x,yp,z] + qf.qxx[x,ym,z] + qf.qxx[x,y,zp] + qf.qxx[x,y,zm] - 6.0*Qxx;
        const double lap_Qxy = qf.qxy[xp,y,z] + qf.qxy[xm,y,z] + qf.qxy[x,yp,z] + qf.qxy[x,ym,z] + qf.qxy[x,y,zp] + qf.qxy[x,y,zm] - 6.0*Qxy;
        const double lap_Qxz = qf.qxz[xp,y,z] + qf.qxz[xm,y,z] + qf.qxz[x,yp,z] + qf.qxz[x,ym,z] + qf.qxz[x,y,zp] + qf.qxz[x,y,zm] - 6.0*Qxz;
        const double lap_Qyy = qf.qyy[xp,y,z] + qf.qyy[xm,y,z] + qf.qyy[x,yp,z] + qf.qyy[x,ym,z] + qf.qyy[x,y,zp] + qf.qyy[x,y,zm] - 6.0*Qyy;
        const double lap_Qyz = qf.qyz[xp,y,z] + qf.qyz[xm,y,z] + qf.qyz[x,yp,z] + qf.qyz[x,ym,z] + qf.qyz[x,y,zp] + qf.qyz[x,y,zm] - 6.0*Qyz;
        
        // Advection: -u · ∇Q
        const double adv_xx = -(ux * Qxxx + uy * Qxxy + uz * Qxxz);
        const double adv_xy = -(ux * Qxyx + uy * Qxyy + uz * Qxyz);
        const double adv_xz = -(ux * Qxzx + uy * Qxzy + uz * Qxzz);
        const double adv_yy = -(ux * Qyyx + uy * Qyyy + uz * Qyyz);
        const double adv_yz = -(ux * Qyzx + uy * Qyzy + uz * Qyzz);
        
        // ##############################################################################
        // #   corotation       [(Omega Q - Q Omega)_ij]
        // ##############################################################################
        
        const double cor_xx = 2.0 * (Qxy * Wxy + Qxz * Wxz);
        const double cor_xy = -Qxx * Wxy + Qxz * Wyz + Qyy * Wxy + Qyz * Wxz;
        const double cor_xz = -2.0 * Qxx * Wxz - Qxy * Wyz - Qyy * Wxz + Qyz * Wxy;
        const double cor_yy = 2.0 * (-Qxy * Wxy + Qyz * Wyz);
        const double cor_yz = -Qxx * Wyz -Qxy * Wxz -Qxz * Wxy - 2.0 * Qyy * Wyz;
        
        /* ##############################################################################
        // #   higher order order flow alignment   lambda [(E Q + Q E)_ij]
        1->xx, 2->xy, 3->xz, 4->yy, 5->yz
        QE_xx = e_1 Q_1 + e_2 Q_2 + e_3 Q_3
        QE_xy = e_2 Q_1 + e_4 Q_2 + e_5 Q_3
        QE_xz = e_3 Q_1 + e_5 Q_2 + (-e_1 - e_4) Q_3
        QE_yy = e_2 Q_2 + e_4 Q_4 + e_5 Q_5
        QE_yz = e_3 Q_2 + e_5 Q_4 + (-e_1 - e_4) Q_5
        
        Q:E (trace): (2 e_1 + e_4) Q_1 + 2 e_2 Q_2 + 2 e_3 Q_3 + e_1 Q_4 + 2 e_4 Q_4 + 2 e_5 Q_5
        // ##############################################################################
        */
       const double ktwo_thirds = 2.0/3.0;
       
       const double tr_QE = (2.0 * Exx + Eyy) * Qxx + (2.0 * Exy) * Qxy + (2.0 * Exz) * Qxz + (Exx + 2.0 * Eyy) * Qyy + 2.0 * Eyz * Qyz;

       const double aln2_xx = 2.0 * (Exx * Qxx + Exy * Qxy + Exz * Qxz) - ktwo_thirds * tr_QE;
       const double aln2_xy = Exy * Qxx + Eyy * Qxy + Eyz * Qxz
                            + Qxy * Exx + Qyy * Exy + Qyz * Exz;
       const double aln2_xz = Exz * Qxx + Eyz * Qxy + (-Exx - Eyy) * Qxz
                            + Qxz * Exx + Qyz * Exy + (-Qxx - Qyy) * Exz;
       
       const double aln2_yy = 2.0 * (Exy * Qxy + Eyy * Qyy + Eyz * Qyz) - ktwo_thirds * tr_QE;
       const double aln2_yz = Exz * Qxy + Eyz * Qyy + (-Exx - Eyy) * Qyz
                            + Qxz * Exy + Qyz * Eyy + (-Qxx - Qyy) * Eyz;
       
       // Molecular field H
       const double H_xx = L * lap_Qxx - A * Qxx - B * Q2_xx - C * Qxx * TrQ2;
       const double H_xy = L * lap_Qxy - A * Qxy - B * Q2_xy - C * Qxy * TrQ2;
       const double H_xz = L * lap_Qxz - A * Qxz - B * Q2_xz - C * Qxz * TrQ2;
       const double H_yy = L * lap_Qyy - A * Qyy - B * Q2_yy - C * Qyy * TrQ2;
       const double H_yz = L * lap_Qyz - A * Qyz - B * Q2_yz - C * Qyz * TrQ2;
       
       
       qf.qxx_new[x, y, z] = Qxx + DT*(adv_xx + cor_xx + LAMBDA * (ktwo_thirds * Exx + aln2_xx) + GAMMA * H_xx);
       qf.qxy_new[x, y, z] = Qxy + DT*(adv_xy + cor_xy + LAMBDA * (ktwo_thirds * Exy + aln2_xy) + GAMMA * H_xy);
       qf.qxz_new[x, y, z] = Qxz + DT*(adv_xz + cor_xz + LAMBDA * (ktwo_thirds * Exz + aln2_xz) + GAMMA * H_xz);
       qf.qyy_new[x, y, z] = Qyy + DT*(adv_yy + cor_yy + LAMBDA * (ktwo_thirds * Eyy + aln2_yy) + GAMMA * H_yy);
       qf.qyz_new[x, y, z] = Qyz + DT*(adv_yz + cor_yz + LAMBDA * (ktwo_thirds * Eyz + aln2_yz) + GAMMA * H_yz);
    };
    
    #pragma omp parallel for num_threads(numprocs) schedule(static)
    for (int x = 1; x < nx - 1; ++x) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int z = 1; z < nz - 1; ++z) {

                compute_cell(x, y, z, x-1, x+1, y-1, y+1, z-1, z+1);
            }
        }

    }

    // Boundary rows/columns: resolve ghost nodes through Q-stencil offsets.

    // First, the 6 faces
    for (int x = 1; x < nx-1; ++x) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int z : {0, nz-1}) {
                compute_cell(x, y, z, x-1, x+1, y-1, y+1, QZoff(z,-1), QZoff(z,1));
            }
        }
    }

    for (int x = 1; x < nx-1; ++x) {
        for (int y : {0, ny-1}) {
            for (int z = 1; z < nz - 1; ++z) {
                compute_cell(x, y, z, x-1, x+1, QYoff(y,-1), QYoff(y,1), z-1, z+1);
            }
        }
    }

    for (int x : {0, nx-1}) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int z = 1; z < nz - 1; ++z) {
                compute_cell(x, y, z, QXoff(x,-1), QXoff(x,1), y-1, y+1, z-1, z+1);
            }
        }
    }
    // Now, the 12 edges and 8 corners together
    for (int x = 0; x < nx; ++x) {
        compute_cell(x, 0,       0, QXoff(x,-1), QXoff(x,1), QYoff(0,-1),             1, QZoff(0, -1),              1);
        compute_cell(x, ny-1,    0, QXoff(x,-1), QXoff(x,1),        ny-2, QYoff(ny-1,1), QZoff(0, -1),              1);
        compute_cell(x, 0,    nz-1, QXoff(x,-1), QXoff(x,1), QYoff(0,-1),             1,         nz-2, QZoff(nz-1, 1));
        compute_cell(x, ny-1, nz-1, QXoff(x,-1), QXoff(x,1),        ny-2, QYoff(ny-1,1),         nz-2, QZoff(nz-1, 1));
    }

    for (int y = 0; y < ny; ++y) {
        compute_cell(   0, y,    0, QXoff(0,-1),             1, QYoff(y,-1), QYoff(y,1), QZoff(0, -1),              1);
        compute_cell(nx-1, y,    0,        nx-2, QXoff(nx-1,1), QYoff(y,-1), QYoff(y,1), QZoff(0, -1),              1);
        compute_cell(   0, y, nz-1, QXoff(0,-1),             1, QYoff(y,-1), QYoff(y,1),         nz-2, QZoff(nz-1, 1));
        compute_cell(nx-1, y, nz-1,        nx-2, QXoff(nx-1,1), QYoff(y,-1), QYoff(y,1),         nz-2, QZoff(nz-1, 1));
    }

    for (int z = 0; z < nz; ++z) {
        compute_cell(   0,    0, z, QXoff(0,-1),             1, QYoff(0, -1),              1, QZoff(z,-1), QZoff(z,1));
        compute_cell(nx-1,    0, z,        nx-2, QXoff(nx-1,1), QYoff(0, -1),              1, QZoff(z,-1), QZoff(z,1));
        compute_cell(   0, ny-1, z, QXoff(0,-1),             1,         ny-2, QYoff(ny-1, 1), QZoff(z,-1), QZoff(z,1));
        compute_cell(nx-1, ny-1, z,        nx-2, QXoff(nx-1,1),         ny-2, QYoff(ny-1, 1), QZoff(z,-1), QZoff(z,1));
    }


    HandleQBoundary(qf);
    UpdateQnewWithQ(qf);
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-wall Q-tensor boundary handlers
//
// Neumann/Periodic: no action — the stencil clamping/wrapping in QXoff/QYoff
// already enforces ∂Q/∂n = 0 at the correct wall position.
//
// Anchoring<S,θ, phi>: overwrite q_new at the wall with the
// prescribed strong-anchoring value after the FD step.  On the next step, the
// Laplacian of interior cells adjacent to the wall reads this fixed value,
// giving the correct Dirichlet influence.
// ─────────────────────────────────────────────────────────────────────────────

template<typename BC>
template<typename WallSpec>
void QTensorSolver<BC>::HandleQWallZLo(QTensorFields& qf) const {
    using Q = typename WallSpec::QBC;
    if constexpr (is_anchoring_v<Q>) {
        const double cos_phi = std::cos(Q::phi);
        const double sin_phi = std::sin(Q::phi);
        const double sin_th_cos_th = 0.5 * std::sin(2.0 * Q::theta);
        const double sin_phi_cos_phi = 0.5 * std::sin(2.0 * Q::phi);
        const double sin_sq_th = 0.5 * (1.0 - std::cos(2.0 * Q::theta));
        const double sin_sq_phi   = 0.5 * (1.0 - std::cos(2.0 * Q::phi));
        const double cos_sq_phi   = 1.0 - sin_sq_phi;

        const double S = Q::s;
        const double qxx_bc = S * (cos_sq_phi * sin_sq_th - 1.0/3.0);
        const double qxy_bc = S * (sin_phi_cos_phi * sin_sq_th);
        const double qxz_bc = S * (cos_phi * sin_th_cos_th);
        const double qyy_bc = S * (sin_sq_phi * sin_sq_th - 1.0/3.0);
        const double qyz_bc = S * (sin_phi * sin_th_cos_th);

        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                qf.qxx_new[x, y, 0] = qxx_bc;
                qf.qxy_new[x, y, 0] = qxy_bc;
                qf.qxz_new[x, y, 0] = qxz_bc;
                qf.qyy_new[x, y, 0] = qyy_bc;
                qf.qyz_new[x, y, 0] = qyz_bc;
            }
        }
    }
}

template<typename BC>
template<typename WallSpec>
void QTensorSolver<BC>::HandleQWallZHi(QTensorFields& qf) const {
    using Q = typename WallSpec::QBC;
    if constexpr (is_anchoring_v<Q>) {
        const double cos_phi = std::cos(Q::phi);
        const double sin_phi = std::sin(Q::phi);
        const double sin_th_cos_th = 0.5 * std::sin(2.0 * Q::theta);
        const double sin_phi_cos_phi = 0.5 * std::sin(2.0 * Q::phi);
        const double sin_sq_th = 0.5 * (1.0 - std::cos(2.0 * Q::theta));
        const double sin_sq_phi   = 0.5 * (1.0 - std::cos(2.0 * Q::phi));
        const double cos_sq_phi   = 1.0 - sin_sq_phi;

        const double S = Q::s;
        const double qxx_bc = S * (cos_sq_phi * sin_sq_th - 1.0/3.0);
        const double qxy_bc = S * (sin_phi_cos_phi * sin_sq_th);
        const double qxz_bc = S * (cos_phi * sin_th_cos_th);
        const double qyy_bc = S * (sin_sq_phi * sin_sq_th - 1.0/3.0);
        const double qyz_bc = S * (sin_phi * sin_th_cos_th);

        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                qf.qxx_new[x, y, nz-1] = qxx_bc;
                qf.qxy_new[x, y, nz-1] = qxy_bc;
                qf.qxz_new[x, y, nz-1] = qxz_bc;
                qf.qyy_new[x, y, nz-1] = qyy_bc;
                qf.qyz_new[x, y, nz-1] = qyz_bc;
            }
        }
    }
}


template<typename BC>
template<typename WallSpec>
void QTensorSolver<BC>::HandleQWallYLo(QTensorFields& qf) const {
    using Q = typename WallSpec::QBC;
    if constexpr (is_anchoring_v<Q>) {
        const double cos_phi = std::cos(Q::phi);
        const double sin_phi = std::sin(Q::phi);
        const double sin_th_cos_th = 0.5 * std::sin(2.0 * Q::theta);
        const double sin_phi_cos_phi = 0.5 * std::sin(2.0 * Q::phi);
        const double sin_sq_th = 0.5 * (1.0 - std::cos(2.0 * Q::theta));
        const double sin_sq_phi   = 0.5 * (1.0 - std::cos(2.0 * Q::phi));
        const double cos_sq_phi   = 1.0 - sin_sq_phi;

        const double S = Q::s;
        const double qxx_bc = S * (cos_sq_phi * sin_sq_th - 1.0/3.0);
        const double qxy_bc = S * (sin_phi_cos_phi * sin_sq_th);
        const double qxz_bc = S * (cos_phi * sin_th_cos_th);
        const double qyy_bc = S * (sin_sq_phi * sin_sq_th - 1.0/3.0);
        const double qyz_bc = S * (sin_phi * sin_th_cos_th);

        for (int x = 0; x < nx; ++x) {
            for (int z = 0; z < nz; ++z) {
                qf.qxx_new[x, 0, z] = qxx_bc;
                qf.qxy_new[x, 0, z] = qxy_bc;
                qf.qxz_new[x, 0, z] = qxz_bc;
                qf.qyy_new[x, 0, z] = qyy_bc;
                qf.qyz_new[x, 0, z] = qyz_bc;
            }
        }
    }
}

template<typename BC>
template<typename WallSpec>
void QTensorSolver<BC>::HandleQWallYHi(QTensorFields& qf) const {
    using Q = typename WallSpec::QBC;
    if constexpr (is_anchoring_v<Q>) {
        const double cos_phi = std::cos(Q::phi);
        const double sin_phi = std::sin(Q::phi);
        const double sin_th_cos_th = 0.5 * std::sin(2.0 * Q::theta);
        const double sin_phi_cos_phi = 0.5 * std::sin(2.0 * Q::phi);
        const double sin_sq_th = 0.5 * (1.0 - std::cos(2.0 * Q::theta));
        const double sin_sq_phi   = 0.5 * (1.0 - std::cos(2.0 * Q::phi));
        const double cos_sq_phi   = 1.0 - sin_sq_phi;

        const double S = Q::s;
        const double qxx_bc = S * (cos_sq_phi * sin_sq_th - 1.0/3.0);
        const double qxy_bc = S * (sin_phi_cos_phi * sin_sq_th);
        const double qxz_bc = S * (cos_phi * sin_th_cos_th);
        const double qyy_bc = S * (sin_sq_phi * sin_sq_th - 1.0/3.0);
        const double qyz_bc = S * (sin_phi * sin_th_cos_th);

        for (int x = 0; x < nx; ++x) {
            for (int z = 0; z < nz; ++z) {
                qf.qxx_new[x, ny-1, z] = qxx_bc;
                qf.qxy_new[x, ny-1, z] = qxy_bc;
                qf.qxz_new[x, ny-1, z] = qxz_bc;
                qf.qyy_new[x, ny-1, z] = qyy_bc;
                qf.qyz_new[x, ny-1, z] = qyz_bc;
            }
        }
    }
}


template<typename BC>
template<typename WallSpec>
void QTensorSolver<BC>::HandleQWallXLo(QTensorFields& qf) const {
    using Q = typename WallSpec::QBC;
    if constexpr (is_anchoring_v<Q>) {
        const double cos_phi = std::cos(Q::phi);
        const double sin_phi = std::sin(Q::phi);
        const double sin_th_cos_th = 0.5 * std::sin(2.0 * Q::theta);
        const double sin_phi_cos_phi = 0.5 * std::sin(2.0 * Q::phi);
        const double sin_sq_th = 0.5 * (1.0 - std::cos(2.0 * Q::theta));
        const double sin_sq_phi   = 0.5 * (1.0 - std::cos(2.0 * Q::phi));
        const double cos_sq_phi   = 1.0 - sin_sq_phi;

        const double S = Q::s;
        const double qxx_bc = S * (cos_sq_phi * sin_sq_th - 1.0/3.0);
        const double qxy_bc = S * (sin_phi_cos_phi * sin_sq_th);
        const double qxz_bc = S * (cos_phi * sin_th_cos_th);
        const double qyy_bc = S * (sin_sq_phi * sin_sq_th - 1.0/3.0);
        const double qyz_bc = S * (sin_phi * sin_th_cos_th);

        for (int y = 0; y < ny; ++y) {
            for (int z = 0; z < nz; ++z) {
                qf.qxx_new[0, y, z] = qxx_bc;
                qf.qxy_new[0, y, z] = qxy_bc;
                qf.qxz_new[0, y, z] = qxz_bc;
                qf.qyy_new[0, y, z] = qyy_bc;
                qf.qyz_new[0, y, z] = qyz_bc;
            }
        }
    }
}

template<typename BC>
template<typename WallSpec>
void QTensorSolver<BC>::HandleQWallXHi(QTensorFields& qf) const {
    using Q = typename WallSpec::QBC;
    if constexpr (is_anchoring_v<Q>) {
        const double cos_phi = std::cos(Q::phi);
        const double sin_phi = std::sin(Q::phi);
        const double sin_th_cos_th = 0.5 * std::sin(2.0 * Q::theta);
        const double sin_phi_cos_phi = 0.5 * std::sin(2.0 * Q::phi);
        const double sin_sq_th = 0.5 * (1.0 - std::cos(2.0 * Q::theta));
        const double sin_sq_phi   = 0.5 * (1.0 - std::cos(2.0 * Q::phi));
        const double cos_sq_phi   = 1.0 - sin_sq_phi;

        const double S = Q::s;
        const double qxx_bc = S * (cos_sq_phi * sin_sq_th - 1.0/3.0);
        const double qxy_bc = S * (sin_phi_cos_phi * sin_sq_th);
        const double qxz_bc = S * (cos_phi * sin_th_cos_th);
        const double qyy_bc = S * (sin_sq_phi * sin_sq_th - 1.0/3.0);
        const double qyz_bc = S * (sin_phi * sin_th_cos_th);

        for (int y = 0; y < ny; ++y) {
            for (int z = 0; z < nz; ++z) {
                qf.qxx_new[nx-1, y, z] = qxx_bc;
                qf.qxy_new[nx-1, y, z] = qxy_bc;
                qf.qxz_new[nx-1, y, z] = qxz_bc;
                qf.qyy_new[nx-1, y, z] = qyy_bc;
                qf.qyz_new[nx-1, y, z] = qyz_bc;
            }
        }
    }
}


template<typename BC>
void QTensorSolver<BC>::HandleQBoundary(QTensorFields& qf) const {
    HandleQWallXLo<typename BC::XLo>(qf);
    HandleQWallXHi<typename BC::XHi>(qf);
    HandleQWallYLo<typename BC::YLo>(qf);
    HandleQWallYHi<typename BC::YHi>(qf);
    HandleQWallZLo<typename BC::ZLo>(qf);
    HandleQWallZHi<typename BC::ZHi>(qf);
}

template<typename BC>
void QTensorSolver<BC>::UpdateQnewWithQ(QTensorFields& qf) const {
    qf.SwapWithNew();
}

template<typename BC>
void QTensorSolver<BC>::ComputeActiveBodyForce(FluidFields& ff, const QTensorFields& qf) const {
    auto compute_cell = [&](int x, int y, int z, int xm, int xp, int ym, int yp, int zm, int zp) {
        ff.fx[x, y, z] = -ALPHA*((qf.qxx[xp, y, z] - qf.qxx[xm, y, z])/2.0
                               + (qf.qxy[x, yp, z] - qf.qxy[x, ym, z])/2.0
                               + (qf.qxz[x, y, zp] - qf.qxz[x, y, zm])/2.0)
                         -MU * ff.ux[x, y, z];

        ff.fy[x, y, z] = -ALPHA*((qf.qxy[xp, y, z] - qf.qxy[xm, y, z])/2.0
                               + (qf.qyy[x, yp, z] - qf.qyy[x, ym, z])/2.0
                               + (qf.qyz[x, y, zp] - qf.qyz[x, y, zm])/2.0)
                         -MU * ff.uy[x, y, z];

        ff.fz[x, y, z] = -ALPHA*((qf.qxz[xp, y, z] - qf.qxz[xm, y, z])/2.0
                               + (qf.qyz[x, yp, z] - qf.qyz[x, ym, z])/2.0
                               - (qf.qxx[x, y, zp] - qf.qxx[x, y, zm])/2.0
                               - (qf.qyy[x, y, zp] - qf.qyy[x, y, zm])/2.0) // Since Qzz = -(Qxx + Qyy)
                         -MU * ff.uz[x, y, z];
    };

    #pragma omp parallel for default(shared) num_threads(numprocs) schedule(static)
    for (int x = 1; x < nx - 1; ++x)
        for (int y = 1; y < ny - 1; ++y)
            for (int z = 1; z < nz - 1; ++z)
                compute_cell(x, y, z, x-1, x+1, y-1, y+1, z-1, z+1);

        // Boundary rows/columns: resolve ghost nodes through Q-stencil offsets.

    // First, the 6 faces
    for (int x = 1; x < nx-1; ++x) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int z : {0, nz-1}) {
                compute_cell(x, y, z, x-1, x+1, y-1, y+1, QZoff(z,-1), QZoff(z,1));
            }
        }
    }

    for (int x = 1; x < nx-1; ++x) {
        for (int y : {0, ny-1}) {
            for (int z = 1; z < nz - 1; ++z) {
                compute_cell(x, y, z, x-1, x+1, QYoff(y,-1), QYoff(y,1), z-1, z+1);
            }
        }
    }

    for (int x : {0, nx-1}) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int z = 1; z < nz - 1; ++z) {
                compute_cell(x, y, z, QXoff(x,-1), QXoff(x,1), y-1, y+1, z-1, z+1);
            }
        }
    }
    // Now, the 12 edges and 8 corners together
    for (int x = 0; x < nx; ++x) {
        compute_cell(x, 0,       0, QXoff(x,-1), QXoff(x,1), QYoff(0,-1),             1, QZoff(0, -1),              1);
        compute_cell(x, ny-1,    0, QXoff(x,-1), QXoff(x,1),        ny-2, QYoff(ny-1,1), QZoff(0, -1),              1);
        compute_cell(x, 0,    nz-1, QXoff(x,-1), QXoff(x,1), QYoff(0,-1),             1,         nz-2, QZoff(nz-1, 1));
        compute_cell(x, ny-1, nz-1, QXoff(x,-1), QXoff(x,1),        ny-2, QYoff(ny-1,1),         nz-2, QZoff(nz-1, 1));
    }

    for (int y = 0; y < ny; ++y) {
        compute_cell(   0, y,    0, QXoff(0,-1),             1, QYoff(y,-1), QYoff(y,1), QZoff(0, -1),              1);
        compute_cell(nx-1, y,    0,        nx-2, QXoff(nx-1,1), QYoff(y,-1), QYoff(y,1), QZoff(0, -1),              1);
        compute_cell(   0, y, nz-1, QXoff(0,-1),             1, QYoff(y,-1), QYoff(y,1),         nz-2, QZoff(nz-1, 1));
        compute_cell(nx-1, y, nz-1,        nx-2, QXoff(nx-1,1), QYoff(y,-1), QYoff(y,1),         nz-2, QZoff(nz-1, 1));
    }

    for (int z = 0; z < nz; ++z) {
        compute_cell(   0,    0, z, QXoff(0,-1),             1, QYoff(0, -1),              1, QZoff(z,-1), QZoff(z,1));
        compute_cell(nx-1,    0, z,        nx-2, QXoff(nx-1,1), QYoff(0, -1),              1, QZoff(z,-1), QZoff(z,1));
        compute_cell(   0, ny-1, z, QXoff(0,-1),             1,         ny-2, QYoff(ny-1, 1), QZoff(z,-1), QZoff(z,1));
        compute_cell(nx-1, ny-1, z,        nx-2, QXoff(nx-1,1),         ny-2, QYoff(ny-1, 1), QZoff(z,-1), QZoff(z,1));
    }

}

template<typename BC>
void QTensorSolver<BC>::Step(QTensorFields& qf, FluidFields& ff) const {
    FiniteDifferenceStep(qf, ff);
    ComputeActiveBodyForce(ff, qf);
}
