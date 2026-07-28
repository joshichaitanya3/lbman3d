#include <gtest/gtest.h>
#include "params.h"
#include "lattice_stencil.h"
#include "physics_helpers.h"
#include "local_grid.h"
#include <vector>
#include <random>

using namespace Lattice;

TEST(FeqMoments, ZeroVelocityFeq) {

    Vec3 u{0.0, 0.0, 0.0};
    double rho = 1.5;
    Moments m{rho, u};
    for (int i = 0; i < ndir; i++) {
        EXPECT_DOUBLE_EQ(Feq(m, 0, 0, w[i]), rho*w[i]);
    }
}

TEST(FeqMoments, FeqZerothMoment) {

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> rho_dist(0.5, 2.0);
    std::uniform_real_distribution<double> u_dist(-0.3, 0.3);

    for (int s = 0; s < 5; s++) {
        double sum = 0.0;
        Vec3 u{u_dist(rng), u_dist(rng), u_dist(rng)};
        double rho = rho_dist(rng);
        if (s==0) {
            u.x = 0.0; u.y = 0.0; u.z = 0.0;
            rho = 1.0;
        }

        double u2 = u.Dot(u);

        Moments m{rho, u};
        for (int i = 0; i < ndir; i++) {
            Vec3 e_i{
                static_cast<double>(ex[i]),
                static_cast<double>(ey[i]),
                static_cast<double>(ez[i])
            };

            sum += Feq(m, u.Dot(e_i), u2, w[i]);
        }
        EXPECT_NEAR(sum, rho, 1e-15);
    }
}

TEST(FeqMoments, FeqFirstMoment) {

    std::mt19937 rng(33);
    std::uniform_real_distribution<double> rho_dist(0.5, 2.0);
    std::uniform_real_distribution<double> u_dist(-0.3, 0.3);

    for (int s = 0; s < 5; s++) {

        Vec3 u{u_dist(rng), u_dist(rng), u_dist(rng)};
        double rho = rho_dist(rng);
        if (s==0) {
            u.x = 0.0; u.y = 0.0; u.z = 0.0;
            rho = 1.0;
        }

        double u2 = u.Dot(u);

        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;
        Moments m{rho, u};
        for (int i = 0; i < ndir; i++) {
            Vec3 e_i{
                static_cast<double>(ex[i]),
                static_cast<double>(ey[i]),
                static_cast<double>(ez[i])
            };
            
            double feq = Feq(m, u.Dot(e_i), u2, w[i]);

            sum_x += e_i.x * feq;
            sum_y += e_i.y * feq;
            sum_z += e_i.z * feq;
        }
        EXPECT_NEAR(sum_x, rho * u.x, 1e-12);
        EXPECT_NEAR(sum_y, rho * u.y, 1e-12);
        EXPECT_NEAR(sum_z, rho * u.z, 1e-12);
    }
}

TEST(FeqMoments, MomentsRoundtrip) {

    std::mt19937 rng(78);
    std::uniform_real_distribution<double> rho_dist(0.5, 2.0);
    std::uniform_real_distribution<double> u_dist(-0.3, 0.3);

    for (int s = 0; s < 5; s++) {

        Vec3 u{u_dist(rng), u_dist(rng), u_dist(rng)};
        double rho = rho_dist(rng);
        if (s==0) {
            u.x = 0.0; u.y = 0.0; u.z = 0.0;
            rho = 1.0;
        }

        double u2 = u.Dot(u);

        double f[ndir];
        Moments m{rho, u};
        for (int i = 0; i < ndir; i++) {
            Vec3 e_i{
                static_cast<double>(ex[i]),
                static_cast<double>(ey[i]),
                static_cast<double>(ez[i])
            };
            
            double feq = Feq(m, u.Dot(e_i), u2, w[i]);
            f[i] = feq;
        }

        LocalGrid g = LocalGrid::SingleRank();
        Moments m2 = ComputeMoments(
            f,
            {0,0,0},
            {0.0, 0.0, 0.0},
            ex,
            ey,
            ez,
            g
        );

        EXPECT_NEAR(m2.rho, rho, 1e-15);
        EXPECT_NEAR(m2.u.x, u.x, 1e-12);
        EXPECT_NEAR(m2.u.y, u.y, 1e-12);
        EXPECT_NEAR(m2.u.z, u.z, 1e-12);
    }
}

TEST(FeqMoments, GuoHalfStepCorrection) {

    std::mt19937 rng(57);
    std::uniform_real_distribution<double> rho_dist(0.5, 2.0);
    std::uniform_real_distribution<double> force_dist(-0.3, 0.3);

    for (int s = 0; s < 5; s++) {
        double rho = rho_dist(rng);
        double force_x = force_dist(rng);
        if (s==0) {
            rho = 1.0;
            force_x = 0.0;
        }

        double f[ndir];
        for (int i = 0; i < ndir; i++) {

            f[i] = rho*w[i]; // For zero velocity, f[i] = rho*w[i]
        }

        LocalGrid g = LocalGrid::SingleRank();

        Moments m = ComputeMoments(
            f,
            {0,0,0},
            {force_x, 0.0, 0.0},
            ex,
            ey,
            ez,
            g
        );

        EXPECT_NEAR(m.u.x, 0.5 * force_x * DT / rho , 1e-12);
    }
}
