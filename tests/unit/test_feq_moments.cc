#include <gtest/gtest.h>
#include "params.h"
#include "lattice_stencil.h"
#include "physics_helpers.h"
#include <vector>

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

    std::vector<Vec3> test_velocities;
    
    test_velocities.push_back({0.0, 0.0, 0.0});
    test_velocities.push_back({0.6, -1.0, 2.0});
    test_velocities.push_back({-1.0, -1.0, -1.0});
    test_velocities.push_back({0.003, 0.0, 0.0});

    std::vector<double> test_rhos{0.5, 1.0, 5.2, 0.01};
    for (int s = 0; s < 4; s++) {
        double sum = 0.0;
        Vec3 u = test_velocities[s];
        double u2 = u.Dot(u);
        double rho = test_rhos[s];

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

    std::vector<Vec3> test_velocities;
    
    test_velocities.push_back({0.0, 0.0, 0.0});
    test_velocities.push_back({0.6, -1.0, 2.0});
    test_velocities.push_back({-1.0, -1.0, -1.0});
    test_velocities.push_back({0.003, 0.0, 0.0});

    std::vector<double> test_rhos{0.5, 1.0, 5.2, 0.01};
    for (int s = 0; s < 4; s++) {
        Vec3 u = test_velocities[s];
        double u2 = u.Dot(u);
        double rho = test_rhos[s];
        
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

    std::vector<Vec3> test_velocities;
    
    test_velocities.push_back({0.0, 0.0, 0.0});
    test_velocities.push_back({0.6, -1.0, 2.0});
    test_velocities.push_back({-1.0, -1.0, -1.0});
    test_velocities.push_back({0.003, 0.0, 0.0});

    std::vector<double> test_rhos{0.5, 1.0, 5.2, 0.01};
    for (int s = 0; s < 4; s++) {
        Vec3 u = test_velocities[s];
        double u2 = u.Dot(u);
        double rho = test_rhos[s];
        
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;
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

        Moments m2 = ComputeMoments(
            f,
            {0,0,0},
            {0.0, 0.0, 0.0},
            ex,
            ey,
            ez
        );

        EXPECT_NEAR(m2.rho, rho, 1e-15);
        EXPECT_NEAR(m2.u.x, u.x, 1e-12);
        EXPECT_NEAR(m2.u.y, u.y, 1e-12);
        EXPECT_NEAR(m2.u.z, u.z, 1e-12);
    }
}

TEST(FeqMoments, GuoHalfStepCorrection) {

    std::vector<double> test_rhos{0.5, 1.0, 5.2, 0.01};
    std::vector<double> test_forces{0.0, 0.1, -0.5, 10.0};
    for (int s = 0; s < 4; s++) {
        Vec3 u{0.0, 0.0, 0.0};
        double u2 = u.Dot(u);
        double rho = test_rhos[s];
        double force_x = test_forces[s];
        
        double f[ndir];
        for (int i = 0; i < ndir; i++) {

            f[i] = rho*w[i]; // For zero velocity, f[i] = rho*w[i]
        }

        Moments m = ComputeMoments(
            f,
            {0,0,0},
            {force_x, 0.0, 0.0},
            ex,
            ey,
            ez
        );

        EXPECT_NEAR(m.u.x, 0.5 * force_x * DT / rho , 1e-12);
    }
}
