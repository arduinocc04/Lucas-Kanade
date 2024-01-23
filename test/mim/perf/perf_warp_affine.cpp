/**
 * @file perf_warp_affine.cpp
 * @author Daniel Cho
 * @date 2024.1.9
 * @version 0.0.1
*/
#include <iostream>
#include <ctime>
#include "mim.h"

#define N 400

int main() {
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(4000, 4000);
    Eigen::MatrixXd z = Eigen::MatrixXd::Zero(4000, 4000);
    Eigen::Matrix3d w;

    std::clock_t st = std::clock();

    for(int i = 0; i < N; i++) {
        std::cout << i + 1 << "/" << N << std::endl;
        w = Eigen::Matrix3d::Random(3, 3);
        w(2, 0) = w(2, 1) = 0;
        w(2, 2) = 1;
        mim::warp_affine(a, z, w);
    }

    std::clock_t end = std::clock();
    std::cout << static_cast<double>(end - st)/CLOCKS_PER_SEC << std::endl;
}