/**
 * @file test_warp_affine.cpp
 * @author Daniel Cho
 * @date 2024.1.4, 2024.1.8
 * @version 0.0.1
*/
#include <iostream>
#include <cmath>
#include <vector>

#include "mim.h"
#include "test_tools.h"

#define eps 1e-2

inline double deg_to_rad(double deg) {
    return 3.1415926535*deg/180;
}

void print_matrix3d(const Eigen::Matrix3d & a) {
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            std::cout << a(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

Eigen::Matrix3d get_rotate_matrix(double angle, int rows, int cols) {
    const double x = static_cast<double>(cols - 1)/2;
    const double y = static_cast<double>(rows - 1)/2;
    if(angle == 90) {
        return Eigen::Matrix3d {{0, -1, x + y}, {1, 0, 0}, {0, 0, 1}};
    }
    else if(angle == -90) {
        return Eigen::Matrix3d {{0, 1, 0}, {-1, 0, x + y}, {0, 0, 1}};
    }
    const double c = std::cos(deg_to_rad(angle));
    const double s = std::sin(deg_to_rad(angle));

    Eigen::Matrix3d mov {{1, 0, x},
                         {0, 1, y},
                         {0, 0, 1}};

    Eigen::Matrix3d rot {{c, -s, 0},
                         {s, c, 0},
                         {0, 0, 1}};
    Eigen::Matrix3d affine = mov*rot*mov.inverse();
    return affine;
}

void rotate_matrix_by_angle(Eigen::MatrixXd & a, double angle) {
    assert(a.rows() != 0 && a.cols() != 0);

    Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(a.rows(), a.cols());
    auto affine = get_rotate_matrix(angle, a.rows(), a.cols());
    std::cout << "======== angle: " << angle << std::endl;
    print_matrix3d(affine);
    std::cout << "======== angle: " << angle << std::endl;
    mim::warp_affine(a, tmp, affine);

    a = tmp;
}

int main(int argc, char * argv[]) {
    std::string image_name = "centered_horizontal_line_65.png";
    Eigen::MatrixXd a = load_grayscale_of_image(image_name);
    Eigen::MatrixXd b = a; // deep copy

    show_grayscale_given_ms(mim::warp_affine_fit(a, get_rotate_matrix(45, a.rows(), a.cols())).first, 1'000);

    show_grayscale_given_ms(a, 1'000);

    const double angle = 90;
    rotate_matrix_by_angle(a, angle);
    rotate_matrix_by_angle(a, -angle);
    if(a.rows() != b.rows() || a.cols() != b.cols()) {
        std::cerr << "Back to original test failed: Image size doesn't match" << std::endl;
        return -1;
    }

    show_grayscale_given_ms(a, 1'000);

    std::vector<std::pair<int, int>> error_dots;

    for(int i = 0; i < a.rows(); i++) {
        for(int j = 0; j < a.cols(); j++) {
            if(std::abs(a(i, j) - b(i, j)) > eps) {
                error_dots.push_back(std::make_pair(i, j));
            }
        }
    }

    for(int i = 0; i < error_dots.size(); i++) {
        int x = error_dots[i].first;
        int y = error_dots[i].second;
        std::cerr << "Back to original test failed: Wrong value at ("
                  << x << ", " << y << ") with difference "
                  << std::abs(a(x, y) - b(x, y)) << std::endl;
    }
    if(error_dots.size()) return -2;

    return 0;
}
