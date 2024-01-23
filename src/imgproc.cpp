/**
 * @file imgproc.cpp
 * @author Daniel Cho
 * @date 2024.1.3 - 
 * @version 0.0.1
*/
#include "imgproc.h"
#include "tools.h"
#define eps 1e-6

void mim::add_zero_pading(Eigen::MatrixXd & a, int left, int bottom, int top, int right) {
    if(left != 0 || top != 0) {
        a.conservativeResize(a.rows() + top, a.cols() + left);
        for(int i = a.rows() - 1 - top; i >= 0; --i) {
            for(int j = a.cols() - 1 - left; j >= 0; --j) {
                a(i + top, j + left) = a(i, j);
            }
        }

        for(int i = 0; i < top; i++)
            for(int j = 0; j < a.cols(); j++)
                a(i, j) = 0;
        for(int i = 0; i < a.rows(); i++)
            for(int j = 0; j < left; j++)
                a(i, j) = 0;
    }
    a.conservativeResize(a.rows() + bottom, a.cols() + right);
    // requires EIGEN_INITIALIZE_MATRICES_BY_ZERO defined.
}

void mim::warp_affine(const Eigen::MatrixXd & src, Eigen::MatrixXd & dst, const Eigen::Matrix3d & affine) {
    assert(std::abs(affine(2, 0)) < eps);
    assert(std::abs(affine(2, 1)) < eps);
    assert(std::abs(affine(2, 2) - 1) < eps);

    const Eigen::Matrix3d affine_inv = affine.inverse();

    for(int i = 0; i < dst.rows(); i++) {
        for(int j = 0; j < dst.cols(); j++) {
            const double x = static_cast<double> (j);
            const double y = static_cast<double> (i);
            const Eigen::Vector3d pixel_in_dst = {x, y, 1};
            const Eigen::Vector3d pixel_in_src = affine_inv*pixel_in_dst;
            if(mim::is_pixel_outside(pixel_in_src, src.rows(), src.cols())) continue;

            dst(i, j) += linear_interpolate_fractional_pixel(src, pixel_in_src);
        }
    }
}

void mim::translate(Eigen::MatrixXd & a, double r, double d, std::pair<int, int> dst_size) {
    Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(dst_size.first, dst_size.second);
    Eigen::Matrix3d affine {{1, 0, r},
                            {0, 1, d},
                            {0, 0, 1}};
    warp_affine(a, tmp, affine);
    a = tmp;
}

std::pair<Eigen::MatrixXd, std::pair<double, double>> mim::warp_affine_fit(const Eigen::MatrixXd & src, const Eigen::Matrix3d & affine) {
    assert(std::abs(affine(2, 0)) < eps);
    assert(std::abs(affine(2, 1)) < eps);
    assert(std::abs(affine(2, 2) - 1) < eps);

    Eigen::Vector3d vertices[4] = {{0.f, 0.f, 1.f},
                                   {0.f, static_cast<double>(src.rows() - 1), 1.f},
                                   {static_cast<double>(src.cols() - 1), 0.f, 1.f},
                                   {static_cast<double>(src.cols() - 1), static_cast<double>(src.rows() - 1), 1.f}};

    double min_x, min_y, max_x, max_y;
    for(int i = 0; i < 4; i++) {
        Eigen::Vector3d warped_vertex = affine*vertices[i];
        if(i == 0 || min_x > warped_vertex(0)) {
            min_x = warped_vertex(0);
        }
        if(i == 0 || max_x < warped_vertex(0)) {
            max_x = warped_vertex(0);
        }
        if(i == 0 || min_y > warped_vertex(1)) {
            min_y = warped_vertex(1);
        }
        if(i == 0 || max_y < warped_vertex(1)) {
            max_y = warped_vertex(1);
        }
    }

    Eigen::MatrixXd ans = Eigen::MatrixXd::Zero(std::ceil(max_y - min_y), std::ceil(max_x - min_x));
    Eigen::Matrix3d move_to_fit = Eigen::Matrix3d {{1, 0, -min_x},
                                                   {0, 1, -min_y},
                                                   {0, 0, 1}};
    Eigen::Matrix3d affine_fitted = move_to_fit*affine;
    mim::warp_affine(src, ans, affine_fitted);

    return std::make_pair(ans, std::make_pair(min_x, min_y));
}

Eigen::MatrixXd mim::gaussian_blur5(const Eigen::MatrixXd & src) {
    const Eigen::Array<double, 5, 5> gaussian_blur_kernel = Eigen::Array<double, 5, 5> {{1, 4,  6,  4,  1},
                                                                                        {4, 16, 24, 16, 4},
                                                                                        {6, 24, 36, 24, 6},
                                                                                        {4, 16, 24, 16, 4},
                                                                                        {1, 4,  6,  4,  1}}/256;
    Eigen::MatrixXd ans(src.rows(), src.cols());
    for(int i = 0; i < src.rows(); i++) {
        for(int j = 0; j < src.cols(); j++) {
            if(i < 2 || i + 3 >= src.rows()) {
                ans(i, j) = src(i, j);
                continue;
            }
            if(j < 2 || j + 3 >= src.cols()) {
                ans(i, j) = src(i, j);
                continue;
            }
            ans(i, j) = (src.block(i - 2, j - 2, 5, 5).array()*gaussian_blur_kernel).sum();
        }
    }
    return ans;
}

Eigen::MatrixXd mim::pyr_down(const Eigen::MatrixXd & a) {
    const Eigen::Array<double, 5, 5> gaussian_blur_kernel = Eigen::Array<double, 5, 5> {{1, 4,  6,  4,  1},
                                                                                        {4, 16, 24, 16, 4},
                                                                                        {6, 24, 36, 24, 6},
                                                                                        {4, 16, 24, 16, 4},
                                                                                        {1, 4,  6,  4,  1}}/256;

    Eigen::MatrixXd ans(a.rows()/2, a.cols()/2);
    for(int i = 0; i < a.rows()/2; i++) {
        for(int j = 0; j < a.cols()/2; j++) {
            const int i_idx_in_a = 2*i + 1;
            const int j_idx_in_a = 2*j + 1;
            if(i_idx_in_a < 2 || i_idx_in_a + 3 >= a.rows()) {
                ans(i, j) = a(i_idx_in_a, j_idx_in_a);
                continue;
            }
            if(j_idx_in_a < 2 || j_idx_in_a + 3 >= a.cols()) {
                ans(i, j) = a(i_idx_in_a, j_idx_in_a);
                continue;
            }
            ans(i, j) = (a.block(i_idx_in_a - 2, j_idx_in_a - 2, 5, 5).array()*gaussian_blur_kernel).sum();
        }
    }
    return ans;
}

Eigen::MatrixXd mim::blend(const Eigen::MatrixXd & a, const Eigen::MatrixXd & b, double alpha) {
    assert(0 <= alpha && alpha <= 1);
    assert(a.rows() == b.rows() && a.cols() == b.cols());
    return alpha*a + (1 - alpha)*b;
}
