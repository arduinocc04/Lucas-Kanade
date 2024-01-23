/**
 * @file imgalign.cpp
 * @author Daniel Cho
 * @date 2024.1.4 - 
 * @version 0.0.1
*/
#include <cmath>
#include "imgalign.h"
#include "imgproc.h"
#include "tools.h"

#define WARP_TRANSLATION 2 /* Degree of Freedom of translation */
#define WARP_EUCLIDEAN 3
#define WARP_AFFINE 6
#define WARP_DOF WARP_TRANSLATION // DOF means Degree of Freedom

#define LK_IC 0 // Lucas-Kanade Inverse Compositional
#define LK_FA 1 // Lucas-Kanade Forward Additive
#define LK_MODE LK_IC

/**
 * @todo This result can be used to linear-interpolate pixel value. I think it can boost speed slightly.
*/
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> calculate_img_grad(const Eigen::MatrixXd & src) {
    Eigen::MatrixXd dy(src.rows(), src.cols());
    Eigen::MatrixXd dx(src.rows(), src.cols());
    for(int i = 0; i < src.rows(); i++) {
        for(int j = 0; j < src.cols(); j++) {
            if(i == src.rows() - 1)
                dy(i, j) = 0;
            else
                dy(i, j) = src(i + 1, j) - src(i, j);

            if(j == src.cols() - 1)
                dx(i, j) = 0;
            else
                dx(i, j) = src(i, j + 1) - src(i, j);
        }
    }
    return std::make_pair(dx, dy);
}

inline void set_last_row_zero_zero_one(Eigen::Matrix3d & a) {
    a(2, 0) = a(2, 1) = 0;
    a(2, 2) = 1;
}

inline void set_delta_warp_by_delta_p(Eigen::Matrix3d & delta_p_matrix, const Eigen::Matrix<double, WARP_DOF, 1> & delta_p) {
#if (WARP_DOF == WARP_TRANSLATION)
    delta_p_matrix(0, 2) = delta_p(0);
    delta_p_matrix(1, 2) = delta_p(1);
#elif (WARP_DOF == WARP_EUCLIDEAN)
    const double c = cos(delta_p(2));
    const double s = sin(delta_p(2));
    delta_p_matrix(0, 0) = c;
    delta_p_matrix(1, 0) = s;
    delta_p_matrix(0, 1) = -s;
    delta_p_matrix(1, 1) = c;
    delta_p_matrix(0, 2) = delta_p(0);
    delta_p_matrix(1, 2) = delta_p(1);
#elif (WARP_DOF == WARP_AFFINE)
    delta_p_matrix(0, 0) = 1 + delta_p(0, 0);
    delta_p_matrix(1, 0) = delta_p(1, 0);
    delta_p_matrix(0, 1) = delta_p(2, 0);
    delta_p_matrix(1, 1) = 1 + delta_p(3, 0);
    delta_p_matrix(0, 2) = delta_p(4, 0);
    delta_p_matrix(1, 2) = delta_p(5, 0);
#endif
}

Eigen::Matrix<double, WARP_DOF, 1> * get_steepest_descent_images_transposed(const Eigen::MatrixXd & T) {
    Eigen::MatrixXd T_dy, T_dx;
    std::tie(T_dx, T_dy) = calculate_img_grad(T);

    const int n = T.rows()*T.cols();
    Eigen::Matrix<double, WARP_DOF, 1> * steepest_descent_images_transposed = new Eigen::Matrix<double, WARP_DOF, 1>[n];
    Eigen::Matrix<double, 2, WARP_DOF> partial_warp_over_p;
    Eigen::Vector2d dst_gradient;
    Eigen::Matrix<double, 1, WARP_DOF> steepest_descent_image;

    for(int i = 0; i < T.rows(); i++) {
        for(int j = 0; j < T.cols(); j++) {
            double x = static_cast<double>(j);
            double y = static_cast<double>(i);
#if (WARP_DOF == WARP_TRANSLATION) /* p = (tx, ty) */
            partial_warp_over_p = Eigen::Matrix<double, 2, 2>::Identity();
#elif (WARP_DOF == WARP_EUCLIDEAN) /* p = (tx, ty, theta) */
            partial_warp_over_p = Eigen::Matrix<double, 2, 3> {{1, 0, -y},
                                                               {0, 1, x}};
#elif (WARP_DOF == WARP_AFFINE) /* p = (a00, a10, a01, a11, a02, a12) */
            partial_warp_over_p = Eigen::Matrix<double, 2, 6> {{x,   0.f, y,   0.f, 1.f, 0.f},
                                                               {0.f, x,   0.f, y,   0.f, 1.f}};
#endif
            dst_gradient = Eigen::Matrix<double, 2, 1> {T_dx(i, j), T_dy(i, j)};
            steepest_descent_image = dst_gradient.transpose()*partial_warp_over_p;

            steepest_descent_images_transposed[i*T.cols() + j] = steepest_descent_image.transpose();
        }
    }
    return steepest_descent_images_transposed;
}

Eigen::Matrix<double, WARP_DOF, WARP_DOF> get_hessian_inv(const Eigen::Matrix<double, WARP_DOF, 1> * steepest_descent_images_transposed, int n) {
    Eigen::Matrix<double, WARP_DOF, WARP_DOF> hessian = Eigen::Matrix<double, WARP_DOF, WARP_DOF>::Zero();
    for(int i = 0; i < n; i++) {
        hessian += steepest_descent_images_transposed[i]*steepest_descent_images_transposed[i].transpose();
    }
    return hessian.inverse();
}

Eigen::Matrix3d lucas_kanade_one_layer(const Eigen::MatrixXd & src, const Eigen::MatrixXd & dst, const Eigen::Matrix3d & initial_warp, double threshold, int max_iter) {
    assert(src.rows() > 1 && src.cols() > 1); // size of dst must be same so do not need checking
    const Eigen::MatrixXd & I = src;
    const Eigen::MatrixXd & T = dst;
    /* pre-compute */
#if (LK_MODE == LK_IC)
    Eigen::Matrix<double, WARP_DOF, 1> * steepest_descent_images_transposed = get_steepest_descent_images_transposed(T); // [∇T(∂W/∂p)]^T
    Eigen::Matrix<double, WARP_DOF, WARP_DOF> hessian_inv = get_hessian_inv(steepest_descent_images_transposed, T.rows()*T.cols()); // (∑[∇T(∂W/∂p)]^T[∇T(∂W/∂p)])^(-1)
    Eigen::Matrix3d delta_warp = Eigen::Matrix3d::Identity();
#endif
    Eigen::Matrix3d warp = initial_warp;
    Eigen::Matrix<double, WARP_DOF, 1> delta_p;

    /* iterate */
    int it = 0;
    do {
        Eigen::Vector3d pixel_in_dst, pixel_in_src;
        delta_p = Eigen::Matrix<double, WARP_DOF, 1>::Zero();
#if (LK_MODE == LK_FA)
        Eigen::Matrix<double, WARP_DOF, WARP_DOF> hessian = Eigen::Matrix<double, WARP_DOF, WARP_DOF>::Zero();
        Eigen::MatrixXd I_dx, I_dy;
        std::tie(I_dx, I_dy) = calculate_img_grad(I);
#endif
        for(int i = 0; i < T.rows(); i++) {
            for(int j = 0; j < T.cols(); j++) {
                const double x = static_cast<double>(j);
                const double y = static_cast<double>(i);
                pixel_in_dst = {x, y, 1.f};
                pixel_in_src = warp*pixel_in_dst;

                if(mim::is_pixel_outside(pixel_in_src, I.rows(), I.cols())) continue;
#if (LK_MODE == LK_IC)
                // [∇T(∂W/∂p)]^T(I(W(x;p))-T(x))
                const double err = mim::linear_interpolate_fractional_pixel(I, pixel_in_src) - T(i, j);
                delta_p += steepest_descent_images_transposed[i*T.cols() + j]*err;
#elif (LK_MODE == LK_FA)
    #if (WARP_DOF == WARP_TRANSLATION) /* p = (tx, ty) */
                Eigen::Matrix<double, 2, 2> partial_warp_over_p = Eigen::Matrix<double, 2, 2>::Identity();
    #elif (WARP_DOF == WARP_EUCLIDEAN) /* p = (tx, ty, theta) */
                const double c = warp(0, 0);
                const double s = warp(1, 0);
                Eigen::Matrix<double, 2, 3> partial_warp_over_p = Eigen::Matrix<double, 2, 3> {{1, 0, -s*x - c*y},
                                                                                               {0, 1, c*x - s*y}};
    #elif (WARP_DOF == WARP_AFFINE) /* p = (a00, a10, a01, a11, a02, a12) */
                Eigen::Matrix<double, 2, 6> partial_warp_over_p = Eigen::Matrix<double, 2, 6> {{x,   0.f, y,   0.f, 1.f, 0.f},
                                                                                               {0.f, x,   0.f, y,   0.f, 1.f}};
    #endif
                const int warped_x = static_cast<int>(std::floor(pixel_in_src(0)));
                const int warped_y = static_cast<int>(std::floor(pixel_in_src(1)));
                if(warped_x < 0 || warped_y < 0 || warped_x >= I.cols() || warped_y >= I.rows()) continue;
                Eigen::Matrix<double, 1, 2> grad = {I_dx(warped_y, warped_x), I_dy(warped_y, warped_x)}; /* use nearest integer-coordinate gradient. May be it can cause problems.. */
                Eigen::Matrix<double, 1, WARP_DOF> steepest_descent = grad*partial_warp_over_p;
                hessian += steepest_descent.transpose()*steepest_descent;

                const double err = T(i, j) - mim::linear_interpolate_fractional_pixel(I, pixel_in_src);
                delta_p += steepest_descent.transpose()*err;
#endif
            }
        }
#if (LK_MODE == LK_IC)
        delta_p = hessian_inv*delta_p;
        set_delta_warp_by_delta_p(delta_warp, delta_p);
        assert(delta_warp.determinant() != 0);
        warp = warp*delta_warp.inverse(); // W(x;p) <- W(x;p)∘(W(x;Δp)^(-1))
#elif (LK_MODE == LK_FA)
        delta_p = hessian.inverse()*delta_p;
    #if (WARP_DOF == WARP_TRANSLATION)
        warp(0, 2) += delta_p(0);
        warp(1, 2) += delta_p(1);
    #elif (WARP_DOF == WARP_EUCLIDEAN)
        const double prev_c = warp(0, 0);
        const double prev_s = warp(1, 0);
        const double delta_c = cos(delta_p(2));
        const double delta_s = sin(delta_p(2));
        const double c = prev_c*delta_c - prev_s*delta_s;
        const double s = prev_s*delta_c + prev_c*delta_s;
        warp(0, 0) = c;
        warp(1, 0) = s;
        warp(0, 1) = -s;
        warp(1, 1) = c;
        warp(0, 2) += delta_p(0);
        warp(1, 2) += delta_p(1);
    #elif (WARP_DOF == WARP_AFFINE)
        for(int i = 0; i < 6; i++)
            warp(i % 2, i/2) += delta_p(i);
    #endif
#endif
    }while(++it <= max_iter && delta_p.norm() >= threshold);

#if (LK_MODE == LK_IC)
    delete[] steepest_descent_images_transposed;
#endif
    return warp;
}

Eigen::Matrix3d lucas_kanade_multiple_layer(const Eigen::MatrixXd & src, const Eigen::MatrixXd & dst, const Eigen::Matrix3d & initial_warp, int level, double threshold, int max_iter) {
    if(level == 0) return initial_warp;

    const Eigen::MatrixXd src_down = mim::pyr_down(src);
    const Eigen::MatrixXd dst_down = mim::pyr_down(dst);

    Eigen::Matrix3d one_more_coarse_warp = lucas_kanade_multiple_layer(src_down, dst_down, initial_warp, level - 1, threshold, max_iter);
#if (WARP_DOF == WARP_TRANSLATION || WARP_DOF == WARP_EUCLIDEAN || WARP_DOF == WARP_AFFINE)
    one_more_coarse_warp(0, 2) *= 2;
    one_more_coarse_warp(1, 2) *= 2;
#endif
    /* we don't need to do set_last_row_zero_zero_one because it will be done in mim::get_warp */
    return lucas_kanade_one_layer(src, dst, one_more_coarse_warp, threshold, max_iter);
}

Eigen::Matrix3d mim::get_warp(const Eigen::MatrixXd & src, const Eigen::MatrixXd & dst, const Eigen::Matrix3d & initial_warp, int pyr_cnt, double threshold, int max_iter) {
    assert(src.rows() == dst.rows() && src.cols() == dst.cols());

    /*
     * Because lucas_kanade_one_layer returns 3x3 affine matrix that warps dst to src,
     * We need to inverse the matrix to warp src to dst.
    */
    Eigen::Matrix3d ans = lucas_kanade_multiple_layer(src, dst, initial_warp, pyr_cnt, threshold, max_iter).inverse();
    return ans;
}
