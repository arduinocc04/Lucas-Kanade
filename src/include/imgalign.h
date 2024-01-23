/**
 * @file imgalign.h
 * @author Daniel Cho
 * @date 2024.1.4 -
 * @version 0.0.1
 */
#include <Eigen/Dense>

namespace mim {
  /**
   * @brief Calculate 3x3 affine matrix that warps src to dst.
   * @details Calculate 3x3 affine matrix that warps src to dst using inverse compositional
   * lucas-kanade method with coarse to fine technique. \par We used inverse compositional algorithm
   * because it's fast. Since we used the algorithm, we can pre-compute hessian and some other
   * things.
   * @attention 1/2^(pyr_cnt + 1) scaled dst warped with initial_warp must be close enough to src. NOT 1/2^(pyr_cnt) scaled!!
   * @param src
   * @param dst
   * @param initial_warp affine matrix that warps 1/2^(pyr_cnt + 1) scaled dst to 1/2^(pyr_cnt + 1) scaled src.
   * @param pyr_cnt count of image pyramid. If this is to big, it will cause error because
   * generating one more pyramid shrinks both width and height of image by 1/2.
   * @param threshold if norm of delta_p is lower than it, terminate iteration. By default, 1e-4
   * @param max_iter if count of oterations is bigger than it, terminate iteration. By default, 20
   * @return warp: src -> dst
   * @see https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2004_1/baker_simon_2004_1.pdf
   * @see lucas_kanade_one_layer in imgalign.cpp
   * @see lucas_kanade_multiple_layer in imgalign.cpp
   */
  Eigen::Matrix3d get_warp(const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst,
                           const Eigen::Matrix3d& initial_warp, int pyr_cnt,
                           double threshold = 1e-4, int max_iter = 20);
}