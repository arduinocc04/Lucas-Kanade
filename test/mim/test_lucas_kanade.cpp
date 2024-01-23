/**
 * @file test_lucas_kanade.cpp
 * @author Daniel Cho
 * @date 2024.1.9
 * @version 0.0.1
 */
#include <ctime>
#include <iostream>
#include <algorithm>
#include <iomanip>

#include "mim.h"
#include "test_tools.h"

void print_matrix3d(const Eigen::Matrix3d& a) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << a(i, j) << " ";
    }
    std::cout << std::endl;
  }
}

inline double deg_to_rad(double deg) { return 3.1415926535 * deg / 180; }

Eigen::Matrix3d get_rotate_matrix(double angle, int rows, int cols) {
  const double c = std::cos(deg_to_rad(angle));
  const double s = std::sin(deg_to_rad(angle));

  Eigen::Matrix3d mov{{1, 0, static_cast<double>(rows - 1) / 2},
                      {0, 1, static_cast<double>(cols - 1) / 2},
                      {0, 0, 1}};

  Eigen::Matrix3d rot{{c, -s, 0}, {s, c, 0}, {0, 0, 1}};
  Eigen::Matrix3d affine = mov * rot * mov.inverse();
  return affine;
}

Eigen::MatrixXd blend_matrices(const std::pair<Eigen::MatrixXd, std::pair<double, double>> * matrices, int n) {
  Eigen::MatrixXd big_image_warped_by_assumed;
  std::pair<double, double> min_coord = matrices[0].second;
  for(int i = 0; i < n; i++) {
    min_coord.first = std::min(min_coord.first, matrices[i].second.first);
    min_coord.second = std::min(min_coord.second, matrices[i].second.second);
  }
  Eigen::MatrixXd * tmps = new Eigen::MatrixXd[n];
  for(int i = 0; i < n; i++) {
    tmps[i] = matrices[i].first;
    mim::add_zero_pading(tmps[i], static_cast<int>(matrices[i].second.first - min_coord.first), 0, matrices[i].second.second - min_coord.second, 0);
  }
  int max_rows = tmps[0].rows(), max_cols = tmps[0].cols();
  for(int i = 0; i < n; i++) {
    max_rows = std::max(max_rows, static_cast<int>(tmps[i].rows()));
    max_cols = std::max(max_cols, static_cast<int>(tmps[i].cols()));
  }
  for(int i = 0; i < n; i++) {
    mim::add_zero_pading(tmps[i], 0, max_rows - tmps[i].rows(), 0, max_cols - tmps[i].cols());
  }
  Eigen::MatrixXd ans = Eigen::MatrixXd::Zero(max_rows, max_cols);
  for(int i = 0; i < n; i++)
    ans += tmps[i];
  delete[] tmps;
  ans /= n;
  return ans;
}

int main(int argc, char* argv[]) {
  unsigned int seed = (unsigned int)time(0);
  std::cout << "SEED: " << seed << std::endl;
  srand(seed);

  Eigen::MatrixXd image_original;
  if (argc == 1)
    image_original = load_grayscale_of_image("test1.PNG");
  else
    image_original = load_grayscale_of_image(argv[1]);

  image_original = mim::gaussian_blur5(image_original);

  Eigen::Matrix3d w {{1, 0, 40},
                     {0, 1, 60},
                     {0, 0, 1}};

  Eigen::MatrixXd image_warped_by_ideal = load_grayscale_of_image("test0.PNG");
  image_warped_by_ideal = mim::gaussian_blur5(image_warped_by_ideal);
  const int pyr_cnt = 5;

  Eigen::Matrix3d initial = Eigen::Matrix3d::Identity();
  initial(0, 2) = static_cast<double>(880);
  initial(1, 2) = static_cast<double>(90);

  Eigen::MatrixXd ttttmp = Eigen::MatrixXd::Zero(image_original.rows(), image_original.cols());
  mim::warp_affine(image_warped_by_ideal, ttttmp, initial);

  Eigen::MatrixXd weight_for_src = Eigen::MatrixXd::Zero(image_original.rows(), image_original.cols());
  mim::warp_affine(Eigen::MatrixXd::Ones(image_original.rows(), image_original.cols()), weight_for_src, initial);

  std::cout << "sim - initial: " << mim::compare_matrices(image_original, ttttmp, 10*weight_for_src, mim::COMPARE_METHODS::NCC) << std::endl;

  Eigen::MatrixXd weight_for_dst = Eigen::MatrixXd::Zero(image_original.rows(), image_original.cols());
  mim::warp_affine(Eigen::MatrixXd::Ones(image_original.rows(), image_original.cols()), weight_for_dst, initial.inverse());

  Eigen::MatrixXd src_tmp = (image_original.array()*weight_for_src.array()).matrix();
  Eigen::MatrixXd src_warp = Eigen::MatrixXd::Zero(src_tmp.rows(), src_tmp.cols());
  mim::warp_affine(src_tmp, src_warp, initial.inverse());

  Eigen::MatrixXd dst_tmp = (image_warped_by_ideal.array()*weight_for_dst.array()).matrix();

  initial(0, 2) /= (1 << (pyr_cnt));
  initial(1, 2) /= (1 << (pyr_cnt));

  std::clock_t st = std::clock();
  Eigen::Matrix3d assumed = mim::get_warp(image_original, image_warped_by_ideal, initial, pyr_cnt, 1e-4, 100);
  std::clock_t end = std::clock();
  std::cout << "time: " << static_cast<double>(end - st)/CLOCKS_PER_SEC << std::endl;

  std::cout << "=====assumed=====" << std::endl;
  print_matrix3d(assumed);
  std::cout << "=====assumed=====" << std::endl;

  Eigen::MatrixXd image_warped_by_assumed = Eigen::MatrixXd::Zero(image_warped_by_ideal.rows(), image_warped_by_ideal.cols());
  mim::warp_affine(src_tmp, image_warped_by_assumed, assumed);
  show_matrices_three_channel(dst_tmp, dst_tmp, image_warped_by_assumed, 0); // B G R. R means wrong, gray(white to black) means right.
  show_grayscale_given_ms(image_warped_by_assumed, 0);
  double sim = mim::compare_matrices(image_warped_by_assumed, dst_tmp, Eigen::MatrixXd::Ones(image_warped_by_ideal.rows(), image_warped_by_ideal.cols()),
                                     mim::COMPARE_METHODS::NCC);

  std::pair<Eigen::MatrixXd, std::pair<double, double>> images[2] = {{image_warped_by_ideal, {0.f, 0.f}}, mim::warp_affine_fit(image_original, assumed)};
  show_grayscale_given_ms(blend_matrices(images, 2), 0);
  std::cout << "sim: " << std::fixed << std::setprecision(4) << sim << std::endl;

  return sim <= 0.9;
}
