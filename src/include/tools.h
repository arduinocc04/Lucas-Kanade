/**
 * @file tools.h
 * @author Daniel Cho
 * @date 2024.1.15 - 
 * @version 0.0.1
*/
#include <cmath>
#include <Eigen/Dense>
namespace mim {
    inline bool is_pixel_outside(const Eigen::Vector3d & pixel, int row_cnt, int col_cnt) {
        return pixel(0) < 0 || pixel(0) > col_cnt - 1 || pixel(1) < 0 || pixel(1) > row_cnt - 1 ||
               std::isnan(pixel(0)) || std::isnan(pixel(1));
    }
}