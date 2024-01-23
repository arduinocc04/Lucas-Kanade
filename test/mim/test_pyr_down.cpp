/**
 * @file test_pyr_down.cpp
 * @author Daniel Cho
 * @date 2024.1.15
 * @version 0.0.1
*/
#include <iostream>
#include "mim.h"
#include "test_tools.h"

int main() {
    Eigen::MatrixXd a = load_grayscale_of_image("opencv_logo.jpg");
    show_grayscale_given_ms(a, 5000);
    Eigen::MatrixXd t = mim::pyr_down(a);
    show_grayscale_given_ms(t, 5000);
    show_grayscale_given_ms(mim::pyr_down(t), 5000);
}
