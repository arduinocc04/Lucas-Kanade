/**
 * @file test_imgcmp.cpp
 * @author Daniel Cho
 * @date 2024.1.3, 2024.1.8
 * @version 0.0.1
*/
#include <iostream>
#include "mim.h"
#include "test_tools.h"

bool wrong(const Eigen::MatrixXd & original, const Eigen::MatrixXd & moved, int translated) {
    if(original.rows() != moved.rows() || original.cols() != moved.cols()) return true;
    for(int i = 0; i < original.rows(); i++) {
        for(int j = 0; j < translated; j++) {
            if(moved(i, j) != 0) return true;
        }
    }
    for(int i = 0; i < moved.rows(); i++) {
        for(int j = translated; j < moved.cols(); j++) {
            if(moved(i, j) != original(i, j - translated)) return true;
        }
    }
    return false;
}

int main(int argc, char * argv[]) {
    std::string image_name = "opencv_logo.jpg";
    Eigen::MatrixXd a = load_grayscale_of_image(image_name);
    Eigen::MatrixXd b = a; // deep copy

    show_grayscale_given_ms(mim::pyr_down(a), 1000);

    Eigen::MatrixXd w = Eigen::MatrixXd::Ones(a.rows(), a.cols());

    const int stride = 20;

    for(int x = 0; x < a.cols(); x += stride) {
        std::cout << "x: " << x << " NCC val: " << mim::compare_matrices(a, b, w, mim::COMPARE_METHODS::NCC) << std::endl;
        if(wrong(a, b, x)) {
            std::cerr << "translation wrong: " << x << std::endl;
            return -1;
        }

        mim::translate(b, stride, 0, {static_cast<double>(a.rows()), static_cast<double>(a.cols())});
        show_grayscale_given_ms(b, 500);
        for(int i = 0; i < w.rows(); i++) {
            for(int j = x; j < w.cols() && j < x + stride; j++)
                w(i, j) = 0;
        }
    }

    return 0;
}
