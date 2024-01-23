/**
 * @file imgcmp.cpp
 * @author Daniel Cho
 * @date 2024.1.3 - 
 * @version 0.0.1
*/
#include <stdexcept>
#include <cmath>
#include "imgcmp.h"

double compare_Mat2d_NCC(const Eigen::MatrixXd & a, const Eigen::MatrixXd & b, const Eigen::MatrixXd & w) {
    double n = w.sum();
    assert(n != 0);

    double a0 = a.sum()/n;
    double b0 = b.sum()/n;

    Eigen::ArrayXXd ones = Eigen::ArrayXXd::Ones(a.rows(), a.cols());
    Eigen::ArrayXXd tmpA = w.array()*(a.array() - a0*ones);
    Eigen::ArrayXXd tmpB = w.array()*(b.array() - b0*ones);

    return (tmpA*tmpB).sum()/(std::sqrt((tmpA*tmpA).sum()*(tmpB*tmpB).sum()));
}

double mim::compare_matrices(const Eigen::MatrixXd & a, const Eigen::MatrixXd & b, const Eigen::MatrixXd & weight, int method) {
    assert(a.rows() == b.rows() && a.cols() == b.cols());
    assert(a.rows() == weight.rows() && a.cols() == weight.cols());

    switch (method) {
        case mim::COMPARE_METHODS::NCC:
            return compare_Mat2d_NCC(a, b, weight);
        default:
            throw std::invalid_argument("Method doesn't exist!");
    }
}
