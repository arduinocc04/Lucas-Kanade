/**
 * @file imgcmp.h
 * @author Daniel Cho
 * @date 2024.1.3 - 
 * @version 0.0.1
*/
#include <Eigen/Dense>

namespace mim {
    /**
     * @enum COMPARE_METHODS
     * @brief compare methods for mim::compare_matrices
     * @see mim::compare_matrices
    */
    enum COMPARE_METHODS {
        NCC // normalized cross-correlation
    };

    /**
     * @brief compare two Eigen matrices.
     * @attention return values may vary. It may be similarlity or difference etc.. \par
     *            behavior of w may vary in choice of method. You must read implementation before compare two matrices. \par
     *            All implementations are inside imgcmp.cpp
     * @see mim::COMPARE_METHODS
     * @see imgcmp.cpp
    */
    double compare_matrices(const Eigen::MatrixXd & a, const Eigen::MatrixXd & b, const Eigen::MatrixXd & weight, int method);
}