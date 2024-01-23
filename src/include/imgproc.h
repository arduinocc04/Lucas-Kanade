/**
 * @file imgproc.h
 * @author Daniel Cho
 * @date 2024.1.3 - 
 * @version 0.0.1
*/
#include <utility> // for std::pair
#include <Eigen/Dense>

namespace mim {
    /**
     * @brief append zeros at given matrix.
     * @param a a 2d double matrix(Mat2d)
     * @param left count of columns to added at lower column index side.
     * @param bottom count of rows to added at upper row index side.
     * @param top count of rows to added at lower row index side.
     * @param right count of columns to added at upper column index side.
    */
    void add_zero_pading(Eigen::MatrixXd & src, int left, int bottom, int top, int right);

    /**
     * @brief translate entire image.
     * @param src matrix that need to be translated.
     * @param r double value to be shifted in horizontal(direction of increasing column). It may be negative.
     * @param d double value to be shifted in vertical(direction of increasing row). It may be negative.
     * @attention this will change given matrix
    */
    void translate(Eigen::MatrixXd & src, double r, double d, std::pair<int, int> dst_size);

    /**
     * @brief do warp affine 
     * @attention given affine matrix must be invertible.
     * @details It calculates inverse of affine matrix and find src pixel(usually fractional) corresponds to dst pixel. \par
     *          Then, fill the dst pixel with linear linterpolated value from src.
     * @param src double-valued matrix
     * @param dst double-valued matrix. warped value of src will be added to this matrix.
     * @param affine 3x3 double-valued matrix its last(third) row must be \f$ \begin{bmatrix}0 & 0 & 1\end{bmatrix} \f$.
    */
    void warp_affine(const Eigen::MatrixXd & src, Eigen::MatrixXd & dst, const Eigen::Matrix3d & affine);

    /**
     * @brief do warp and automatically export full image
     * @attention This is implemented under the assumption that vertices are still vertices even after warped.
     * @param src
     * @param affine 3x3 double-valued matrix its last(third) row must be \f$ \begin{bmatrix}0 & 0 & 1\end{bmatrix} \f$.
     * @return warped src and minimum coordinate(leftmost and topmost)
    */
    std::pair<Eigen::MatrixXd, std::pair<double, double>> warp_affine_fit(const Eigen::MatrixXd & src, const Eigen::Matrix3d & affine);

    /**
     * @brief return gaussian-blurred matrix.
     * @attention This isn't well-implemented. Handling edges is poor.
     * @details convolve matrix by 5x5 gaussian kernel: 
     *          \f$ \frac{1}{256}
     *          \begin{bmatrix}
     *          1 & 4 & 6 & 4 & 1 \\
     *          4 & 16 & 24 & 16 & 4 \\
     *          6 & 24 & 36 & 24 & 6 \\
     *          4 & 16 & 24 & 16 & 4 \\
     *          1 & 4 & 6 & 4 & 1
     *          \end{bmatrix} \f$
     * @see mim::pyr_down
    */
    Eigen::MatrixXd gaussian_blur5(const Eigen::MatrixXd & src);

    /**
     * @brief return half-size image
     * @details return image that both height and width are half of given image and pixel values are weighted average of given image. \par
     *          It's identical to mim::gaussian_blur5 except it shrinks image.
     * @todo I didn't test much. Just watch some generated images. And I don't have confidence that I have implement this right.
     * @param src
     * @see https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html
     * @see mim::gaussian_blur5
    */
    Eigen::MatrixXd pyr_down(const Eigen::MatrixXd & src);

    /**
     * @brief return linear-interpolated value of fractional pixel
     * @param src
     * @param pixel double-valued vector that has three elements. Last element must be one. If not, nothing happens but it may not return value that you expected.
    */
    inline double linear_interpolate_fractional_pixel(const Eigen::MatrixXd & src, const Eigen::Vector3d & pixel) {
        const int x_f = static_cast<int>(std::floor(pixel(0)));
        const int x_c = static_cast<int>(std::ceil(pixel(0)));
        const int y_f = static_cast<int>(std::floor(pixel(1)));
        const int y_c = static_cast<int>(std::ceil(pixel(1)));
        Eigen::Vector2d u = {pixel(0) - x_f, pixel(1) - y_f};
        Eigen::Vector2d grad = {src(y_f, x_c) - src(y_f, x_f), src(y_c, x_f) - src(y_f, x_f)}; // pixel(x, y) and matrix coordinate(y, x) are different
        return src(y_f, x_f) + grad.dot(u); // pixel(x, y) and matrix coordinate(y, x) are different
    }

    Eigen::MatrixXd blend(const Eigen::MatrixXd & a, const Eigen::MatrixXd & b, double alpha);
}