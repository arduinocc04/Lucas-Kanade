/**
 * @file test_tools.h
 * @author Daniel Cho
 * @date 2024.1.3 - 
 * @version 0.0.1
*/
#include <string>
#include <Eigen/Dense>

/**
 * @brief return gray scale image of given input_path or defined debug-purpose image or random image.
 * @details If DO_NOT_USE_OPENCV is defined, then it returns one of following three. \par
 *          1. if input_path is "centered_horizontal_line_65.png" then returns 65x65 binary(filled with 0 or 255) image.
 *             It has white horizontal line as long as width of image at the center, with black background. It's defined in test_tools.cpp \par
 *          2. if input_path is "centered_vertical_line_65" then returns 65x65 binary(0 or 255) image.
 *             It has white vertical line as long as height of image at the center, with black background. It's also defined in test_tools.cpp \par
 *          3. otherwise, return 65x65 random generate Image filled with floating points in the range [64, 192]. \par
 *          Otherwise(DO_NOT_USE_OPENCV isn't defined), if input_path is "random", return random numbers with same method in 3, otherwise it returns grayscale image of given input_path.
 * @see test_tools.cpp
*/
Eigen::MatrixXd load_grayscale_of_image(std::string input_path);

/**
 * @brief show grayscale image or do nothing.
 * @details If DO_NOT_USE_OPENCV is defined, then it does nothing. Otherwise, it shows grayscale image given milliseconds
*/
void show_grayscale_given_ms(const Eigen::MatrixXd & src, int ms);

void show_matrices_three_channel(const Eigen::MatrixXd & a, const Eigen::MatrixXd & b, const Eigen::MatrixXd & c, int ms);