/**
 * @file test_tools.cpp
 * @author Daniel Cho
 * @date 2024.1.3 - 
 * @version 0.0.1
*/
#include "test_tools.h"

#ifndef DO_NOT_USE_OPENCV
#include "opencv2/opencv.hpp"
#endif

Eigen::MatrixXd load_grayscale_of_image(std::string input_path) {
    const int n = 128;
#ifndef DO_NOT_USE_OPENCV
    if(input_path == "random")
        return 64*Eigen::MatrixXd::Random(n, n) + 128*Eigen::MatrixXd::Ones(n, n); // returns matrix filled with random double numbers in the range [-1*64 + 128, 1*64 + 128] = [64, 192].

    cv::Mat image = cv::imread(input_path);
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    Eigen::MatrixXd ans(gray.rows, gray.cols);
    for(int i = 0; i < gray.rows; i++)
        for(int j = 0; j < gray.cols; j++)
            ans(i, j) = gray.at<uint8_t>(i, j);
    return ans;
#else
    if(input_path == "centered_horizontal_line_65.png") {
        Eigen::MatrixXd ans(n, n);
        for(int i = 0; i < n; i++)
            ans(n/2, i) = 255;
        return ans;
    }
    else if(input_path == "centered_vertical_line_65.png") {
        Eigen::MatrixXd ans(n, n);
        for(int i = 0; i < n; i++)
            ans(i, n/2) = 255;
        return ans;
    }
    else {
        return 64*Eigen::MatrixXd::Random(n, n) + 128*Eigen::MatrixXd::Ones(n, n); // returns matrix filled with random double numbers in the range [-1*64 + 128, 1*64 + 128] = [64, 192].
    }
#endif
}

void show_grayscale_given_ms(const Eigen::MatrixXd & src, int ms) {
#ifndef DO_NOT_USE_OPENCV
    cv::Mat tmp(src.rows(), src.cols(), CV_64FC1);
    for(int i = 0; i < src.rows(); i++)
        for(int j = 0; j < src.cols(); j++)
            tmp.at<double>(i, j) = src(i, j)/255;
    cv::imshow("img1", tmp);
    cv::waitKey(ms);
#endif
}

void show_matrices_three_channel(const Eigen::MatrixXd & a, const Eigen::MatrixXd & b, const Eigen::MatrixXd & c, int ms) {
    assert(a.rows() == b.rows() && b.rows() == c.rows() && a.cols() == b.cols() && b.cols() == c.cols());
#ifndef DO_NOT_USE_OPENCV
    cv::Mat tmp(a.rows(), a.cols(), CV_64FC3, cv::Scalar(0, 0, 0));
    for(int i = 0; i < a.rows(); i++) {
        for(int j = 0; j < a.cols(); j++) {
            tmp.at<cv::Vec3d>(i, j)[0] = a(i, j)/255;
            tmp.at<cv::Vec3d>(i, j)[1] = b(i, j)/255;
            tmp.at<cv::Vec3d>(i, j)[2] = c(i, j)/255;
        }
    }
    cv::imshow("three", tmp);
    cv::waitKey(ms);
#endif
}