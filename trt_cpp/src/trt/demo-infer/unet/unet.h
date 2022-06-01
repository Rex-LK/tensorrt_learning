#ifndef UNET_HPP
#define UNET_HPP
#include<iostream>
#include <opencv2/opencv.hpp>

namespace Unet{
    void unet_inference();
    void render(cv::Mat& image, const cv::Mat& prob, const cv::Mat& iclass);
    std::tuple<cv::Mat, cv::Mat> post_process(float* output, int output_width, int output_height, int num_class, int ibatch);
};

#endif // UNET_HPP