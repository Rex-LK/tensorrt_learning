#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "preprocess.h"
#include "yolov5-detect.h"

int main(int argc, char **argv)
{
    cudaSetDevice(0);

    std::string engine_name = "yolov5s.engine";
    yolov5 *det = new yolov5(engine_name);

    cv::Mat img = cv::imread("1.jpg");
    int w = img.cols;
    int h = img.rows;
    unsigned char *d_image;
    cudaMalloc((void **)&d_image, sizeof(unsigned char) * w * h * 3);
    cudaMemcpy(d_image, img.data, w * h * 3 * sizeof(unsigned char),cudaMemcpyHostToDevice);

    bool flag = det->detect(d_image, w, h,img);

    cudaFree(d_image);
    return 0;
}
