
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include "postprocess.h"
using namespace std;
using namespace cv;

int main(){

    //data vector<uint8_t>
    auto data = load_file("/home/rex/Desktop/deeplearning_rex/yolov5-postprocess/predict.data");
    auto image = cv::imread("/home/rex/Desktop/deeplearning_rex/yolov5-postprocess/input-image.jpg");
    // ptr 指向 data的第一个元素位置
    float* ptr = (float*)data.data();
    
    int nelem = data.size() / sizeof(float);
    
    // 类别数量
    int ncols = 85;
    int nrows = nelem / ncols;
    // 85 * 25200 二维矩阵


    auto boxes = cpu_decode(ptr, nrows, ncols,0.25f,0.45f);
    
    for(auto& box : boxes){
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);
        cv::putText(image, cv::format("%.2f", box.confidence), cv::Point(box.left, box.top - 7), 0, 0.8, cv::Scalar(0, 0, 255), 2, 16);
    }
    

    cv::imwrite("image-draw.jpg", image);
    return 0;
}