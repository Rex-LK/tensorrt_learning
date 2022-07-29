#include <iostream>
#include <vector>
#include "yolov5_infer_nodelet/yolov5_infer_nodelet.h"
#include "cuda_runtime.h"
#include "dirent.h"

using namespace std;
using namespace cv;

namespace yolov5_infer_nodelet
{
    Yolov5InferNodelet::~Yolov5InferNodelet() {}

    void Yolov5InferNodelet::onInit()
    {
        init();
        gpu_id = stoi(params_["gpu_id"]);
        thread_ = thread(boost::bind(&Yolov5InferNodelet::run, this));
        // roi_engine_name = params_["roi_engine_name"];
        // classNum = stoi(params_["classNum"]);
    }

    void Yolov5InferNodelet::run()
    {
        std::string engine_name = "/home/rex/Desktop/cv_demo/yolov5-6.0-TensorRT/tensorrtx/yolov5s.engine";
        yolov5 *det = new yolov5(engine_name);
        cv::Mat img = cv::imread("/home/rex/Desktop/cv_demo/yolov5-6.0-TensorRT/tensorrtx/images/zidane.jpg");
        int w = img.cols;
        int h = img.rows;
        unsigned char *d_image;
        cudaMalloc((void **)&d_image, sizeof(unsigned char) * w * h * 3);
        cudaMemcpy(d_image, img.data, w * h * 3 * sizeof(unsigned char),cudaMemcpyHostToDevice);
        det->detect(d_image, w, h,img);
        cv::imshow("test_image", img);
        cv::waitKey(0);
        cudaFree(d_image);
    }
} 
PLUGINLIB_EXPORT_CLASS(yolov5_infer_nodelet::Yolov5InferNodelet,
                       nodelet::Nodelet)