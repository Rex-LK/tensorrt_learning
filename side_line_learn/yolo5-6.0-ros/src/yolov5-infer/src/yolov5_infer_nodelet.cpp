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
        engine_path = params_["engine_path"];
        image_path = params_["image_path"];
        thread_ = thread(boost::bind(&Yolov5InferNodelet::run, this));
    }

    void Yolov5InferNodelet::run()
    {   
        cudaSetDevice(gpu_id);
        infer = new yolov5(engine_path);
        cv::Mat img = cv::imread(image_path);
        int w = img.cols;
        int h = img.rows;
        unsigned char *d_image;
        cudaMalloc((void **)&d_image, sizeof(unsigned char) * w * h * 3);
        cudaMemcpy(d_image, img.data, w * h * 3 * sizeof(unsigned char),cudaMemcpyHostToDevice);
        infer->detect(d_image, w, h,img);
        cv::imshow("show_image", img);
        cv::waitKey(0);
        cudaFree(d_image);
    }
} 
PLUGINLIB_EXPORT_CLASS(yolov5_infer_nodelet::Yolov5InferNodelet,
                       nodelet::Nodelet)