#include <iostream>
#include <vector>
#include <cv_bridge/cv_bridge.h>
#include "dirent.h"
#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>
#include "yolov5_server_nodelet/yolov5_server_nodelet.h"

using namespace std;
using namespace cv;

namespace yolov5_server_nodelet
{
    Yolov5ServerNodelet::~Yolov5ServerNodelet() {}

    void Yolov5ServerNodelet::onInit()
    {
        init();
        gpu_id = stoi(params_["gpu_id"]);
        std::string engine_name = "/home/rex/Desktop/cv_demo/yolov5-6.0-TensorRT/tensorrtx/yolov5s.engine";
        infer = new yolov5(engine_name);
        ros::ServiceServer server = nh_.advertiseService("yolov5", &Yolov5ServerNodelet::inference, this);
        ros::spin();
    }

    bool Yolov5ServerNodelet::inference(base::RosImage::Request &request,
                      base::RosImage::Response &response)
    {
        ROS_INFO("get_image");
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(request.image, sensor_msgs::image_encodings::TYPE_8UC3);
        cv::Mat img = cv_ptr->image;
        int w = img.cols;
        int h = img.rows;
        cudaMalloc((void **)&d_image, sizeof(unsigned char) * w * h * 3);
        cudaMemcpy(d_image, img.data, w * h * 3 * sizeof(unsigned char),cudaMemcpyHostToDevice);
        bool flag = infer->detect(d_image, w, h, img);
        // cv::imshow("show_image", img);
        // cv::waitKey(0);
        cudaFree(d_image);
        if(flag){
            response.result = "detect target";
        }
        else{
            response.result = "there no target";
        }
        return true;
    }
} 
PLUGINLIB_EXPORT_CLASS(yolov5_server_nodelet::Yolov5ServerNodelet,
                       nodelet::Nodelet)