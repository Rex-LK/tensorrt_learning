#ifndef YOLOV5_SERVER_NODELET_H
#define YOLOV5_SERVER_NODELET_H

#include <string>
#include <opencv2/opencv.hpp>
#include <std_msgs/String.h>

#include "base/base.h"
#include "base/RosImage.h"
#include "yolov5-detect.h"

using namespace cv;
using namespace std;

namespace yolov5_server_nodelet
{
    class Yolov5ServerNodelet : public base::Base
    {
    public:
        ~Yolov5ServerNodelet();

        virtual void onInit();
        void run();

    private:
        bool inference(base::RosImage::Request &request,
                      base::RosImage::Response &response);

        int gpu_id;
        unsigned char *d_image;
        yolov5 *infer;
    };
} 

#endif // YOLOV5_SERVER_NODELET_H