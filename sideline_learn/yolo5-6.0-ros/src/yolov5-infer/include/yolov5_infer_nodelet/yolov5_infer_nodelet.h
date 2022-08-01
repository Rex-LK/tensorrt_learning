#ifndef YOLOV5_INFER_NODELET_H
#define YOLOV5_INFER_NODELET_H

#include <std_msgs/String.h>
#include <string>
#include <thread>
#include <string>
#include "base/base.h"
#include <opencv2/opencv.hpp>
#include "yolov5-detect.h"

using namespace cv;
using namespace std;

namespace yolov5_infer_nodelet
{
    class Yolov5InferNodelet : public base::Base
    {
    public:
        ~Yolov5InferNodelet();

        virtual void onInit();

    private:
        void run();

    private:
        int gpu_id;
        string engine_path;
        string image_path;
        thread thread_;
        yolov5 *infer;
    };
} 

#endif // YOLOV5_INFER_NODELET_H