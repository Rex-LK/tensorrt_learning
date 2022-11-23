#include<infer/trt-infer.hpp>
#include<common/basic_tools.hpp>
#include<common/cuda-tools.hpp>
#include <common/trt-tensor.hpp>
#include <opencv2/opencv.hpp>
#include<demo-infer/yolov5seg/yolov5seg.h>

using namespace std;
using namespace cv;


struct yolov5seg_bbox{
    float left, top, right, bottom, confidence;
    int class_label;

    yolov5seg_bbox() = default;

    yolov5seg_bbox(float left, float top, float right, float bottom, float confidence, int class_label)
    :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label){}
};


void Yolov5Seg::yolov5Seg_inference(){

    auto engine = TRT::load_infer("/home/rex/Desktop/tensorrt_learning/trt_cpp/workspace/yolov7.trtmodel");
    if(!engine){
        printf("load engine failed \n");
        return;
    }
    auto input       = engine->input();
    auto output      = engine->output();
    int input_width  = input->width();
    int input_height = input->height();
    auto image = imread("/home/rex/Desktop/tensorrt_learning/demo/yolov7/inference/images/street.jpg");
    auto img_o = image.clone();
    int img_w = image.cols;
    int img_h = image.rows;
    auto warp_image = BaiscTools::warpaffine_cpu(image,input_height,input_width);
    auto input_image = warp_image.dst_image;
    auto m2x3_d2i = warp_image.m2x3_d2i;
    cout << m2x3_d2i << endl;
    float *d2i = m2x3_d2i.ptr<float>();
    input_image.convertTo(input_image, CV_32F);

    Mat channel_based[3];
    for(int i = 0; i < 3; ++i)
        channel_based[i] = Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2-i));

    split(input_image, channel_based);
    for(int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f);
    
    engine->forward(true);
    cout << output->shape_string() << endl;
    float *prob = output->cpu<float>();

}