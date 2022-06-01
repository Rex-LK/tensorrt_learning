#include<infer/trt-infer.hpp>
#include<common/basic_tools.hpp>
#include<common/cuda-tools.hpp>
#include <common/trt-tensor.hpp>
#include <opencv2/opencv.hpp>
#include<demo-infer/hrnet/hrnet.h>

using namespace std;
using namespace cv;

void Hrnet::hrnet_inference(){
    //1*17 *64*48
    //将图片分成64*48个点
    //分别预测64*48个点中 17类的概率
    //论文中将最后得到的点 偏移了一些

    auto engine = TRT::load_infer("hrnet.trtmodel");
    if(!engine){
        printf("load engine failed \n");
        return;
    }
    auto input       = engine->input();
    auto output      = engine->output();
    int input_width  = input->width();
    int input_height = input->height();
    float mean[] = {0.485, 0.456, 0.406};
    float std[]  = {0.229, 0.224, 0.225};
    auto image = imread("person2.jpeg");
    auto img_o = image.clone();
    int img_w = image.cols;
    int img_h = image.rows;
    auto warp_image = BaiscTools::warpaffine_cpu(image,input_height,input_width);
    auto input_image = warp_image.dst_image;
    auto m2x3_d2i = warp_image.m2x3_d2i;
    float* d2i = m2x3_d2i.ptr<float>();
    // cv::imwrite("warp-affine.jpg", input_image);
    input_image.convertTo(input_image, CV_32F);

    Mat channel_based[3];
    for(int i = 0; i < 3; ++i)
        channel_based[i] = Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2-i));

    split(input_image, channel_based);
    for(int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f - mean[i]) / std[i];

    engine->forward(true);
    float* prob = output->cpu<float>();
    float* start = prob;
    int nums = 17;
    int pic_region = 3072;
    for(int i=0;i<output->count();i+=pic_region){
        start = prob + i;
        int label = (max_element(start,start+pic_region) - start);
        float score = start[label];
        if(score<0.2)
            continue;
        float x = label % 48 ;
        float y = label / 48 ;
        //特征图是原图的1/4
        int x_o = (x * d2i[0] * 4) + d2i[2];
        int y_o = (y * d2i[4] * 4) + d2i[5]; 
        cv::circle(img_o, cv::Point((int)x_o,(int)y_o), 1, (0, 0, 255), 2);
        cv::imwrite("hrnet-pred.jpg", img_o);
    }
}