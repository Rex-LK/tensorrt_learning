
#include<infer/trt-infer.hpp>
#include<common/basic_tools.hpp>
#include<common/cuda-tools.hpp>
#include <common/trt-tensor.hpp>
#include <opencv2/opencv.hpp>
#include<demo-infer/unet/unet.h>

using namespace std;
using namespace cv;


vector<int> _classes_colors = {
0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 
128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 
64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 12
};



void Unet::unet_inference(){
    auto engine = TRT::load_infer("/home/rex/Desktop/deeplearning_rex/onnx2trt/workspace/unet.trtmodel");
    if(!engine){
        printf("load engine failed \n");
        return;
    }
    auto input       = engine->input();
    auto output      = engine->output();
    int input_width  = input->width();
    int input_height = input->height();
    int num_classes = 21;

    auto image = cv::imread("/home/rex/Desktop/deeplearning_rex/onnx2trt/workspace/street.jpg");
    auto imge_warpaffined = BaiscTools::warpaffine_cpu(image,input_width,input_height);
    auto input_image = imge_warpaffined.dst_image;
    input_image.convertTo(input_image, CV_32F);
    Mat channel_based[3];
    for(int i = 0; i < 3; ++i)
        channel_based[i] = Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2-i));

    split(input_image, channel_based);
    for(int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f);

    engine->forward(true);
    float* prob = output->cpu<float>();
    cv::Mat unet_prob, iclass;
    tie(unet_prob, iclass) = post_process(prob, 512, 512, num_classes, 0);
    cv::warpAffine(unet_prob, unet_prob, imge_warpaffined.m2x3_d2i, image.size(), cv::INTER_LINEAR);
    cv::warpAffine(iclass, iclass, imge_warpaffined.m2x3_d2i, image.size(), cv::INTER_NEAREST);
    render(image, unet_prob, iclass);
    printf("Done, Save to image-draw.jpg\n");
    cv::imwrite("unet-pred.jpg", image);
}



void Unet::render(cv::Mat& image, const cv::Mat& prob, const cv::Mat& iclass){

    auto pimage = image.ptr<cv::Vec3b>(0);
    auto pprob  = prob.ptr<float>(0);
    auto pclass = iclass.ptr<uint8_t>(0);
    //0~512*512
    for(int i = 0; i < image.cols*image.rows; ++i, ++pimage, ++pprob, ++pclass){

        int iclass        = *pclass;
        float probability = *pprob;
        auto& pixel       = *pimage;
        float foreground  = min(0.6f + probability * 0.2f, 0.8f);
        float background  = 1 - foreground;
        for(int c = 0; c < 3; ++c){
            auto value = pixel[c] * background + foreground * _classes_colors[iclass * 3 + 2-c];
            pixel[c] = min((int)value, 255);
        }
    }
}

tuple<cv::Mat, cv::Mat> Unet::post_process(float* output, int output_width, int output_height, int num_class, int ibatch){
    // output 1*(numclass)*512*512）
    cv::Mat output_prob(output_height, output_width, CV_32F);
    cv::Mat output_index(output_height, output_width, CV_8U);
    //从第几个batch开始
    //每次加一个numclass 重复512*512次
    float* pnet   = output + ibatch * output_width * output_height * num_class;
    float* prob   = output_prob.ptr<float>(0);
    uint8_t* pidx = output_index.ptr<uint8_t>(0);

    for(int k = 0; k < output_prob.cols * output_prob.rows; ++k, pnet+=num_class, ++prob, ++pidx){
        //找到num_class中得分最大的值
        int ic = std::max_element(pnet, pnet + num_class) - pnet;
        *prob  = pnet[ic];
        *pidx  = ic;
    }
    return make_tuple(output_prob, output_index);
}