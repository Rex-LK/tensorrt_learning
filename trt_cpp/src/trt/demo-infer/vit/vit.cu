
#include<infer/trt-infer.hpp>
#include<common/basic_tools.hpp>
#include<common/cuda-tools.hpp>
#include <common/trt-tensor.hpp>
#include <opencv2/opencv.hpp>
#include<demo-infer/vit/vit.h>

using namespace cv;
using namespace std;

void Vit::vit_inference(){

    auto engine = TRT::load_infer("vit.trtmodel");
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        return;
    }

    engine->print();

    auto input       = engine->input();
    auto output      = engine->output();
    int input_width  = input->width();
    int input_height = input->height();

    auto image = imread("flower.jpg");
    float mean[] = {0.5, 0.5, 0.5};
    float std[]  = {0.5, 0.5, 0.5};

    resize(image, image, Size(input_width, input_height));
    image.convertTo(image, CV_32F);

    Mat channel_based[3];
    //channel_based 为{}

    for(int i = 0; i < 3; ++i)
        channel_based[i] = Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2-i));

    split(image, channel_based);
    for(int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f - mean[i]) / std[i];

    engine->forward(true);
    int num_classes   = output->size(1);
    float* prob       = output->cpu<float>();
    int predict_label = std::max_element(prob, prob + num_classes) - prob;
    auto labels       = BaiscTools::load_labels("labels.imagenet.txt");
    auto predict_name = labels[predict_label];
    float confidence  = prob[predict_label];
    printf("Predict: %s, confidence = %f, label = %d\n", predict_name.c_str(), confidence, predict_label);
}