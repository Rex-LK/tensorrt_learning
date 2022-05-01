
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>
#include <common/ilogger.hpp>
#include <builder/trt_builder.hpp>
#include <common/basic_tools.hpp>
#include<cmath>
using namespace std;
using namespace cv;

struct Pred
{
    vector<vector<float>>bbox;
    vector<int>label;
};

vector<string>classes = {
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

bool build_model(string model){

    if(BaiscTools::exists(model + ".trtmodel")){
        printf("%s.trtmodel has exists.\n",model);
        return true;
    }

    TRT::compile(
        TRT::Mode::FP32,
        10,
        model + ".onnx",
        model + ".trtmodel"
    );
    INFO("Done.");
    return true;
}


void vit_inference(){

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


void detr_inference(){

    auto engine = TRT::load_infer("detr_sim.trtmodel");
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        return;
    }

    engine->print();

    auto input       = engine->input();
    auto output      = engine->output();
    int input_width  = input->width();
    int input_height = input->height();


    auto image = imread("demo.jpg");
    auto img_o = image.clone();
    // imshow("test",image);
    // waitKey(0);
    int img_w = image.cols;
    int img_h = image.rows;
    float mean[] = {0.485, 0.456, 0.406};
    float std[]  = {0.229, 0.224, 0.225};
    resize(image, image, Size(input_width, input_height));
    
    image.convertTo(image, CV_32F);

    Mat channel_based[3];
    for(int i = 0; i < 3; ++i)
        channel_based[i] = Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2-i));

    split(image, channel_based);
    for(int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f - mean[i]) / std[i];

    engine->forward(true);
    float* prob = output->cpu<float>();
    int prob_count = output->count()/output->batch();
    int cols = 95;
    int num_class = 91;
    //这张图有100个结果，每个结果一行 每行95个元素,前91个元素为概率值,后四个为这个结果的框预测框
    //每一行在onnx中已经 softmax, 这里只要取最大值就行,然后取最大值 看是否大于0.7 如果大于0.7，就将这个结果的预测框保存，
    //做完之后将,预测框缩放回原图大小

    float* start = prob;
    Pred pred;
    for(int i = 0;i<=prob_count;i+=95){
        
        //每一行最大的得分的索引
        int label = max_element(start, start + num_class) - start;
        float score = start[label];
        if(score <= 0.7){
            start = prob+i;
            continue;
        }
        else{
            float p_x = start[num_class];
            float p_y = start[num_class+1];

            float p_w = start[num_class+2];
            float p_h = start[num_class+3];
            
            int p_x1 =  (int)((p_x - 0.5 * p_w) * img_w);
            int p_x2 =  (int)((p_x + 0.5 * p_w) * img_w); 
            int p_y1 =  (int)((p_y - 0.5 * p_h) * img_h);
            int p_y2 =  (int)((p_y + 0.5 * p_h) * img_h);
            
            Rect rect(p_x1,p_y1,p_x2 - p_x1,p_y2 - p_y1);
            rectangle(img_o,rect,Scalar(255,0,0),2);
            putText(img_o, classes[label],  Point(p_x1,p_y1+20), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255), 2, 2, 0);
        }
        start = prob+i;
    }
    imwrite("test.jpg",img_o);
}


int main(){
    // string model = "detr_sim";
    // if(!build_model(model)){
    //     return -1;
    // };
    // vit_inference();
    detr_inference();
    return 0;
}