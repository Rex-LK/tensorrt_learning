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

    auto engine = TRT::load_infer("/home/rex/Desktop/tensorrt_learning/trt_cpp/workspace/yolov5n-seg.trtmodel");
    if(!engine){
        printf("load engine failed \n");
        return;
    }
    auto input       = engine->input();
    auto output      = engine->output();
    auto output1      = engine->output(1);

    int input_width  = input->width();
    int input_height = input->height();
    auto image = imread("/home/rex/Desktop/tensorrt_learning/demo/yolov7/inference/images/street.jpg");
    auto img_o = image.clone();
    int img_w = image.cols;
    int img_h = image.rows;
    auto warp_image = BaiscTools::warpaffine_cpu(image,input_height,input_width);
    auto input_image = warp_image.dst_image;
    auto m2x3_d2i = warp_image.m2x3_d2i;
    float *d2i = m2x3_d2i.ptr<float>();
    input_image.convertTo(input_image, CV_32F);

    Mat channel_based[3];
    for(int i = 0; i < 3; ++i)
        channel_based[i] = Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2-i));

    split(input_image, channel_based);
    for(int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f);
    
    engine->forward(true);
    float *prob = output->cpu<float>();
    float *prob1 = output1->cpu<float>();
    cout << output->shape_string() << endl;
    cout << output1->shape_string() << endl;
    float *predict = prob1;
    int cols = 85;
    int num_classes = cols - 5;
    int rows = 25200;
    vector<yolov5seg_bbox> boxes;
    float confidence_threshold = 0.3;
    float nms_threshold = 0.2;
    for (int i = 0; i < rows; ++i)
    {
        float* pitem = predict + i * cols;
        float objness = pitem[4];

        if(objness < confidence_threshold)
            continue;
        float *pclass = pitem + 5;
        int label     = std::max_element(pclass, pclass + num_classes) - pclass;
        float prob    = pclass[label];
        float confidence = prob * objness;

        if(confidence < confidence_threshold)
            continue;
        float cx = pitem[0];
        float cy     = pitem[1];
        float width  = pitem[2];
        float height = pitem[3];

        float left   = (cx - width * 0.5) * d2i[0] + d2i[2];
        float top    = (cy - height * 0.5) * d2i[0] + d2i[5];
        float right  = (cx + width * 0.5) * d2i[0] + d2i[2];
        float bottom = (cy + height * 0.5) * d2i[0] + d2i[5];
        boxes.emplace_back(left, top, right, bottom, confidence, (float)label);
    }
    std::sort(boxes.begin(), boxes.end(), [](yolov5seg_bbox &a, yolov5seg_bbox &b)
              { return a.confidence > b.confidence; });
    std::vector<bool> remove_flags(boxes.size());
    std::vector<yolov5seg_bbox> box_result;
    box_result.reserve(boxes.size());

    auto iou = [](const yolov5seg_bbox& a, const yolov5seg_bbox& b){
        float cross_left   = std::max(a.left, b.left);
        float cross_top    = std::max(a.top, b.top);
        float cross_right  = std::min(a.right, b.right);
        float cross_bottom = std::min(a.bottom, b.bottom);

        float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
        float union_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top) 
                        + std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top) - cross_area;
        if(cross_area == 0 || union_area == 0) return 0.0f;
        return cross_area / union_area;
    };

    for(int i = 0; i < boxes.size(); ++i){
        if(remove_flags[i]) continue;

        auto& ibox = boxes[i];
        box_result.emplace_back(ibox);
        for(int j = i + 1; j < boxes.size(); ++j){
            if(remove_flags[j]) continue;

            auto& jbox = boxes[j];
            if(ibox.class_label == jbox.class_label){
                // class matched
                if(iou(ibox, jbox) >= nms_threshold)
                    remove_flags[j] = true;
            }
        }
    }

    for(auto& box : box_result){
        cv::rectangle(img_o, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);
        cv::putText(img_o, cv::format("%.2f", box.confidence), cv::Point(box.left, box.top - 7), 0, 0.8, cv::Scalar(0, 0, 255), 2, 16);
    }
    cv::imwrite("yolov7-pred.jpg", img_o);
}