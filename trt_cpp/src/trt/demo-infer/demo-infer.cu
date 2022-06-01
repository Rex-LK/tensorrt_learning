#include <demo-infer/yolov5/yolov5.hpp>
#include<infer/trt-infer.hpp>
#include<common/basic_tools.hpp>
#include<demo-infer/demo-infer.hpp>
#include<common/cuda-tools.hpp>
#include <common/trt-tensor.hpp>

using namespace cv;
using namespace std;
static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

void demoInfer::yolov5_inference(){

    auto image = cv::imread("rq.jpg");
    auto yolov5 = YoloV5::create_infer("yolov5s.trtmodel");
    auto boxes = yolov5->commit(image).get();
    for(auto& box : boxes){
        cv::Scalar color(0, 255, 0);
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);

        auto name      = cocolabels[box.class_label];
        auto caption   = cv::format("%s %.2f", name, box.confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(box.left-3, box.top-33), cv::Point(box.left + text_width, box.top), color, -1);
        cv::putText(image, caption, cv::Point(box.left, box.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    cv::imwrite("yolov5-image-draw.jpg", image);
}


void load_from_py(){
    //用于保存模型推理之后confidence大于阈值的框
    int input_h = 1330;
    int input_w = 1330;
    vector<bbox> boxes;
    TRT::Tensor tensor;
    tensor.load_from_file("pred.tensor");
    float* prob = tensor.cpu<float>();
    int num = 20;  
    int count = tensor.count();
    float *start = prob;
    for(int i=0;i<count;i+=24){
        //现在有128*128个点  就有128*128行，每行24个，前20个为类别，后四个为 w,h,x,y
        start = prob+i;
        int label = max_element(start,start+num) - start;
        float confidence = start[label];
        if(confidence<0.3)
            continue;
        float w = start[ num];
        float h = start[num+1];
        //这里的x,y 还要加上偏移量
        float x = start[num+2] + (i/24)%128;
        float y = start[num+3] + (i/24)/128;
        float left   = (x - w * 0.5) /128 * input_h;
        float top    = (y - h * 0.5) /128 * input_w;
        float right  = (x + w * 0.5) /128 * input_h;
        float bottom = (y + h * 0.5) /128 * input_w;
        boxes.emplace_back(left, top, right, bottom, confidence, (float)label);
    }
    //nms
    sort(boxes.begin(), boxes.end(), [](bbox& a, bbox& b){return a.confidence > b.confidence;});
    //然后使用vector<bool>来标记是是否要删除box
    vector<bool> remove_flags(boxes.size());
    vector<bbox> box_result;
    box_result.reserve(boxes.size());

    //计算两个box之间的IOU
    auto iou = [](const bbox& a, const bbox& b){
        float cross_left   = max(a.left, b.left);
        float cross_top    = max(a.top, b.top);
        float cross_right  = min(a.right, b.right);
        float cross_bottom = min(a.bottom, b.bottom);
        //计算重叠部分的面积,注意面积非0
        float cross_area = max(0.0f, cross_right - cross_left) * max(0.0f, cross_bottom - cross_top);
        //A面积+B面积 - 一个重叠面积
        float union_area = max(0.0f, a.right - a.left) * max(0.0f, a.bottom - a.top) 
                         + max(0.0f, b.right - b.left) * max(0.0f, b.bottom - b.top) - cross_area;
        if(cross_area == 0 || union_area == 0) return 0.0f;
        return cross_area/union_area;
    };

    for(int i = 0;i<boxes.size();i++){
        if(remove_flags[i]) continue;
        auto &ibox  = boxes[i];
        //第一个box必定不会remove
        box_result.emplace_back(ibox);
        //第一个box与 之后的box两两比较
        for(int j = i + 1;j<boxes.size();j++){
            if(remove_flags[j]) continue;
            auto& jbox = boxes[j];
            //如果两个box的lable相同
            if(ibox.label == jbox.label){
                //则比较IOU
                if(iou(ibox, jbox) >= 0.3)
                    remove_flags[j] = true;
            }
        }
    }
    Mat image = imread("street.jpg");
    for(auto& box : box_result){
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);
        cv::putText(image, cv::format("%.2f", box.confidence), cv::Point(box.left, box.top - 7), 0, 0.8, cv::Scalar(0, 0, 255), 2, 16);
    }
    cv::imwrite("centernet-pred.jpg", image);
}



















