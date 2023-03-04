#ifndef __FASTDET_H_
#define __FASTDET_H_
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "net.h"
namespace fastdet
{

    // bboxes
    class TargetBox
    {
    private:
        float GetWidth() { return (x2 - x1); };
        float GetHeight() { return (y2 - y1); };

    public:
        int x1;
        int y1;
        int x2;
        int y2;

        int category;
        float score;

        float area() { return GetWidth() * GetHeight(); };
    };

    float Sigmoid(float x);
    float Tanh(float x);
    float IntersectionArea(const TargetBox &a, const TargetBox &b);
    bool scoreSort(TargetBox a, TargetBox b);
    int nmsHandle(std::vector<TargetBox> &src_boxes,
                  std::vector<TargetBox> &dst_boxes);
    class FastDet
    {
    public:
        FastDet(int input_width, int input_height, std::string param_path,
                std::string model_path);
        ~FastDet();
        void prepare_input(cv::Mat img);
        void infrence(std::string inputName, std::string outputName, int num_threads);
        void postprocess(int img_width, int img_height, int class_num, float thresh);

    public:
        static const char *class_names[];
        std::vector<TargetBox> target_boxes;
        std::vector<TargetBox> nms_boxes;

    private:
        // 模型
        ncnn::Net net_;
        int input_width_;
        int input_height_;
        ncnn::Mat input_;
        ncnn::Mat output_;
        const float mean_vals_[3] = {0.f, 0.f, 0.f};
        const float norm_vals_[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    };
} // namespace fastdet
#endif //__FASTDET_H_

const char *fastdet::FastDet::class_names[] = {
    "person", "bicycle", "car",
    "motorcycle", "airplane", "bus",
    "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird",
    "cat", "dog", "horse",
    "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup",
    "fork", "knife", "spoon",
    "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza",
    "donut", "cake", "chair",
    "couch", "potted plant", "bed",
    "dining table", "toilet", "tv",
    "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink",
    "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};