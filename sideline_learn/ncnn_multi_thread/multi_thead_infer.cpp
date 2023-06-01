#include <iostream>
#include <opencv2/opencv.hpp>

#include "infer.hpp"

using namespace cv;
using namespace std;
using namespace fastdet;

int main()
{
    string param_path =
        "/home/rex/Desktop/ncnn_multi_thread/data/model/FastestDet.param";
    string model_path =
        "/home/rex/Desktop/ncnn_multi_thread/data/model/FastestDet.bin";

    auto infer = create_infer(
        param_path,
        model_path); // 创建及初始化推理器

    if (infer == nullptr)
    {
        printf("Infer is nullptr.\n");
        return 0;
    }

    string img_path = "/home/rex/Desktop/ncnn_multi_thread/data/imgs/3.jpg";
    Mat img = cv::imread(img_path);
    auto fut = infer->commit(img);     // 将任务提交给推理器（推理器执行commit)
    vector<TargetBox> res = fut.get(); // 等待结果

    for (size_t i = 0; i < res.size(); i++)
    {
        TargetBox box = res[i];
        rectangle(img, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2),
                  cv::Scalar(0, 0, 255), 2);
        // cv::putText(img, pred->class_names[box.category], cv::Point(box.x1,
        // box.y1),
        //             cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("result_test.jpg", img);

    return 0;
}
