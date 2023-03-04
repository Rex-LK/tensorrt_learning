#include "fastdet.h"

#include <iostream>
#include <string>
using namespace std;
using namespace fastdet;

int main() {
  string img_path = "/data/rex/ncnn_proj/ncnn_multi_thread/data/imgs/3.jpg";
  cv::Mat img = cv::imread(img_path);
  int img_width = img.cols;
  int img_height = img.rows;
  string param_path =
      "/data/rex/ncnn_proj/ncnn_multi_thread/data/model/FastestDet.param";
  string model_path =
      "/data/rex/ncnn_proj/ncnn_multi_thread/data/model/FastestDet.bin";
  FastDet* pred = new FastDet(352, 352, param_path, model_path);
  int class_num = sizeof(pred->class_names) / sizeof(pred->class_names[0]);
  pred->prepare_input(img);
  pred->infrence("input.1", "758", 6);
  pred->postprocess(img_width, img_height, class_num, 0.65);
  for (size_t i = 0; i < pred->nms_boxes.size(); i++) {
    TargetBox box = pred->nms_boxes[i];
    cv::rectangle(img, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2),
                  cv::Scalar(0, 0, 255), 2);
    cv::putText(img, pred->class_names[box.category], cv::Point(box.x1, box.y1),
                cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2);
  }
  cv::imwrite("result_test.jpg", img);
  return 0;
}

// 单模型 多视频
// 多模型 单视频
// 多模型 多视频