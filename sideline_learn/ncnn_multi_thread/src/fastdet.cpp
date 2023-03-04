#include "fastdet.h"

#include <vector>
using namespace std;

float fastdet::Sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
float fastdet::Tanh(float x) { return 2.0f / (1.0f + exp(-2 * x)) - 1; }
float fastdet::IntersectionArea(const TargetBox &a, const TargetBox &b) {
  if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1) {
    // no intersection
    return 0.f;
  }

  float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

  return inter_width * inter_height;
}
bool fastdet::scoreSort(fastdet::TargetBox a, fastdet::TargetBox b) {
  return (a.score > b.score);
}

int fastdet::nmsHandle(std::vector<fastdet::TargetBox> &src_boxes,
                       std::vector<fastdet::TargetBox> &dst_boxes) {
  std::vector<int> picked;

  sort(src_boxes.begin(), src_boxes.end(), scoreSort);

  for (int i = 0; i < src_boxes.size(); i++) {
    int keep = 1;
    for (int j = 0; j < picked.size(); j++) {
      // 交集
      float inter_area = IntersectionArea(src_boxes[i], src_boxes[picked[j]]);
      // 并集
      float union_area =
          src_boxes[i].area() + src_boxes[picked[j]].area() - inter_area;
      float IoU = inter_area / union_area;

      if (IoU > 0.45 &&
          src_boxes[i].category == src_boxes[picked[j]].category) {
        keep = 0;
        break;
      }
    }

    if (keep) {
      picked.push_back(i);
    }
  }

  for (int i = 0; i < picked.size(); i++) {
    dst_boxes.push_back(src_boxes[picked[i]]);
  }

  return 0;
}

fastdet::FastDet::FastDet(int input_width, int input_height,
                          std::string param_path, std::string model_path) {
  this->input_width_ = input_width;
  this->input_height_ = input_height;

  net_.load_param(param_path.c_str());
  net_.load_model(model_path.c_str());
  printf("ncnn model load sucess...\n");
}

fastdet::FastDet::~FastDet() {}

void fastdet::FastDet::prepare_input(cv::Mat img) {
  input_ =
      ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, img.cols,
                                    img.rows, input_width_, input_height_);
  // Normalization of input image data
  input_.substract_mean_normalize(mean_vals_, norm_vals_);
}

void fastdet::FastDet::infrence(std::string inputName, std::string outputName,
                                int num_threads) {
  ncnn::Extractor ex = net_.create_extractor();
  cout << "inputName:" << inputName << endl;
  cout << "outputName:" << outputName << endl;
  ex.set_num_threads(num_threads);
  ex.input(inputName.c_str(), input_);
  ex.extract(outputName.c_str(), output_);
}

void fastdet::FastDet::postprocess(int img_width, int img_height, int class_num,
                                   float thresh) {
  target_boxes.clear();
  for (int h = 0; h < output_.h; h++) {
    for (int w = 0; w < output_.h; w++) {
      // 前景概率
      int obj_score_index = (0 * output_.h * output_.w) + (h * output_.w) + w;
      float obj_score = output_[obj_score_index];

      // 解析类别
      int category;
      float max_score = 0.0f;
      for (size_t i = 0; i < class_num; i++) {
        int obj_score_index =
            ((5 + i) * output_.h * output_.w) + (h * output_.w) + w;
        float cls_score = output_[obj_score_index];
        if (cls_score > max_score) {
          max_score = cls_score;
          category = i;
        }
      }
      float score = pow(max_score, 0.4) * pow(obj_score, 0.6);

      // 阈值筛选
      if (score > thresh) {
        // 解析坐标
        int x_offset_index = (1 * output_.h * output_.w) + (h * output_.w) + w;
        int y_offset_index = (2 * output_.h * output_.w) + (h * output_.w) + w;
        int box_width_index = (3 * output_.h * output_.w) + (h * output_.w) + w;
        int box_height_index =
            (4 * output_.h * output_.w) + (h * output_.w) + w;

        float x_offset = Tanh(output_[x_offset_index]);
        float y_offset = Tanh(output_[y_offset_index]);
        float box_width = Sigmoid(output_[box_width_index]);
        float box_height = Sigmoid(output_[box_height_index]);

        float cx = (w + x_offset) / output_.w;
        float cy = (h + y_offset) / output_.h;

        int x1 = (int)((cx - box_width * 0.5) * img_width);
        int y1 = (int)((cy - box_height * 0.5) * img_height);
        int x2 = (int)((cx + box_width * 0.5) * img_width);
        int y2 = (int)((cy + box_height * 0.5) * img_height);

        target_boxes.push_back(TargetBox{x1, y1, x2, y2, category, score});
      }
    }
  }
  // NMS处理
  nms_boxes.clear();
  nmsHandle(target_boxes, nms_boxes);
}
