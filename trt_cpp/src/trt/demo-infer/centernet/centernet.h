#ifndef CENTERNET_HPP
#define CENTERNET_HPP

#include <string>
#include <future>
#include <memory>

struct bbox{
    float left, top, right, bottom, confidence;
    int label;

    bbox() = default;
    bbox(float left, float top, float right, float bottom, float confidence, int label):
    left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label){}
};

namespace Centernet{
    void centernet_inference();
    void centernet_inference_gpu();
};




#endif // CENTERNET_HPP