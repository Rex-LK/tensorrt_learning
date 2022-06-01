#ifndef DETR_HPP
#define DETR_HPP

#include <string>
#include <future>
#include <memory>
#include<vector>

namespace Detr{
    void detr_inference();
};

struct Pred
{
    std::vector<std::vector<float>>bbox;
    std::vector<int>label;
};




#endif // DETR_HPP