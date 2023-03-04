#ifndef INFER_HPP
#define INFER_HPP

#include <future>
#include <memory>
#include <string>
#include <vector>

#include "fastdet.h"
// 封装接口类
class Infer
{
public:
    virtual std::shared_future<std::vector<fastdet::TargetBox>> commit(
        cv::Mat &input) = 0;
};

std::shared_ptr<Infer> create_infer(const std::string &param_path,
                                    const std::string &model_path);

#endif // INFER_HPP