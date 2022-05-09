#ifndef BaiscTools_HPP
#define BaiscTools_HPP

// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// onnx解析器的头文件
#include <onnx-tensorrt/NvOnnxParser.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

#include <string>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <common/simple-logger.hpp>
#include <builder/trt-builder.hpp>

#define min(a, b)  ((a) < (b) ? (a) : (b))


namespace BaiscTools{
    //判断文件是否存在
    bool exists(const std::string& path);
    // 构造engine
    bool build_model(std::string model,int maxbatch);
    //从.txt加载label
    std::vector<std::string> load_labels(const char* file);
    //cpu 进行warpaffine
    struct affined_img_matrix{
        // i2d 正变换的M矩阵
        cv::Mat m2x3_i2d;
        // d2i M矩阵的逆矩阵 便于恢复图像
        cv::Mat m2x3_d2i;
        //目标图像
        cv::Mat dst_image;
    };

    affined_img_matrix warpaffine_cpu(cv::Mat &ori_img,int dst_height,int dst_weight);


    //gpu warpaffine
    struct MySize{
        int width = 0, height = 0;

        MySize() = default;
        MySize(int w, int h)
        :width(w), height(h){}
    };


    struct AffineMatrix{
        // i2d 正变换的M矩阵
        float i2d[6];
        // d2i M矩阵的逆矩阵
        float d2i[6];
        // 求逆矩阵
        void invertAffineTransform(float imat[6], float omat[6]);

        void compute(const MySize& from, const MySize& to);
    };

    __device__ void affine_project(float* matrix, int x, int y, float* proj_x, float* proj_y);

    __global__ void warp_affine_bilinear_kernel(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
    uint8_t fill_value, AffineMatrix matrix
    );

    void warp_affine_bilinear(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
    uint8_t fill_value
    );

    cv::Mat warpaffine_gpu(const cv::Mat& image,  const int dst_height,const int dst_weight);
    // gpu warpaffine

    std::tuple<cv::Mat, cv::Mat> unet_post_process(float* output, int output_width, int output_height, int num_class, int ibatch);
    void render(cv::Mat& image, const cv::Mat& prob, const cv::Mat& iclass);

    //计算了两个roi的重叠区域
    //region1 为数组的指针
    float box_overlap(float *region1, float *region2);
    
    //获得当前13位时间戳
    long get_current_time();
    //
    // bool string_to_bool(std::string value)
    // {
    //     if (value == "True" || value == "true" || value == "1")
    //     {
    //         return true;
    //     }
    //     else if (value == "False" || value == "false" || value == "0" || value == "")
    //     {
    //         return false;
    //     }
    //     else
    //     {
    //         exit(-1);
    //     }
    // }

    // //将 传入的字符串 "A"  转为 {A}
    // std::vector<std::string> parse_string_v1(std::string &param_string)
    // {
    //     std::vector<std::string> result;

    //     std::istringstream param_string_stream(param_string);
    //     std::string temp;

    //     while (param_string_stream)
    //     {
    //         if (!getline(param_string_stream, temp, ','))
    //             break;
    //         temp.erase(remove(temp.begin(), temp.end(), ' '), temp.end());
    //         result.push_back(temp);
    //     }
    //     return result;
    // }
    // //将 传入的字符串 "A,B"  转为 {A,B}
    // std::vector<std::vector<std::string>> parse_string_v2(std::string &param_string)
    // {
    //     std::vector<std::string> param_list;
    //     std::vector<std::string> param;
    //     std::vector<std::vector<std::string>> result;

    //     std::istringstream param_string_stream(param_string);
    //     std::string temp;
    //     while (param_string_stream)
    //     {
    //         if (!getline(param_string_stream, temp, ';'))
    //             break;
    //         param_list.push_back(temp);
    //     }

    //     for (int i = 0; i < param_list.size(); i++)
    //     {
    //         std::istringstream param_list_stream(param_list[i]);
    //         while (param_list_stream)
    //         {
    //             if (!getline(param_list_stream, temp, ','))
    //                 break;
    //             temp.erase(remove(temp.begin(), temp.end(), ' '), temp.end());
    //             param.push_back(temp);
    //         }
    //         result.push_back(param);
    //         param.clear();
    //     }

    //     param_list.clear();
    //     param.clear();
    //     std::vector<std::string>().swap(param_list);
    //     std::vector<std::string>().swap(param);
    //     return result;
    // }
}


#endif // BaiscTools_HPP