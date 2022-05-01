
#include <cuda_runtime.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#define min(a, b)  ((a) < (b) ? (a) : (b))
#define num_threads   512

typedef unsigned char uint8_t;

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

cv::Mat warpaffine_to_center_align(const cv::Mat& image, const cv::Size& size);

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);