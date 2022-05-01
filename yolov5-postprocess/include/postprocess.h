
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm> 
#ifndef BOX_HPP
#define BOX_HPP

struct Box{
    float left, top, right, bottom, confidence;
    int label;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int label):
    left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label){}
};

#endif // BOX_HPP

 __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy);

__global__ void decode_kernel(
    float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT
);

void decode_kernel_invoker(
    float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    float nms_threshold, float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream);

__device__ float box_iou(
    float aleft, float atop, float aright, float abottom, 
    float bleft, float btop, float bright, float bbottom
);

__global__ void fast_nms_kernel(float* bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT);

std::vector<Box> gpu_decode(float* predict, int rows, int cols, float confidence_threshold, float nms_threshold);

std::vector<Box> gpu_decode(float* predict, int rows, int cols, float confidence_threshold, float nms_threshold);

std::vector<uint8_t> load_file(const std::string& file);


#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);