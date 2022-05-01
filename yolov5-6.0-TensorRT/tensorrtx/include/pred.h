#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "yololayer.h"
#include "basic_transform.h"
#define USE_FP16 // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0 // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1; // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char *INPUT_BLOB_NAME = "data";
static Logger gLogger;

class yolov5
{
public:
    yolov5(std::string engine_name, int ClassNum)
    {
        m_ClassNum = ClassNum;
        this->engine_name = engine_name;
        std::ifstream file(engine_name, std::ios::binary);
        size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        this->trtModelStream = new char[size];
        assert(this->trtModelStream);
        file.read(this->trtModelStream, size);
        file.close();
        std::vector<std::string> file_names;

        this->runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        this->engine = runtime->deserializeCudaEngine(trtModelStream, size);
        assert(engine != nullptr);
        this->context = engine->createExecutionContext();
        assert(context != nullptr);
        this->m_InputBindingIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        delete[] trtModelStream;

        CUDA_CHECK(cudaMalloc(&buffers[0], 3 * INPUT_H * INPUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[1], OUTPUT_SIZE * sizeof(float)));

        CUDA_CHECK(cudaMalloc((void **)&d_resized_img, 3 * 640 * 640 * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc((void **)&d_norm_img, sizeof(float) * 640 * 640 * 3));
    }
    ~yolov5()
    {
        context->destroy();
        engine->destroy();
        runtime->destroy();

        CUDA_CHECK(cudaFree(buffers[0]));
        CUDA_CHECK(cudaFree(buffers[1]));
    }
    void detect(unsigned char *d_roi_image, int roi_w, int roi_h, cv::Mat &img)
    {
        float image_ratio =
            roi_w > roi_h ? float(640) / float(roi_w) : float(640) / float(roi_h);
        int width_out = roi_w > roi_h ? 640 : (int)(roi_w * image_ratio);
        int height_out = roi_w < roi_h ? 640 : (int)(roi_h * image_ratio);
        cudaMemset(d_resized_img, 0, sizeof(unsigned char) * 640 * 640 * 3);
        RGB2Resize(d_roi_image, d_resized_img, roi_w, roi_h, width_out, height_out);
        RGB2Normalize(d_resized_img, d_norm_img, 640, 640);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        buffers[0] = d_norm_img;
        this->context->enqueue(1, this->buffers, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(this->out_put, this->buffers[1], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        std::vector<std::vector<Yolo::Detection>> batch_res(1);
        std::vector<Yolo::Detection> &res = batch_res[0];
        nms(res, &out_put[0], CONF_THRESH, NMS_THRESH);

        for (size_t j = 0; j < res.size(); j++)
        {
            cv::Rect r = get_rect(roi_w, roi_h, res[j].bbox);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 1);
            cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y + 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
    };

private:
    std::string engine_name;
    IRuntime *runtime;
    ICudaEngine *engine;
    IExecutionContext *context;
    char *trtModelStream = nullptr;
    cudaStream_t m_CudaStream;
    int m_InputBindingIndex;
    std::vector<void *> m_DeviceBuffers;
    float out_put[OUTPUT_SIZE];
    void *buffers[2];
    int m_ClassNum;

    unsigned char *d_resized_img;
    float *d_norm_img;
};