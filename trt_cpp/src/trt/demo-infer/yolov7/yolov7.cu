#include<infer/trt-infer.hpp>
#include<common/basic_tools.hpp>
#include<common/cuda-tools.hpp>
#include <common/trt-tensor.hpp>
#include <opencv2/opencv.hpp>
#include<demo-infer/yolov7/yolov7.h>

using namespace std;
using namespace cv;


struct yolov7_bbox{
    float left, top, right, bottom, confidence;
    int class_label;

    yolov7_bbox() = default;

    yolov7_bbox(float left, float top, float right, float bottom, float confidence, int class_label)
    :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label){}
};


void Yolov7::yolov7_inference(){

    auto engine = TRT::load_infer("/home/rex/Desktop/tensorrt_learning/trt_cpp/workspace/yolov7.trtmodel");
    if(!engine){
        printf("load engine failed \n");
        return;
    }
    auto input       = engine->input();
    auto output      = engine->output();
    int input_width  = input->width();
    int input_height = input->height();
    auto image = imread("/home/rex/Desktop/tensorrt_learning/demo/yolov7/inference/images/street.jpg");
    auto img_o = image.clone();
    int img_w = image.cols;
    int img_h = image.rows;
    auto warp_image = BaiscTools::warpaffine_cpu(image,input_height,input_width);
    auto input_image = warp_image.dst_image;
    auto m2x3_d2i = warp_image.m2x3_d2i;
    cout << m2x3_d2i << endl;
    float *d2i = m2x3_d2i.ptr<float>();
    input_image.convertTo(input_image, CV_32F);

    Mat channel_based[3];
    for(int i = 0; i < 3; ++i)
        channel_based[i] = Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2-i));

    split(input_image, channel_based);
    for(int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f);
    
    engine->forward(true);
    cout << output->shape_string() << endl;
    float *prob = output->cpu<float>();
    float *predict = prob;
    int cols = 85;
    int num_classes = cols - 5;
    int rows = 25200;
    vector<yolov7_bbox> boxes;
    float confidence_threshold = 0.3;
    float nms_threshold = 0.2;
    for (int i = 0; i < rows; ++i)
    {
        float* pitem = predict + i * cols;
        float objness = pitem[4];
        if(objness < confidence_threshold)
            continue;

        float* pclass = pitem + 5;
        int label     = std::max_element(pclass, pclass + num_classes) - pclass;
        float prob    = pclass[label];
        float confidence = prob * objness;
        if(confidence < confidence_threshold)
            continue;

        float cx     = pitem[0];
        float cy     = pitem[1];
        float width  = pitem[2];
        float height = pitem[3];

        float left   = (cx - width * 0.5) * d2i[0] + d2i[2];
        float top    = (cy - height * 0.5) * d2i[0] + d2i[5];
        float right  = (cx + width * 0.5) * d2i[0] + d2i[2];
        float bottom = (cy + height * 0.5) * d2i[0] + d2i[5];
        boxes.emplace_back(left, top, right, bottom, confidence, (float)label);
    }
    std::sort(boxes.begin(), boxes.end(), [](yolov7_bbox &a, yolov7_bbox &b)
              { return a.confidence > b.confidence; });
    std::vector<bool> remove_flags(boxes.size());
    std::vector<yolov7_bbox> box_result;
    box_result.reserve(boxes.size());

    auto iou = [](const yolov7_bbox& a, const yolov7_bbox& b){
        float cross_left   = std::max(a.left, b.left);
        float cross_top    = std::max(a.top, b.top);
        float cross_right  = std::min(a.right, b.right);
        float cross_bottom = std::min(a.bottom, b.bottom);

        float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
        float union_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top) 
                        + std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top) - cross_area;
        if(cross_area == 0 || union_area == 0) return 0.0f;
        return cross_area / union_area;
    };

    for(int i = 0; i < boxes.size(); ++i){
        if(remove_flags[i]) continue;

        auto& ibox = boxes[i];
        box_result.emplace_back(ibox);
        for(int j = i + 1; j < boxes.size(); ++j){
            if(remove_flags[j]) continue;

            auto& jbox = boxes[j];
            if(ibox.class_label == jbox.class_label){
                // class matched
                if(iou(ibox, jbox) >= nms_threshold)
                    remove_flags[j] = true;
            }
        }
    }

    for(auto& box : box_result){
        cv::rectangle(img_o, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);
        cv::putText(img_o, cv::format("%.2f", box.confidence), cv::Point(box.left, box.top - 7), 0, 0.8, cv::Scalar(0, 0, 255), 2, 16);
    }
    cv::imwrite("yolov7-pred.jpg", img_o);
}







static __global__ void decode_kernel(
    float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT
){  
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float* pitem     = predict + (5 + num_classes) * position;
    float objectness = pitem[4];
    if(objectness < confidence_threshold)
        return;

    float* class_confidence = pitem + 5;
    float confidence        = *class_confidence++;
    int label               = 0;
    for(int i = 1; i < num_classes; ++i, ++class_confidence){
        if(*class_confidence > confidence){
            confidence = *class_confidence;
            label      = i;
        }
    }

    confidence *= objectness;
    if(confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if(index >= max_objects)
        return;
    float cx = *pitem++;
    float cy         = *pitem++;
    float width      = *pitem++;
    float height     = *pitem++;

    float left   = (cx - width * 0.5f)*invert_affine_matrix[0] + invert_affine_matrix[2];
    float top    = (cy - height * 0.5f)* invert_affine_matrix[0] + invert_affine_matrix[5];
    float right  = (cx + width * 0.5f)* invert_affine_matrix[0] + invert_affine_matrix[5];
    float bottom = (cy + height * 0.5f)* invert_affine_matrix[0] + invert_affine_matrix[5];


    // left, top, right, bottom, confidence, class, keepflag
    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
}

static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom, 
    float bleft, float btop, float bright, float bbottom
){

    float cleft 	= max(aleft, bleft);
    float ctop 		= max(atop, btop);
    float cright 	= min(aright, bright);
    float cbottom 	= min(abottom, bbottom);
    
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if(c_area == 0.0f)
        return 0.0f;
    
    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float* bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT){

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, max_objects);
    if (position >= count) 
        return;
    
    // left, top, right, bottom, confidence, class, keepflag
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for(int i = 0; i < count; ++i){
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if(i == position || pcurrent[5] != pitem[5]) continue;
    
        if(pitem[4] >= pcurrent[4]){
            if(pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0],    pitem[1],    pitem[2],    pitem[3]
            );

            if(iou > threshold){
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
} 


void decode_kernel_invoker(
    float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    float nms_threshold, float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream){
    
    auto block = num_bboxes > 512 ? 512 : num_bboxes;
    auto grid = (num_bboxes + block - 1) / block;
    decode_kernel<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, confidence_threshold,
        invert_affine_matrix, parray, max_objects, NUM_BOX_ELEMENT);

    block = max_objects > 512 ? 512 : max_objects;
    grid = (max_objects + block - 1) / block;
    fast_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold, NUM_BOX_ELEMENT);
}




vector<yolov7_bbox> gpu_decode(float* predict, int rows, int cols, float confidence_threshold,float nms_threshold,Mat m2x3_d2i){
    
    vector<yolov7_bbox> box_result;
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    float* predict_device = nullptr;
    float* output_device = nullptr;
    float* output_host = nullptr;
    float* affine_device = nullptr;
    int max_objects = 1000;
    int NUM_BOX_ELEMENT = 7;  // left, top, right, bottom, confidence, class, keepflag
    checkRuntime(cudaMalloc(&predict_device, rows * cols * sizeof(float)));
    checkRuntime(cudaMalloc(&output_device, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    checkRuntime(cudaMallocHost(&output_host, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    checkRuntime(cudaMalloc(&affine_device, 6 * sizeof(float)));
    checkRuntime(cudaMemcpyAsync(affine_device, m2x3_d2i.data, 6 * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaMemcpyAsync(predict_device, predict, rows * cols * sizeof(float), cudaMemcpyHostToDevice, stream));
    decode_kernel_invoker(
        predict_device, rows, cols - 5, confidence_threshold, 
        nms_threshold, affine_device, output_device, max_objects, NUM_BOX_ELEMENT, stream
    );
    checkRuntime(cudaMemcpyAsync(output_host, output_device, 
        sizeof(int) + max_objects * NUM_BOX_ELEMENT * sizeof(float), 
        cudaMemcpyDeviceToHost, stream
    ));
    checkRuntime(cudaStreamSynchronize(stream));

    int num_boxes = min((int)output_host[0], max_objects);
    for(int i = 0; i < num_boxes; ++i){
        float* ptr = output_host + 1 + NUM_BOX_ELEMENT * i;
        int keep_flag = ptr[6];
        if(keep_flag){
            box_result.emplace_back(
                ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], (int)ptr[5]
            );
        }
    }
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFree(predict_device));
    checkRuntime(cudaFree(output_device));
    checkRuntime(cudaFreeHost(output_host));
    return box_result;
}



void Yolov7::yolov7_inference_gpu(){

auto engine = TRT::load_infer("/home/rex/Desktop/tensorrt_learning/trt_cpp/workspace/yolov7.trtmodel");
    if(!engine){
        printf("load engine failed \n");
        return;
    }
    auto input       = engine->input();
    auto output      = engine->output();
    int input_width  = input->width();
    int input_height = input->height();
    auto image = imread("/home/rex/Desktop/tensorrt_learning/demo/yolov7/inference/images/horses.jpg");
    auto img_o = image.clone();
    int img_w = image.cols;
    int img_h = image.rows;
    auto warp_image = BaiscTools::warpaffine_cpu(image,input_height,input_width);
    auto input_image = warp_image.dst_image;
    auto m2x3_d2i = warp_image.m2x3_d2i;
    float *d2i = m2x3_d2i.ptr<float>();
    input_image.convertTo(input_image, CV_32F);

    Mat channel_based[3];
    for(int i = 0; i < 3; ++i)
        channel_based[i] = Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2-i));

    split(input_image, channel_based);
    for(int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f);
    
    engine->forward(true);
    float* predict = output->cpu<float>();
    int rows = 17640;
    int cols = 85;
    float confidence_threshold = 0.2f;
    float nms_threshold = 0.2f;
    auto bboxes = gpu_decode(predict, rows, cols,confidence_threshold,nms_threshold, m2x3_d2i);
    for (auto &box : bboxes)
    {
        cv::rectangle(img_o, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);
        cv::putText(img_o, cv::format("%.2f", box.confidence), cv::Point(box.left, box.top - 7), 0, 0.8, cv::Scalar(0, 0, 255), 2, 16);
    }
    cv::imwrite("yolov7-pred-gpu-decode.jpg", img_o);
}
