#include<infer/trt-infer.hpp>
#include<common/basic_tools.hpp>
#include<common/cuda-tools.hpp>
#include <common/trt-tensor.hpp>
#include <opencv2/opencv.hpp>
#include<demo-infer/hrnet/hrnet.h>

using namespace std;
using namespace cv;


struct hrnet_bbox{
    float x, y, label, score;

    hrnet_bbox() = default;
    hrnet_bbox(float x, float y, float label, float score):
    x(x), y(y), label(label), score(score){}
};


void Hrnet::hrnet_inference(){
    //1*17 *64*48
    //将图片分成64*48个点
    //分别预测64*48个点中 17类的概率
    //论文中将最后得到的点 偏移了一些

    auto engine = TRT::load_infer("hrnet.trtmodel");
    if(!engine){
        printf("load engine failed \n");
        return;
    }
    auto input       = engine->input();
    auto output      = engine->output();
    int input_width  = input->width();
    int input_height = input->height();
    float mean[] = {0.485, 0.456, 0.406};
    float std[]  = {0.229, 0.224, 0.225};
    auto image = imread("person2.jpeg");
    auto img_o = image.clone();
    int img_w = image.cols;
    int img_h = image.rows;
    auto warp_image = BaiscTools::warpaffine_cpu(image,input_height,input_width);
    auto input_image = warp_image.dst_image;
    auto m2x3_d2i = warp_image.m2x3_d2i;
    float* d2i = m2x3_d2i.ptr<float>();
    // cv::imwrite("warp-affine.jpg", input_image);
    input_image.convertTo(input_image, CV_32F);

    //brg2rgb
    Mat channel_based[3];
    for(int i = 0; i < 3; ++i)
        channel_based[i] = Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2-i));
    // (image -maen) / std 
    split(input_image, channel_based);
    for(int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f - mean[i]) / std[i];

    engine->forward(true);
    // output->save_to_file("hrnet.tensor");
    float* prob = output->cpu<float>();
    float* start = prob;
    int nums = 17;
    int pic_region = 3072;
    for(int i=0;i<output->count();i+=pic_region){
        start = prob + i;
        int label = (max_element(start,start+pic_region) - start);
        float score = start[label];
        if(score<0.2)
            continue;
        float x = label % 48 ;
        float y = label / 48 ;
        //这里计算的warpaffine的变换矩阵中的缩放系数 与 pyhton中image_transformer差四倍
        int x_o = (x * d2i[0] * 4) + d2i[2];
        int y_o = (y * d2i[4] * 4) + d2i[5]; 
        cv::circle(img_o, cv::Point((int)x_o,(int)y_o), 1, (0, 0, 255), 2);
        cv::imwrite("hrnet-pred.jpg", img_o);
    }
}





static __global__ void hrnet_decode_kernel(
    float* predict,int num_bboxes, int num_classes, float confidence_threshold, 
    float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT
){  
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;
    // //每隔64*48个位置为 跳到下一行
    float* pitem     = predict + (64*48) * position;
    //pitem 这个类别 在图中所有点 的概率
    int label = 0;
    //第一个位置的socre
    float* class_confidence = pitem;
    float confidence  = *class_confidence;
    
    for(int i = 1; i < num_classes; ++i, ++class_confidence){
        if(*class_confidence > confidence){
            confidence = *class_confidence;
            label      = i;
        }
    }
    
    if(confidence < confidence_threshold)
        return;
    // printf("%f\n",confidence);
    // printf("%d\n",label-1);
    // printf("%d\n",position);

    int index = atomicAdd(parray, 1);
    if(index >= max_objects)
        return;
    float* pwhxy = pitem;

    float cx  = ((label-1) % 48);
    float cy  = ((label-1) / 48);

    // 第一个位置用来计数
    // left, top, right, bottom, confidence, class, keepflag
    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = cx;
    *pout_item++ = cy;
    *pout_item++ = position;
    *pout_item++ = confidence;
}




void hrnet_decode_kernel_invoker(
    float* predict,int num_bboxes, int num_classes, float confidence_threshold, 
    float nms_threshold, float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream){
    //parry:一维 bbox *1000
    auto block = num_bboxes > 512 ? 512 : num_bboxes;
    auto grid = (num_bboxes + block - 1) / block;
    hrnet_decode_kernel<<<grid, block, 0, stream>>>(
        predict,num_bboxes, num_classes, confidence_threshold, 
        invert_affine_matrix, parray, max_objects, NUM_BOX_ELEMENT
    
    );
}

vector<hrnet_bbox> hrnet_gpu_decode(float* predict, int rows, int cols,float confidence_threshold , float nms_threshold){
    
    vector<hrnet_bbox> box_result;
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    float* predict_device = nullptr;
    //用于存放bbox
    float* output_device = nullptr;
    float* output_host = nullptr;
    int max_objects = 1000;

    int NUM_BOX_ELEMENT = 4;  // x_o,y_o,label,score
    checkRuntime(cudaMalloc(&predict_device, rows * cols * sizeof(float)));
    checkRuntime(cudaMalloc(&output_device, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    checkRuntime(cudaMallocHost(&output_host, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    checkRuntime(cudaMemcpyAsync(predict_device, predict, rows * cols * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    hrnet_decode_kernel_invoker(
        predict_device,rows, cols, confidence_threshold, 
        nms_threshold, nullptr, output_device, max_objects, NUM_BOX_ELEMENT, stream
    );
    checkRuntime(cudaMemcpyAsync(output_host, output_device, 
        sizeof(int) + max_objects * NUM_BOX_ELEMENT * sizeof(float), 
        cudaMemcpyDeviceToHost, stream
    ));
    checkRuntime(cudaStreamSynchronize(stream));
    
    int num_boxes = min((int)output_host[0], max_objects);
    for(int i = 0; i < num_boxes; ++i){
        float* ptr = output_host + 1 + NUM_BOX_ELEMENT * i;
        box_result.emplace_back(
                ptr[0], ptr[1], ptr[2], ptr[3]
            );
    }
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFree(predict_device));
    checkRuntime(cudaFree(output_device));
    checkRuntime(cudaFreeHost(output_host));
    return box_result;
}




void Hrnet::hrnet_inference_gpu(){


 auto engine = TRT::load_infer("hrnet.trtmodel");
    if(!engine){
        printf("load engine failed \n");
        return;
    }
    auto input       = engine->input();
    auto output      = engine->output();
    int input_width  = input->width();
    int input_height = input->height();
    float mean[] = {0.485, 0.456, 0.406};
    float std[]  = {0.229, 0.224, 0.225};
    auto image = imread("person2.jpeg");
    auto img_o = image.clone();
    int img_w = image.cols;
    int img_h = image.rows;
    auto warp_image = BaiscTools::warpaffine_cpu(image,input_height,input_width);
    auto input_image = warp_image.dst_image;
    auto m2x3_d2i = warp_image.m2x3_d2i;
    float* d2i = m2x3_d2i.ptr<float>();
    // cv::imwrite("warp-affine.jpg", input_image);
    input_image.convertTo(input_image, CV_32F);

    Mat channel_based[3];
    for(int i = 0; i < 3; ++i)
        channel_based[i] = Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2-i));

    split(input_image, channel_based);
    for(int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f - mean[i]) / std[i];

    engine->forward(true);
    float* prob = output->cpu<float>();
    int rows = 17;
    int cols = 64*48;
    float confidence_threshold = 0.2f;
    float nms_threshold = 0.1f;
    auto bboxes = hrnet_gpu_decode(prob,rows,cols,confidence_threshold,nms_threshold);
    for(int i=0;i<bboxes.size();i++){
        int x_o = bboxes[i].x * d2i[0] * 4 + d2i[2];
        int y_o = bboxes[i].y * d2i[4] * 4 + d2i[5];
        cv::circle(img_o, cv::Point((int)x_o,(int)y_o), 1, (255, 0, 0), 2);
        cv::imwrite("hrnet-cuda-pred.jpg", img_o);
    }
}