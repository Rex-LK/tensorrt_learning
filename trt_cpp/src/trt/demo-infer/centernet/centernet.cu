#include <infer/trt-infer.hpp>
#include <common/basic_tools.hpp>
#include <common/cuda-tools.hpp>
#include <common/trt-tensor.hpp>
#include <opencv2/opencv.hpp>
#include <demo-infer/centernet/centernet.h>
using namespace std;
using namespace cv;

static __global__ void decode_kernel(
    float* predict,int im_h,int im_w,int num_bboxes, int num_classes, float confidence_threshold, 
    float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT
){  
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;
    // //每隔24个位置为 跳到下一行
    float* pitem     = predict + (4 + num_classes) * position;
    //pitem 前20个位置表示概率，后四个位置表示whxy
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

    int index = atomicAdd(parray, 1);
    if(index >= max_objects)
        return;
    float* pwhxy = pitem;
    


    float width      = *(pwhxy+20);
    float height     = *(pwhxy+21);
    //偏移量，还需要加上 当前热力点的坐标
    float cx         = *(pwhxy+22) + position%128;
    float cy         = *(pwhxy+23) + position/128;

    float left   = (cx - width * 0.5f) * im_h / 128;
    float top    = (cy - height * 0.5f) * im_w / 128;
    float right  = (cx + width * 0.5f) * im_h / 128;
    float bottom = (cy + height * 0.5f) * im_w / 128;

    // 第一个位置用来计数
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
    //最多计算1000个框
    int count = min((int)*bboxes, max_objects);
    if (position >= count) 
        return;
    //每一个bbox有七个值 , 这里修改bbox的最后一个值
    // left, top, right, bottom, confidence, class, keepflag
    // 每个bbox的起点
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for(int i = 0; i < count; ++i){
        
        //遍历count个框
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        //如果是自己和自己比或者 不同类别 就不需要比较了
        if(i == position || pcurrent[5] != pitem[5]) continue;

        //如果遍历的框的score 大于等于 当前这个线程的socre
        if(pitem[4] >= pcurrent[4]){
            //遍历的bbox的score与当前线程的socre 且遍历的框的索引小于这个框的索引,
            //得分一致的话，只计算当前框与其之后框的iou
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
};


void decode_kernel_invoker(
    float* predict,int im_h,int im_w ,int num_bboxes, int num_classes, float confidence_threshold, 
    float nms_threshold, float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream){
    //parry:一维 bbox *1000
    auto block = num_bboxes > 512 ? 512 : num_bboxes;
    auto grid = (num_bboxes + block - 1) / block;
    decode_kernel<<<grid, block, 0, stream>>>(
        predict,im_h,im_w,num_bboxes, num_classes, confidence_threshold, 
        invert_affine_matrix, parray, max_objects, NUM_BOX_ELEMENT
    
    );
    block = max_objects > 512 ? 512 : max_objects;
    grid = (max_objects + block - 1) / block;
    fast_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold, NUM_BOX_ELEMENT);
}



vector<bbox> gpu_decode(float* predict, int rows, int cols,int im_h,int im_w, float confidence_threshold = 0.3f, float nms_threshold = 0.1f){
    
    vector<bbox> box_result;
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    float* predict_device = nullptr;
    //用于存放bbox
    float* output_device = nullptr;
    float* output_host = nullptr;
    int max_objects = 1000;

    int NUM_BOX_ELEMENT = 7;  // left, top, right, bottom, confidence, class, keepflag
    checkRuntime(cudaMalloc(&predict_device, rows * cols * sizeof(float)));
    checkRuntime(cudaMalloc(&output_device, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    checkRuntime(cudaMallocHost(&output_host, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    checkRuntime(cudaMemcpyAsync(predict_device, predict, rows * cols * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    decode_kernel_invoker(
        predict_device, im_h,im_w,rows, cols - 4, confidence_threshold, 
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

void Centernet::centernet_inference_gpu(){
    auto engine = TRT::load_infer("/home/rex/Desktop/tensorrt_learning/trt_cpp/workspace/centernet.trtmodel");
    if(!engine){
        printf("load engine failed \n");
        return;
    }
    auto input       = engine->input();
    auto output      = engine->output();
    int input_width  = input->width();
    int input_height = input->height();
    float mean[] = {0.40789655, 0.44719303, 0.47026116};
    float std[]  = {0.2886383, 0.27408165, 0.27809834};
    auto image = imread("/home/rex/Desktop/tensorrt_learning/trt_cpp/workspace/street.jpg");
    auto img_o = image.clone();
    int img_w = image.cols;
    int img_h = image.rows;
    resize(image, image, Size(input_width, input_height));
    image.convertTo(image, CV_32F);

    Mat channel_based[3];
    for(int i = 0; i < 3; ++i)
        channel_based[i] = Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2-i));

    split(image, channel_based);
    for(int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f - mean[i]) / std[i];

    engine->forward(true);
    float* predict = output->cpu<float>();
    int rows = 128*128;
    int cols = 24;
    auto bboxes = gpu_decode(predict,rows,cols,img_h,img_w);

    for(auto box : bboxes){
        int x1 = box.left ;
        int y1 = box.top;
        int x2 = box.right;
        int y2 = box.bottom;
        rectangle(img_o, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 255),4);
    }
    imwrite("/home/rex/Desktop/tensorrt_learning/trt_cpp/workspace/centernet-gpu-pred.jpg", img_o);

}



void Centernet::centernet_inference(){
    auto engine = TRT::load_infer("/home/rex/Desktop/tensorrt_learning/trt_cpp/workspace/centernet.trtmodel");
    if(!engine){
        printf("load engine failed \n");
        return;
    }
    auto input       = engine->input();
    auto output      = engine->output();
    int input_width  = input->width();
    int input_height = input->height();
    float mean[] = {0.40789655, 0.44719303, 0.47026116};
    float std[]  = {0.2886383, 0.27408165, 0.27809834};
    auto image = imread("/home/rex/Desktop/tensorrt_learning/trt_cpp/workspace/street.jpg");
    auto img_o = image.clone();
    int img_w = image.cols;
    int img_h = image.rows;
    resize(image, image, Size(input_width, input_height));
    image.convertTo(image, CV_32F);

    Mat channel_based[3];
    for(int i = 0; i < 3; ++i)
        channel_based[i] = Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2-i));

    split(image, channel_based);
    for(int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f - mean[i]) / std[i];

    engine->forward(true);
    float* prob = output->cpu<float>();
    int num = 20;  
    float *start = prob;
    int count = output->count();
     vector<bbox> boxes;
    for(int i=0;i<count;i+=24){
        //现在有128*128个点  就有128*128行，每行24个，前20个为类别，后四个为 w,h,x,y
        start = prob+i;
        int label = max_element(start,start+num) - start;
        float confidence = start[label];
        if(confidence<0.3)
            continue;

        float w = start[ num];
        float h = start[num+1];
        //这里的x,y 还要加上偏移量
        float x = start[num+2] + (i/24)%128;
        float y = start[num+3] + (i/24)/128;
        float left   = (x - w * 0.5) /128 * img_h;
        float top    = (y - h * 0.5) /128 * img_w;
        float right  = (x + w * 0.5) /128 * img_h;
        float bottom = (y + h * 0.5) /128 * img_w;
        boxes.emplace_back(left, top, right, bottom, confidence, (float)label);
    }
    //nms
    sort(boxes.begin(), boxes.end(), [](bbox& a, bbox& b){return a.confidence > b.confidence;});
    //然后使用vector<bool>来标记是是否要删除box
    vector<bool> remove_flags(boxes.size());
    vector<bbox> box_result;
    box_result.reserve(boxes.size());

    //计算两个box之间的IOU
    auto iou = [](const bbox& a, const bbox& b){
        float cross_left   = max(a.left, b.left);
        float cross_top    = max(a.top, b.top);
        float cross_right  = min(a.right, b.right);
        float cross_bottom = min(a.bottom, b.bottom);
        //计算重叠部分的面积,注意面积非0
        float cross_area = max(0.0f, cross_right - cross_left) * max(0.0f, cross_bottom - cross_top);
        //A面积+B面积 - 一个重叠面积
        float union_area = max(0.0f, a.right - a.left) * max(0.0f, a.bottom - a.top) 
                         + max(0.0f, b.right - b.left) * max(0.0f, b.bottom - b.top) - cross_area;
        if(cross_area == 0 || union_area == 0) return 0.0f;
        return cross_area/union_area;
    };

    for(int i = 0;i<boxes.size();i++){
        if(remove_flags[i]) continue;
        auto &ibox  = boxes[i];
        //第一个box必定不会remove
        box_result.emplace_back(ibox);
        //第一个box与 之后的box两两比较
        for(int j = i + 1;j<boxes.size();j++){
            if(remove_flags[j]) continue;
            auto& jbox = boxes[j];
            //如果两个box的lable相同
            if(ibox.label == jbox.label){
                //则比较IOU
                if(iou(ibox, jbox) >= 0.3)
                    remove_flags[j] = true;
            }
        }
    }
    for(auto& box : box_result){
        rectangle(img_o, Point(box.left, box.top), Point(box.right, box.bottom), Scalar(0, 255, 0), 2);
        putText(img_o, format("%.2f", box.confidence), Point(box.left, box.top - 7), 0, 0.8, Scalar(0, 0, 255), 2, 16);
    }
    imwrite("centernet-pred.jpg", img_o);
}