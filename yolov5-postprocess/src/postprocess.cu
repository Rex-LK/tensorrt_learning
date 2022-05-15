#include "postprocess.h"


__device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

__global__ void decode_kernel(
    float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT
){  

    // 每个线程处理一个bbox
    int position = blockDim.x * blockIdx.x + threadIdx.x;

    // 超出最大索引return
    if (position >= num_bboxes) return;

    // 当前bbox起始位置的指针
    float* pitem     = predict + (5 + num_classes) * position;
    float objectness = pitem[4];
    if(objectness < confidence_threshold)
        return;


    // 80个类别的起始位置的指针
    float* class_confidence = pitem + 5;
    float confidence        = *class_confidence++;
    int label               = 0;

    // 遍历获得  最大confidence 以及对应的label
    for(int i = 1; i < num_classes; ++i, ++class_confidence){
        if(*class_confidence > confidence){
            confidence = *class_confidence;
            label      = i;
        }
    }

    confidence *= objectness;
    if(confidence < confidence_threshold)
        return;


    //parray = [count, box1, box2, box3]
    //atomicAdd 的含义是 count++ 并且返回旧的值
    int index = atomicAdd(parray, 1);
    if(index >= max_objects)
        return;

    // 计算坐标
    float cx         = *pitem++;
    float cy         = *pitem++;
    float width      = *pitem++;
    float height     = *pitem++;
    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    // affine_project(invert_affine_matrix, left,  top,    &left,  &top);
    // affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    // left, top, right, bottom, confidence, class, keepflag
    // keepflag 用于储存是否需要被删除
    
    // index 表示当前 结果偏离parray box的个数
    // 加1 是因为第一个位置是count
    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
}

__device__ float box_iou(
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

__global__ void fast_nms_kernel(float* bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT){

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    // 最大bbox的数量
    int count = min((int)*bboxes, max_objects);
    if (position >= count) 
        return;                                                                                                                                    
    
    // left, top, right, bottom, confidence, class, keepflag
    // 输出结果中当前 boxx的起始位置的指针
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for(int i = 0; i < count; ++i){
        // 每个bbox的起始位置的指针
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if(i == position || pcurrent[5] != pitem[5]) continue;

        //如果之后的bbox的 confidence> 当前bbox的 confidence
        if(pitem[4] >= pcurrent[4]){
            // 两个bboxs属于同一类 keep_flag 默认为1
            if(pitem[4] == pcurrent[4] && i < position)
                continue;

            //同类bbox 利用iou除去 重叠度高的框
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
        invert_affine_matrix, parray, max_objects, NUM_BOX_ELEMENT
    );

    block = max_objects > 512 ? 512 : max_objects;
    grid = (max_objects + block - 1) / block;
    fast_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold, NUM_BOX_ELEMENT);
}



std::vector<Box> cpu_decode(float* predict, int rows, int cols, float confidence_threshold , float nms_threshold ){

    std::vector<Box> boxes;
    // 类别数量
    int num_classes = cols - 5;

    for(int i = 0; i < rows; ++i){
        //每一行的起始位置的指针  
        //每一行 x,y,w,h,socre + 80
        
        float* pitem = predict + i * cols;
        
        // score  前景和背景的概率
        float objness = pitem[4];
        //低于阈值的直接过滤,节约计算时间
        if(objness < confidence_threshold)
            continue;

        //pitem 为一行的起始指针 +5 是从第六个位置开始 指向类别概率的 中的第一个位置
        float* pclass = pitem + 5;
        
        // 从指针pclass 到 pclass + num_classes 最大位置的指针  获得偏移量
        int label     = std::max_element(pclass, pclass + num_classes) - pclass;
        
        // pclass[label] 往后找label个位置 对应的值   
        // 最大类别概率
        float prob    = pclass[label];

        // 置信度
        float confidence = prob * objness;
        if(confidence < confidence_threshold)
            continue;
        //计算坐标 
        float cx     = pitem[0];
        float cy     = pitem[1];
        float width  = pitem[2];
        float height = pitem[3];
        float left   = cx - width * 0.5;
        float top    = cy - height * 0.5;
        float right  = cx + width * 0.5;
        float bottom = cy + height * 0.5;
        boxes.emplace_back(left, top, right, bottom, confidence, (float)label);
    }

    //nms
    // lambda 函数 采用引用 避免拷贝   
    std::sort(boxes.begin(), boxes.end(), [](Box& a, Box& b){return a.confidence > b.confidence;});
    
    // 用true false 储存需要删除的 box
    std::vector<bool> remove_flags(boxes.size());
    std::vector<Box> box_result;
    // 先个 box_result 分配 boxes.size() 固定内存，因为 emplace_back 会使得地址发生变化
    // 而使用reserve 后内存只会变化一次,且会节约内存
    box_result.reserve(boxes.size());

    // 计算两个 box之间的 iou
    auto iou = [](const Box& a, const Box& b){
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
        // nms之后的结果
        box_result.emplace_back(ibox);
        for(int j = i + 1; j < boxes.size(); ++j){
            if(remove_flags[j]) continue;

            auto& jbox = boxes[j];

            // 如果两个box属于同一类才进行比较 可以节约时间
            if(ibox.label == jbox.label){
                // class matched
                if(iou(ibox, jbox) >= nms_threshold)
                    remove_flags[j] = true;
            }
        }
    }
    return box_result;
}

std::vector<Box> gpu_decode(float* predict, int rows, int cols, float confidence_threshold , float nms_threshold ){
    
    std::vector<Box> box_result;
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    float* predict_device = nullptr;
    float* output_device = nullptr;
    float* output_host = nullptr;
    // 由于在cpu上 是顺序执行的，可以vector emplace_back 
    // 而在GPU上是并行的  可以用[count box1 box2] 储存
    // 假定一张图上只有1000个框，超过1000个框的过滤
    int max_objects = 1000;
    int NUM_BOX_ELEMENT = 7;  // left, top, right, bottom, confidence, class, keepflag
    checkRuntime(cudaMalloc(&predict_device, rows * cols * sizeof(float)));
    checkRuntime(cudaMalloc(&output_device, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    checkRuntime(cudaMallocHost(&output_host, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));

    checkRuntime(cudaMemcpyAsync(predict_device, predict, rows * cols * sizeof(float), cudaMemcpyHostToDevice, stream));
    decode_kernel_invoker(
        predict_device, rows, cols - 5, confidence_threshold, 
        nms_threshold, nullptr, output_device, max_objects, NUM_BOX_ELEMENT, stream
    );
    checkRuntime(cudaMemcpyAsync(output_host, output_device, 
        sizeof(int) + max_objects * NUM_BOX_ELEMENT * sizeof(float), 
        cudaMemcpyDeviceToHost, stream
    ));
    checkRuntime(cudaStreamSynchronize(stream));

    // 所有框的个数
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

std::vector<uint8_t> load_file(const std::string& file){

    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}