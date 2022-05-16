#include <demo-infer/yolov5/yolov5.hpp>
#include<infer/trt-infer.hpp>
#include<common/basic_tools.hpp>
#include <common/trt-tensor.hpp>
#include <common/basic_tools.hpp>
#include<demo-infer/demo-infer.hpp>
using namespace cv;
using namespace std;
static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};


vector<string>classes = {
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

void demoInfer::yolov5_inference(){

    auto image = cv::imread("rq.jpg");
    auto yolov5 = YoloV5::create_infer("yolov5s.trtmodel");
    auto boxes = yolov5->commit(image).get();
    for(auto& box : boxes){
        cv::Scalar color(0, 255, 0);
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);

        auto name      = cocolabels[box.class_label];
        auto caption   = cv::format("%s %.2f", name, box.confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(box.left-3, box.top-33), cv::Point(box.left + text_width, box.top), color, -1);
        cv::putText(image, caption, cv::Point(box.left, box.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    cv::imwrite("yolov5-image-draw.jpg", image);
}

void demoInfer::vit_inference(){

    auto engine = TRT::load_infer("vit.trtmodel");
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        return;
    }

    engine->print();

    auto input       = engine->input();
    auto output      = engine->output();
    int input_width  = input->width();
    int input_height = input->height();

    auto image = imread("flower.jpg");
    float mean[] = {0.5, 0.5, 0.5};
    float std[]  = {0.5, 0.5, 0.5};

    resize(image, image, Size(input_width, input_height));
    image.convertTo(image, CV_32F);

    Mat channel_based[3];
    //channel_based 为{}

    for(int i = 0; i < 3; ++i)
        channel_based[i] = Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2-i));

    split(image, channel_based);
    for(int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f - mean[i]) / std[i];

    engine->forward(true);
    int num_classes   = output->size(1);
    float* prob       = output->cpu<float>();
    int predict_label = std::max_element(prob, prob + num_classes) - prob;
    auto labels       = BaiscTools::load_labels("labels.imagenet.txt");
    auto predict_name = labels[predict_label];
    float confidence  = prob[predict_label];
    printf("Predict: %s, confidence = %f, label = %d\n", predict_name.c_str(), confidence, predict_label);
}


struct Pred
{
    vector<vector<float>>bbox;
    vector<int>label;
};

void demoInfer::detr_inference(){

    auto engine = TRT::load_infer("detr_sim.trtmodel");
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        return;
    }

    engine->print();

    auto input       = engine->input();
    auto output      = engine->output();
    int input_width  = input->width();
    int input_height = input->height();


    auto image = imread("demo3.jpg");
    auto img_o = image.clone();
    // imshow("test",image);
    // waitKey(0);
    int img_w = image.cols;
    int img_h = image.rows;
    float mean[] = {0.485, 0.456, 0.406};
    float std[]  = {0.229, 0.224, 0.225};
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
    int prob_count = output->count()/output->batch();
    int cols = 95;
    int num_class = 91;
    //这张图有100个结果，每个结果一行 每行95个元素,前91个元素为概率值,后四个为这个结果的框预测框
    //每一行在onnx中已经 softmax, 这里只要取最大值就行,然后取最大值 看是否大于0.7 如果大于0.7，就将这个结果的预测框保存，
    //做完之后将,预测框缩放回原图大小

    float* start = prob;
    Pred pred;
    for(int i = 0;i<=prob_count;i+=95){
        
        //每一行最大的得分的索引
        int label = max_element(start, start + num_class) - start;
        float score = start[label];
        if(score <= 0.7){
            start = prob+i;
            continue;
        }
        else{
            float p_x = start[num_class];
            float p_y = start[num_class+1];

            float p_w = start[num_class+2];
            float p_h = start[num_class+3]; 
            
            int p_x1 =  (int)((p_x - 0.5 * p_w) * img_w);
            int p_x2 =  (int)((p_x + 0.5 * p_w) * img_w); 
            int p_y1 =  (int)((p_y - 0.5 * p_h) * img_h);
            int p_y2 =  (int)((p_y + 0.5 * p_h) * img_h);
            
            Rect rect(p_x1,p_y1,p_x2 - p_x1,p_y2 - p_y1);
            rectangle(img_o,rect,Scalar(255,0,0),2);
            putText(img_o, classes[label],  Point(p_x1,p_y1+20), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255), 2, 2, 0);
        }
        start = prob+i;
    }
    imwrite("demo3-res.jpg",img_o);
}



struct bbox{
    float left, top, right, bottom, confidence;
    int label;

    bbox() = default;
    bbox(float left, float top, float right, float bottom, float confidence, int label):
    left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label){}
};

void demoInfer::centernet_inference(){
    auto engine = TRT::load_infer("centernet.trtmodel");
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
    auto image = imread("street.jpg");
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
        cv::rectangle(img_o, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);
        cv::putText(img_o, cv::format("%.2f", box.confidence), cv::Point(box.left, box.top - 7), 0, 0.8, cv::Scalar(0, 0, 255), 2, 16);
    }
    cv::imwrite("centernet-pred-street.jpg", img_o);
}


void load_from_py(){
    //用于保存模型推理之后confidence大于阈值的框
    int input_h = 1330;
    int input_w = 1330;
    vector<bbox> boxes;
    TRT::Tensor tensor;
    tensor.load_from_file("pred.tensor");
    float* prob = tensor.cpu<float>();
    int num = 20;  
    int count = tensor.count();
    float *start = prob;
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
        float left   = (x - w * 0.5) /128 * input_h;
        float top    = (y - h * 0.5) /128 * input_w;
        float right  = (x + w * 0.5) /128 * input_h;
        float bottom = (y + h * 0.5) /128 * input_w;
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
    cout<<box_result.size()<<endl;
    Mat image = imread("street.jpg");
    for(auto& box : box_result){
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);
        cv::putText(image, cv::format("%.2f", box.confidence), cv::Point(box.left, box.top - 7), 0, 0.8, cv::Scalar(0, 0, 255), 2, 16);
    }
    cv::imwrite("centernet-pred-street.jpg", image);
}



static vector<int> _classes_colors = {
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 
    128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 
    64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 12
};

static tuple<cv::Mat, cv::Mat> post_process(float* output, int output_width, int output_height, int num_class, int ibatch){

    cv::Mat output_prob(output_height, output_width, CV_32F);
    cv::Mat output_index(output_height, output_width, CV_8U);

    float* pnet   = output + ibatch * output_width * output_height * num_class;
    float* prob   = output_prob.ptr<float>(0);
    uint8_t* pidx = output_index.ptr<uint8_t>(0);

    for(int k = 0; k < output_prob.cols * output_prob.rows; ++k, pnet+=num_class, ++prob, ++pidx){
        int ic = std::max_element(pnet, pnet + num_class) - pnet;
        *prob  = pnet[ic];
        *pidx  = ic;
    }
    return make_tuple(output_prob, output_index);
}



static void render(cv::Mat& image, const cv::Mat& prob, const cv::Mat& iclass){

    auto pimage = image.ptr<cv::Vec3b>(0);
    auto pprob  = prob.ptr<float>(0);
    auto pclass = iclass.ptr<uint8_t>(0);

    for(int i = 0; i < image.cols*image.rows; ++i, ++pimage, ++pprob, ++pclass){

        int iclass        = *pclass;
        float probability = *pprob;
        auto& pixel       = *pimage;
        float foreground  = min(0.6f + probability * 0.2f, 0.8f);
        float background  = 1 - foreground;
        for(int c = 0; c < 3; ++c){
            auto value = pixel[c] * background + foreground * _classes_colors[iclass * 3 + 2-c];
            pixel[c] = min((int)value, 255);
        }
    }
}

void demoInfer::unet_inference(){
    auto engine = TRT::load_infer("unet.trtmodel");
    if(!engine){
        printf("load engine failed \n");
        return;
    }
    auto input       = engine->input();
    auto output      = engine->output();
    int input_width  = input->width();
    int input_height = input->height();
    int num_classes = 21;

    auto image = cv::imread("street.jpg");
    auto imge_warpaffined = BaiscTools::warpaffine_cpu(image,input_width,input_height);
    auto input_image = imge_warpaffined.dst_image;
    input_image.convertTo(input_image, CV_32F);
    Mat channel_based[3];
    for(int i = 0; i < 3; ++i)
        channel_based[i] = Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2-i));

    split(input_image, channel_based);
    for(int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f);

    engine->forward(true);
    float* prob = output->cpu<float>();
    cv::Mat unet_prob, iclass;
    tie(unet_prob, iclass) = post_process(prob, 512, 512, num_classes, 0);
    cv::warpAffine(unet_prob, unet_prob, imge_warpaffined.m2x3_d2i, image.size(), cv::INTER_LINEAR);
    cv::warpAffine(iclass, iclass, imge_warpaffined.m2x3_d2i, image.size(), cv::INTER_NEAREST);
    render(image, unet_prob, iclass);
    printf("Done, Save to image-draw.jpg\n");
    cv::imwrite("unet-draw.jpg", image);
}

