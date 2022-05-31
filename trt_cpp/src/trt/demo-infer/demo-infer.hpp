#include<string>
#include <iostream>
#include<vector>

struct bbox{
    float left, top, right, bottom, confidence;
    int label;

    bbox() = default;
    bbox(float left, float top, float right, float bottom, float confidence, int label):
    left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label){}
};



class demoInfer
{
public:
    demoInfer(){}
    void unet_inference();
    
    void centernet_inference();
    void centernet_inference_gpu();

    void yolov5_inference();
    void vit_inference();
    void detr_inference();
    void hrnet_inference();
    void load_from_py();
    void do_infer(std::string demo_name){
        if(demo_name=="unet"){
            unet_inference();
        }
        else if(demo_name == "load_from_py"){
            // load_from_py();
        }
        else if(demo_name == "centernet"){
            // centernet_inference();
            centernet_inference_gpu();
        }
        else if(demo_name == "yolov5"){
            yolov5_inference();
        }
        else if(demo_name == "vit"){
            vit_inference();
        }
        else if(demo_name == "detr_sim"){
            detr_inference();
        }
        else if(demo_name == "hrnet"){
            hrnet_inference();
        }
    }
};


