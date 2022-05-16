#include<string>
#include <iostream>

class demoInfer
{
public:
    demoInfer(){}
    void unet_inference();
    void centernet_inference();
    void yolov5_inference();
    void vit_inference();
    void detr_inference();
    void do_infer(std::string demo_name){
        if(demo_name=="unet"){
            unet_inference();
        }
        else if(demo_name == "centernet"){
            centernet_inference();
        }
        else if(demo_name == "yolov5"){
            yolov5_inference();
        }
        else if(demo_name == "vit"){
            vit_inference();
        }
        else if(demo_name == "detr"){
            detr_inference();
        }
    }
};
