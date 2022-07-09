#include <string>
#include <iostream>
#include <vector>
#include <demo-infer/centernet/centernet.h>
#include <demo-infer/detr/detr.h>
#include <demo-infer/hrnet/hrnet.h>
#include <demo-infer/unet/unet.h>
#include <demo-infer/vit/vit.h>
#include <demo-infer/yolov7/yolov7.h>
class demoInfer
{
public:
    demoInfer() {}
    void yolov5_inference();
    void vit_inference();
    // void load_from_py();
    void do_infer(std::string demo_name)
    {
        if (demo_name == "unet")
        {
            Unet::unet_inference();
        }

        else if (demo_name == "load_from_py")
        {
            // load_from_py();
        }
        else if (demo_name == "centernet")
        {
            Centernet::centernet_inference();
        }
        else if (demo_name == "centernet_gpu_decode")
        {
            Centernet::centernet_inference_gpu();
        }
        else if (demo_name == "yolov5")
        {
            yolov5_inference();
        }
        else if (demo_name == "vit")
        {
            Vit::vit_inference();
        }
        else if (demo_name == "detr")
        {
            Detr::detr_inference();
        }
        else if (demo_name == "hrnet")
        {
            Hrnet::hrnet_inference();
        }
        else if (demo_name == "hrnet_gpu_decode")
        {
            Hrnet::hrnet_inference_gpu();
        }
        else if (demo_name == "yolov7")
        {
            Yolov7::yolov7_inference();
        }
        else if (demo_name == "yolov7_gpu_decode")
        {
            Yolov7::yolov7_inference_gpu();
        }
    }
};
