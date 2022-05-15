
#include <stdio.h>
#include <iostream>
#include <fstream>

#include <demo-infer/demo-infer.hpp>
#include <common/basic_tools.hpp>
#include <common/trt-tensor.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
using namespace std;



int main(){
    // if(!BaiscTools::build_model("centernet",1)){
    //     return -1;
    // }
    // yolov5_inference();
    // vit_inference();
    // detr_inference();
    centernet_inference();
    // load_from_py();

    return 0;
}


