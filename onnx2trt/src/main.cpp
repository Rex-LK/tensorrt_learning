
#include <stdio.h>
#include <iostream>
#include <fstream>

#include <demo-infer/demo-infer.hpp>
#include <common/basic_tools.hpp>
using namespace std;


int main(){
    // if(!BaiscTools::build_model("detr_sim",1)){
    //     return -1;
    // }
    // yolov5_inference();
    vit_inference();
    // detr_inference();

    return 0;
}


