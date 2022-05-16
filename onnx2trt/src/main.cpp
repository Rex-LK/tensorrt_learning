
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <demo-infer/demo-infer.hpp>
#include <common/basic_tools.hpp>

using namespace std;



int main(){
    if(!BaiscTools::build_model("unet",1)){
        return -1;
    }
    demoInfer demo;
    string demo_name = "unet";
    demo.do_infer(demo_name);
    return 0;
}


