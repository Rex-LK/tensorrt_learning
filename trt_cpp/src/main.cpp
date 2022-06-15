
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <demo-infer/demo-infer.hpp>
#include <common/basic_tools.hpp>

using namespace std;
                                                                                                                                                                                                                                                                        
int main(){
    // cmake和make是不同的路径
    if(!BaiscTools::build_model("/home/rex/Desktop/cv_demo/detr-main-0614/detr_sim",1)){
        return -1;
    }
    demoInfer demo;
    string demo_name = "detr"; 
    demo.do_infer(demo_name);
    return 0;
}


