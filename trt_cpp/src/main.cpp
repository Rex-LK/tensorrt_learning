
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <demo-infer/demo-infer.hpp>
#include <common/basic_tools.hpp>

using namespace std;

int main()
{
    demoInfer demo;
    string demo_name = "yolov5seg";
    demo.do_infer(demo_name);
    return  0;
}
