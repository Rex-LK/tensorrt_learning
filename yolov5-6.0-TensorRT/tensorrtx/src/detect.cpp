#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "preprocess.h"
#include "yolov5-detect.h"

using namespace std;
using namespace cv;
void on_mouse(int EVENT, int x, int y, int flags, void* userdata);
int main()
{
    namedWindow("【display】");
    Mat src;
    src = imread("/home/rex/Pictures/98232810b8140c7c80f4d88786b796ab.jpeg");
    //cvtColor(src, src, COLOR_RGB2GRAY);
    setMouseCallback("【display】", on_mouse,&src);
    //以40ms刷新显示
        imshow("【display】", src);
        waitKey(0);
    return 0;
}

void on_mouse(int EVENT, int x, int y, int flags, void* userdata)
{
    Mat hh;
    hh = *(Mat*)userdata;
    Point p(x, y);
    switch (EVENT)
    {
        case EVENT_LBUTTONDOWN:
        {

            // printf("b=%d\t", hh.at<Vec3b>(p)[0]);
            // printf("g=%d\t", hh.at<Vec3b>(p)[1]);
            // printf("r=%d\n", hh.at<Vec3b>(p)[2]);
            cout<<"x: "<<x<<"  "<<"y: "<<y<<endl;
        }
        break;

    }

}