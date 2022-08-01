#include <warpaffine.h>
using namespace cv;

int main(){ 
    Mat image = imread("/home/rex/Desktop/rex_extra/notebook/warpaffine/keji1.jpeg");
    Mat test_image = warpaffine_to_center_align(image, Size(640, 640));
    imwrite("test.jpg", test_image);
    imshow("1",test_image);
    waitKey(0);
    return 0;
}