# CV学习笔记: yolov5-6.0之tensorrt加速

## 1、前言

原yolov5 github地址为:

原tensorrtx github地址为:

本项目github 地址为:

​      由于最近工作转到ros上进行开发，同时需要将识别框架进行加速优化，于是需要对yolov5进行加速。本文将简单介绍构造引擎的方法,并对 tensortx 的代码进行封装，以便有需要的小伙伴直接调用，尤其是对于c++不熟且需要用到的小伙伴，再次感谢上面两位大佬提供的开源项目。

## 2、构造引擎

​	yolov5到目前为止有6个版本，tensorrtx项目中不同的tag对应yolov5的版本，请务必下载 tensorrtx和yolov5对应的版本，才能运行成功。

​	首先请下载本项目，然后进入到yolov5-6.0文件夹中，将训练好权重放入weights文件夹中，本文以官方自带的yolov5s 为例。

​	在终端中输入 python gen_wts.py --w weight/yolov5s.pt --o ./

​	其中 --w 为训练好的模型路径 --o 为输出的.wts文件，等待一段时间，在yolov5-6.0文件中将会产生一个 yolov5s.wts的中间文件，这个文件保存了模型的所有参数。然后将yolov5s.wts文件移动到tensorrtx/engine文件夹中。

​	接下来，就要将产生的wts文件转化为engine文件了，首先根据模型的类别数量，修改yololayer.h中的num_classes 的数量，本文所使用的类别数量为80.

然后修改CMakeLists.txt

```c++
//cuda 路径
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
//tensorrt 路径
include_directories(/usr/local/TensorRT/include/)
link_directories(/usr/local/TensorRT/targets/x86_64-linux-gnu/lib/)
```

​	上述修改完毕之后，在tensorrtx文件夹下新建build文件夹 ,输入 cmake ..  以及cmake -j ,在build文件夹下将会出现yolov5 以及detect两个执行文件，如果没报错，那就十分nice了。

​	在这里先说yolov5，这个可执行文件包含了两个功能，构造引擎以及对文件夹中的图片进行简单预测。那么就开始进行转化吧，在终端输入

​	./yolov5 -s ../engine/yolov5s.wts ../engine/yolov5s.engine s

​	其中 -s 为构造引擎的选项,最后的s为模型的大小，可根据模型的大小进行替换。等待一会，在engine文件夹中就出现了 yolov5s.engine 文件，如果出现错误，可能的原因为 tensorrtx和yolov5 的版本没对应，或者tensorrt的版本和cuda、cudnn版本没有对应。

​	engine生成后，可以使用yolov5这个可执行文件简单的看一下加速后的效果，在终端中执行

​	./yolov5s -d ./engine/yolov5s.engine ../sample

​	其中 -d 选项为预测选项，../samlpe为图片路基，这里简单看一下效果。

​	构造完引擎之后，发现，如果小伙伴需要对视频流进行实时检测，该怎么对原作者的代码进行修改呢，接下来，本文将对源码进行封装，并且添加部分函数，极大简化了engine的调用。

### 3、yolov5_detect.h

​	为了更加简单的调用engine文件，本项目对源码进行了封装，在yolov5_detect.h文件中声明了yolov5 的检测类，在构造函数中加载引擎，在成员函数中进行预测。接下来直接看代码。

    class yolov5
    {
    public:
        yolov5(std::string engine_name, int classnum)
        {
            this->ClassNum = classnum;
            this->engine_name = engine_name;
            std::ifstream file(engine_name, std::ios::binary);
            size_t size = 0;
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            this->trtModelStream = new char[size];
            assert(this->trtModelStream);
            file.read(this->trtModelStream, size);
            file.close();
            std::vector<std::string> file_names;
            this->runtime = createInferRuntime(gLogger);
            assert(runtime != nullptr);
            this->engine = runtime->deserializeCudaEngine(trtModelStream, size);
            assert(engine != nullptr);
            this->context = engine->createExecutionContext();
            assert(context != nullptr);
            this->m_InputBindingIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
            delete[] trtModelStream;
            CUDA_CHECK(cudaMalloc(&buffers[0], 3 * INPUT_H * INPUT_W * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&buffers[1], OUTPUT_SIZE * sizeof(float)));
            CUDA_CHECK(cudaMalloc((void **)&d_resized_img, 3 * INPUT_H * INPUT_H * sizeof(unsigned char)));
            CUDA_CHECK(cudaMalloc((void **)&d_norm_img, sizeof(float) * INPUT_H * INPUT_H * 3));
    }
    ~yolov5()
    {
        context->destroy();
        engine->destroy();
        runtime->destroy();
        CUDA_CHECK(cudaFree(buffers[0]));
        CUDA_CHECK(cudaFree(buffers[1]));
    }
    void detect(unsigned char *d_roi_image, int roi_w, int roi_h,cv::Mat &img)
    {	
        float image_ratio = roi_w > roi_h ? float(INPUT_W) / float(roi_w) : float(INPUT_H) / float(roi_h);
        int width_out = roi_w > roi_h ? INPUT_W : (int)(roi_w * image_ratio);
        int height_out = roi_w < roi_h ? INPUT_H : (int)(roi_h * image_ratio);
        cudaMemset(d_resized_img, 0, sizeof(unsigned char) * INPUT_H * INPUT_W * 3);
        //resize 成640*640
        RGB2Resize(d_roi_image, d_resized_img, roi_w, roi_h, width_out,
                   height_out);
        RGB2Normalize(d_resized_img, d_norm_img, INPUT_W, INPUT_H);
    
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        buffers[0] = d_norm_img;
        //进行预测
        this->context->enqueue(1, this->buffers, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(this->out_put, this->buffers[1], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        std::vector<std::vector<Yolo::Detection>> batch_res(1);
        std::vector<Yolo::Detection> &res = batch_res[0];
        //nms操作 最终结果保存在 res中
        nms(res, &out_put[0], CONF_THRESH, NMS_THRESH);
        for (size_t j = 0; j < res.size(); j++)
        {   
            cv::Rect r = get_rect(roi_w, roi_h, res[j].bbox);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 1);
            cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y + 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
        cv::imshow("1", img);
        cv::waitKey(0);
    };

至此，yolov5-6,0-tensorrt 的检测接口已经检测全部完成了，那么如何进行调用呢？接下来通过一个简单的例子来展示yolov5检测接口的使用方法。

    #include <iostream>
    #include <chrono>
    #include <cmath>
    #include "cuda_utils.h"
    #include "logging.h"
    #include "common.hpp"
    #include "utils.h"
    #include "preprocess.h"
    
    //包含检测接口的头文件
    #include "yolov5-detect.h"
    
    int main(int argc, char **argv)
    {
    	
        cudaSetDevice(0);
        //模型路径
        std::string engine_name = "/home/westwell/Desktop/yolov5-5.0-tensorrt_result/engine_model/container_object.engine";
        int classNum = 80
        yolov5 *det = new yolov5(engine_name, classNum);
    
        cv::Mat img = cv::imread("");
        int w = img.cols;
        int h = img.rows;
        //将图片拷贝到GPU上
        unsigned char *d_image;
        cudaMalloc((void **)&d_image, sizeof(unsigned char) * w * h * 3);
        cudaMemcpy(d_image, img.data, w * h * 3 * sizeof(unsigned char),cudaMemcpyHostToDevice);
        //进行预测
        det->detect(d_image, w, h,img);
        cudaFree(d_image);
        return 0;
    }

预测结果展示

CMakeLists.txt 文件修改  这里也可以 将yolov5-detect.h修改成动态库的形式  ,完整请见项目中的CMakelists

```
cuda_add_library(myplugins SHARED src/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

cuda_add_library(mybasic SHARED src/basic_transform.cu)
target_link_libraries(mybasic nvinfer cudart)


#构造引擎
cuda_add_executable(build_engine src/calibrator.cpp src/build_engine.cpp src/preprocess.cu)
target_link_libraries(build_engine nvinfer cudart  myplugins  ${OpenCV_LIBS})

#预测demo
cuda_add_executable(detect  src/detect.cpp src/preprocess.cu)
target_link_libraries(detect nvinfer cudart  myplugins mybasic  ${OpenCV_LIBS})

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)
```

## 3、总结

​	本项目通过封装源代码，构造了一个 yolov5-tensorrt 的检测接口，极大简化了tenosrrt的使用步骤，同时在项目过程中，在本项目中，接触到c++ 、CUDA编程、cmake编写规则。





