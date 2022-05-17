# CV学习笔记

## 1、前言

### 这个仓库为记录在深度学习方面，主要是在CV领域的一些学习经验，方便日后进行回顾，同时希望能够给有需要的同学提供一些帮助，希望遇到问题能够及时探讨，个人微信为s1586662742,欢迎一起探讨

## 2、更新日志

### 2022-05-17: 为onnx2trt增加了CMakeLists的编译方式，需要修改在CmakeLists中本机的tensorrt和protobuf的路径即可，编译好的文件在workspace中

## 3、内容

### 3.1、cpp_learn

### 3.1.1、mulit-thread

### 这里主要记录了cpp多线程中 promise、condition_variable知识的记录

### 3.2、paddle_code

### 这里使用paddle手动实现vit和SWIN

### 3.3、warpaffine-cuda 

### 使用warpaffine加速图像的预处理

### 3.4、yolov5-6.0-TensorRT

### 这里是一个yolov5加速的仓库，可以直接应用在项目上的封装好的代码

### 3.5、yolov5-postprocess

### 这是对yolov5后处理的cpp代码，包含nms、GPU加速等。

### 3.6、onnx2trt

### 本仓库着重对于pytorh模型转onnx，再从onnx转engine，在手写AI的基础上，对常见的模型进行加速，目前已经支持的加速模型在onnx/demo文件夹内，里面含有导出onnx的方式以及利用onnxruntime进行推理。构造engine在src/main.cpp，通过修改文件名，即可进行加速，同时下面配有实现的demo，可直接使用。

