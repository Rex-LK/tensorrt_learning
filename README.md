# CV学习笔记

## 1、前言

### 这个仓库为记录在深度学习方面，主要是在CV领域的一些学习经理，总结在这里，方便日后进行回顾，同时希望能够给有需要的同学提供一些帮助，希望遇到问题能够及时探讨。

## 2、内容

### 2.1、cpp_learn

### 2.1.1、mulit-thread

### 这里主要记录了cpp多线程中 promise、condition_variable知识的记录

### 2.2、paddle_code

### 这里使用paddle手动实现vit和SWIN

### 2.3、warpaffine-cuda 

### 使用warpaffine加速图像的预处理

### 2.4、yolov5-6.0-TensorRT

### 这里是一个yolov5加速的仓库，可以直接应用在项目上的封装好的代码

### 2.5、yolov5-postprocess

### 这是对yolov5后处理的cpp代码，包含nms、GPU加速等。

### 2.6、onnx2trt

### 本仓库着重对于pytorh模型转onnx，再从onnx转engine，在手写AI的基础上，对常见的模型进行加速，目前已经支持的加速模型在onnx/demo文件夹内，里面含有导出onnx的方式以及利用onnxruntime进行推理。构造engine在src/main.cpp，通过修改文件名，即可进行加速，同时下面配有实现的demo，可直接使用。

