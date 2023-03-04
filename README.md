### CV学习笔记

#### 1、前言

本仓库为记录学习过程中，主要是在CV领域的一些学习经验，方便日后进行回顾，同时希望能够给有需要的同学提供一些帮助，希望遇到问题能够及时联系与探讨，如果有问题或者建议，可以+v:Rex1586662742,后面数字为qq

TensoRT代码:https://github.com/shouxieai/learning-cuda-trt

项目代码:

https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

https://www.bilibili.com/video/BV1yA411M7T4?spm_id_from=333.999.0.0

个人学习仓库地址:https://github.com/Rex-LK/tensorrt_learning

部分模型链接:https://pan.baidu.com/s/18yIsypWMg0sT_uAR_MhDSA 提取码: sh7c

#### 2、内容

2.1、demo: 开源项目

2.2、sideline_learn: 日常学习的知识点，如ros、多线程等

2.3、trt_cpp: 使用cpp进行tensorRT

2.4、trt_py: 使用python进行tensorRT

#### 3、更新日志

202211-27:更新了yolov5-7.0 的 实例分割代码

2022-07-30:增加了ros中使用yolov5&tensorRT的demo，包含两种启动方式，单节点启动以及client/server的方式，具体过程见sideline_learn/yolov5-6.0-ros/README.md

2022-07-09:增加了yolov7算法，支持cpp-tensorrt和python-tensorrt两种加速方式


2022-06-04:增加了hrnet人体关键点检测，使用cupy和torch两种后处理方式，同时在cpp中使用了gpu解码。

2022-06-1:优化了代码存放的文件夹，便于以后进行回溯和分类

2022-05-30:增加了cuda-python-tensorrt的推理方式

onnx2trt:/build_engine/batch_1/build_engine_single_image.py

推理demo，目前只支持centernet以及detr、后续会进行其他模型的支持,例如:

/centernet/centernet_infer.py.

原始项目都可以根目录下的demo文件夹找到，可以追溯到原项目地址。

2022-05-20:在onnx项目下增加了hrnet-tensorrt加速方式，预测效果与demo/HRnet作者里面的预测结果一致

2022-05-17: 为onnx2trt增加了CMakeLists的编译方式，需要修改在CmakeLists中本机的tensorrt和protobuf的路径即可，编译好的文件在workspace中，==使用cmake时使用模型和图片使用绝对路径，使用makefile时使用相对路径即可。==

...centernet、vit、unet、等均已经实现

### 

de
