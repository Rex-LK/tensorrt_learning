# CV学习笔记

## 1、前言

### 本仓库为记录学习过程中，主要是在CV领域的一些学习经验，方便日后进行回顾，同时希望能够给有需要的同学提供一些帮助，希望遇到问题能够及时探讨。

### TensoRT代码可参考：https://github.com/shouxieai/learning-cuda-trt

### 个人学习项目地址: https://github.com/Rex-LK/tensorrt_learning

### 模型链接:https://pan.baidu.com/s/18yIsypWMg0sT_uAR_MhDSA 提取码: sh7c

### 2、更新日志

### 项目代码可参考

### https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

### https://www.bilibili.com/video/BV1yA411M7T4?spm_id_from=333.999.0.0

### 2022-07-09:增加了yolov7算法，支持cpp-tensorrt和python-tensorrt两种加速方式

为了正确导出onnx需要将models/yolo.py文件中56行

```python
# x = x if self.training else (torch.cat(z, 1),x) # 修改为
x = x if self.training else (torch.cat(z, 1))
```

在detect.py中采用如下方式导出

```
    dummy = torch.zeros(1,3,640,640)
    torch.onnx.export(
        model,(dummy,),
        "yolov7.onnx",
        input_names=["image"],
        output_names=["predict"],
        opset_version=13,
        dynamic_axes={"image":{0:"batch"},"predict":{0:"batch"},} 
    )
```



### 2022-06-04:增加了hrnet人体关键点检测，使用cupy和torch两种后处理方式，同时在cpp中使用了gpu解码。

### 2022-06-1:优化了代码存放的文件夹，便于以后进行回溯和分类

### 2022-05-30:增加了cuda-python-tensorrt的推理方式

### onnx2trt:/build_engine/batch_1/build_engine_single_image.py

### 推理demo，目前只支持centernet以及detr、后续会进行其他模型的支持,例如:

### /centernet/centernet_infer.py.

### 原始项目都可以根目录下的demo文件夹找到，可以追溯到原项目地址。

### 2022-05-20:在onnx项目下增加了hrnet-tensorrt加速方式，预测效果与demo/HRnet作者里面的预测结果一致

### 2022-05-17: 为onnx2trt增加了CMakeLists的编译方式，需要修改在CmakeLists中本机的tensorrt和protobuf的路径即可，编译好的文件在workspace中，==使用cmake时使用模型和图片使用绝对路径，使用makefile时使用相对路径即可。==

## ...centernet、vit、unet、等均已经实现

## 3、内容

### 3.1 ONNX2TRT

### 基于不同的视觉项目进行tensort加速，可参考demo里面的具体项目，实现过程可参考onnx2trt里面的readme文件

### 3.2 tool_learn

### 在CV学习或者编程过程中记录的一些知识点，便于回顾。

