

# CV学习笔记之tensorrt加速

## 1、前言

### 由于显卡、算力的广泛普及，准备利用tensorrt对VIT、SWIN、DETR等一系列模型进行加速，基本做法是现将torch的模型转成onnx，然后再将onnx模型转化为engine，其中SWIN在onnx转engine时出现问题，VIT、DETR均未发现问题，其中tensorrt加速以及transformer原理可参考如下两个repo

## 手写AI:https://github.com/shouxieai/tensorRT_Pro

### 霹雳吧啦Wz:https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

### 个人学习项目地址: https://github.com/Rex-LK/deeplearning_rex

### 模型链接:链接: https://pan.baidu.com/s/1NRtkw3H1teEXsXQVQOlVIw?pwd=n46c 提取码: n46c 

## 2、export_onnx

### 在onnx2trt/demo中包含了目前已经加速成功的模型

## 2.1 vit2onnx

### 使用pth模型进行预测

```shell
python predct.py
```

### 导出模型时为了避免对出gather这类节点，可在引用shape的返回值时可进行如下操作

```python
B, C, H, W = map(int,x.shape)
```

### 之后可以导出onnx模型

```shell
python export_onnx.py 
```

### 导出onnx模型后可以检查以下onnx导出的是否正确，可以使用onnxruntim加载onnx模型进行检测

```shell
python infer-onnxruntime.py
```

### onnx推理的结果与torch模型基本一致，说明导出的onnx模型没有问题，同时可以使用netron 查看onnx模型。

## 2.2、swin2onnx

### swin的转换流程与vit基本一致，在swin中torch.roll暂不支持onnx，可以采用torch.cat进行替换

### 可以采用torch模型与onnx模型进行推理，暂不支持进行tensorrt加速。

```shell
python predct.py
python export_onnx.py 
python infer-onnxruntime.py
```

## 2.3、detr2onnx

### 由于detr暂时没有找到合适模型，采用的是torchvision中的预训练模型，其基本方法与上面一致，只是在利用 导出onnx之后，需要利用onnx-smi进行简化，或者可以对源代码修改，否则在转成engine时

```shell
python export_onnx.py 
python onnx_simplify.py 
```

## 2.4、centernet

```
python export_onnx.py 
```

## 2.5、yolov5

## 2.6、其他正在支持中

## 3、onnx2tensorrt

### 在workspace下面放入需要转化的onnx模型，vit.onnx、detr_sim.onnx等

### 在Makefile中修改本机对应的环境，可以参考手写AI提供的自动环境配置进行。

### 在src/main.cpp中提供了build_engine 和 inference的代码。直接在onnx2trt文件夹下，进行执行 make run，即可进行推理，从workspace中的结果来看，基本上与torch模型一致。

### 或者在CMakeLists中修改本机的tensorrt、protobuf路径，可以将模型路径和图片路径修改为绝对路径，执行

### 修改src/main.cpp中的demo名字

### mkdir build && cd build

### cmake .. && make -j

### 在worksapce中执行可执行文件./demo_infer 即可运行相应的demo

## 4、总结

### 本次学习课程学习了torch2onnx2trt的方式，同时复习了深度学习中的比较经典的模型，同时对一些模型的后处理采用c++的处理方式，同时也提高了c++的代码能力。



